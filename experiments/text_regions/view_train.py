"""Matched-initialization shared/private training with independent learning switches."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import resource
import time
import torch
from ..hat_minkron.train import optimizer_for
from .config import GraphConfig
from .learning import RegionLearningModel
from .model import RegionLanguageModel
from .objectives import DelayedUtility, PairedViewObjective, ViewObjectiveConfig
from .probe_train import auxiliary_metrics, parameter_summary
from .state import GraphState
from .train import learning_rate
from .view_data import ViewStream


def tensor_fingerprint(model):
    """Hash tensor identity and values, independent of the enabled loss switches."""
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        value=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        # Hash the exact tensor bytes without an optional NumPy installation.
        digest.update(bytes(value.reshape(-1).view(torch.uint8).tolist()))
    return digest.hexdigest()


def build_learner(cfg, view_enabled=False, utility_enabled=False, base_checkpoint=None, baseline_detach_features=False):
    """All four variants allocate the same heads and train the same baseline.

    Set the initialization RNG before calling. Switches do not change random
    draws, parameter count, or initial tensor values. A bare graph checkpoint
    may initialize the graph; runtime memory and optimizer are always fresh.
    """
    base=RegionLanguageModel(cfg)
    if base_checkpoint is not None:
        saved=torch.load(base_checkpoint,map_location='cpu',weights_only=True)
        source=GraphConfig.from_dict(saved['config'])
        expected=cfg.to_dict();actual=source.to_dict()
        for key in ('seed','evidence_gated_ports'):
            expected.pop(key);actual.pop(key)
        if expected!=actual or saved.get('learner_config') is not None:
            raise ValueError('Initializer must be a bare graph with matching architecture.')
        base.load_state_dict(saved['weights'])
    options=ViewObjectiveConfig(agreement_enabled=view_enabled,prediction_enabled=view_enabled,
                               variance_enabled=view_enabled,covariance_enabled=view_enabled,
                               private_enabled=view_enabled,separation_enabled=view_enabled,
                               private_weight=.05)
    views=PairedViewObjective(cfg.width,views=('A','C'),cfg=options)
    utility=DelayedUtility(tuple(base.connections.edges),tuple(base.nodes),cfg.laces,
                           enabled=utility_enabled)
    return RegionLearningModel(base,view_objective=views,utility=utility,
                               baseline_enabled=True,baseline_view='A',baseline_weight=.1,
                               baseline_detach_features=baseline_detach_features)


def train_step(model,optimizer,batch,state=None):
    model.train();optimizer.zero_grad(set_to_none=True)
    started=time.perf_counter()
    result=model(batch.inputs,batch.targets,state=state,reset=batch.reset,
                 availability=batch.availability,private_targets=batch.pooled_private_targets())
    has_labels=bool((batch.targets!=-100).any())
    view_on=any(value for name,value in vars(model.view_objective.cfg).items() if name.endswith('_enabled'))
    update=has_labels or view_on
    norm=0.
    if update:
        if not bool(torch.isfinite(result['loss'])):raise RuntimeError('Nonfinite view-training loss.')
        result['loss'].backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not bool(torch.isfinite(norm)):raise RuntimeError('Nonfinite view-training gradient.')
        optimizer.step()
        model.commit_feedback(result.get('utility_packet'))
    state=result['state'].detach();prediction=result['logits'].detach().argmax(-1)
    by_query={}
    for row,kind in enumerate(batch.query_types):
        selected=batch.final[row]
        entry=by_query.setdefault(kind,{'correct':0,'count':0})
        entry['correct']+=int((prediction[row,selected]==batch.targets[row,selected]).sum())
        entry['count']+=int(selected.sum())
    metrics={'loss':float(result['loss'].detach()),'ce':float(result['ce'].detach()),
             'gradient_norm':float(norm),'optimizer_steps':int(update),
             'external_events':result['external_events'],'source_occurrences':result['source_occurrences'],
             'expert_executions':result['expert_executions'],'seconds':time.perf_counter()-started,
             'state_bytes':state.storage_bytes(),'by_query':by_query,
             'auxiliary':auxiliary_metrics(result),'utility_updates':int(model.utility.feedback_updates)}
    return state,metrics,result['logits'].detach()


def save_view_checkpoint(path,model,optimizer,state,stream,step,protocol,totals):
    protocol={'baseline_detach_features':model.baseline_detach_features,**protocol}
    if protocol['baseline_detach_features'] != model.baseline_detach_features:
        raise ValueError('Checkpoint protocol and baseline detachment configuration differ.')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save({'view_checkpoint_schema':1,'learner_config':model.get_config(),
                'weights':model.state_dict(),'optimizer':optimizer.state_dict(),
                'graph_state':None if state is None else state.to_dict(),'stream':stream.state_dict(),
                'step':step,'protocol':protocol,'totals':totals,
                'torch_rng':torch.get_rng_state(),'python_rng':random.getstate()},temporary)
    temporary.replace(path)


def load_view_checkpoint(path,stream=None,restore_rng=True):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('view_checkpoint_schema')!=1:raise ValueError('Unknown view-checkpoint schema.')
    model=RegionLearningModel.from_config(saved['learner_config']);model.load_state_dict(saved['weights'])
    saved['protocol']={'baseline_detach_features':False,**saved['protocol']}
    if saved['protocol']['baseline_detach_features'] != model.baseline_detach_features:
        raise ValueError('Checkpoint protocol and baseline detachment configuration differ.')
    optimizer=optimizer_for(model,saved['protocol']['lr']);optimizer.load_state_dict(saved['optimizer'])
    if stream is not None:stream.load_state_dict(saved['stream'])
    if restore_rng:torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng'])
    state=None if saved['graph_state'] is None else GraphState.from_dict(saved['graph_state'])
    return model,optimizer,state,saved


def initial_totals():
    return {'external_events':0,'source_occurrences':0,'expert_executions':0,'optimizer_steps':0,
            'seconds':0.,'by_query':{},'auxiliary_sum':{},'segments':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=240);parser.add_argument('--stop-after',type=int)
    parser.add_argument('--batch',type=int,default=32);parser.add_argument('--segment',type=int,default=16)
    parser.add_argument('--gap',type=int,default=8);parser.add_argument('--query-every',type=int,default=4)
    parser.add_argument('--dropout',type=float,default=.25)
    parser.add_argument('--seed',type=int,default=20261106);parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--lr',type=float,default=.002);parser.add_argument('--rounds',type=int,default=1)
    parser.add_argument('--view-objective',action='store_true');parser.add_argument('--utility',action='store_true')
    parser.add_argument('--baseline-detach-features',action='store_true',
                        help='Train the baseline head without sending its auxiliary gradient into the source encoder.')
    parser.add_argument('--evidence-gated-ports',action='store_true',help='Suppress publications until a node consumes valid evidence; keep all expert executions.')
    parser.add_argument('--base-checkpoint',type=Path);parser.add_argument('--resume',type=Path)
    parser.add_argument('--checkpoint-every',type=int,default=20);parser.add_argument('--log-every',type=int,default=20)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if min(args.steps,args.batch,args.segment,args.query_every,args.threads,args.rounds,args.checkpoint_every,args.log_every)<1 or args.gap<0 or args.lr<=0 or not 0<=args.dropout<=1:
        parser.error('Invalid count, gap, learning rate, or dropout.')
    if args.stop_after is not None and not 1<=args.stop_after<=args.steps:parser.error('Invalid stopping point.')
    if args.resume is not None and args.base_checkpoint is not None:parser.error('Use either exact resume or a new initializer.')
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);random.seed(args.seed)
    protocol={key:value for key,value in vars(args).items()
              if key not in ('out','resume','base_checkpoint','stop_after','checkpoint_every','log_every')}
    protocol['baseline_all_variants']=True
    stream=ViewStream(batch_size=args.batch,gap=args.gap,query_every=args.query_every,
                      seed=args.seed,dropout=args.dropout)
    state=None;start=0;totals=initial_totals()
    if args.resume:
        model,optimizer,state,saved=load_view_checkpoint(args.resume,stream)
        previous={key:value for key,value in {'evidence_gated_ports':False,'baseline_detach_features':False,**saved['protocol']}.items() if key not in ('initial_tensor_sha256','base_initializer')}
        if previous!=protocol:raise ValueError('Exact resume requires the same training protocol and horizon.')
        protocol=saved['protocol'];start=saved['step'];totals=saved['totals']
    else:
        cfg=GraphConfig.tiny(context=args.segment,rounds=args.rounds,recurrent_feedback=args.rounds>1,seed=args.seed,
                             evidence_gated_ports=args.evidence_gated_ports)
        model=build_learner(cfg,args.view_objective,args.utility,args.base_checkpoint,
                            baseline_detach_features=args.baseline_detach_features)
        protocol['initial_tensor_sha256']=tensor_fingerprint(model)
        protocol['base_initializer']=None if args.base_checkpoint is None else {
            'path':str(args.base_checkpoint),'sha256':hashlib.sha256(args.base_checkpoint.read_bytes()).hexdigest()}
        optimizer=optimizer_for(model,args.lr)
    stop=args.stop_after or args.steps
    if stop<=start:raise ValueError('Checkpoint already reached the requested stopping point.')
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    (out/'protocol.json').write_text(json.dumps({**protocol,'learner_config':model.get_config(),
                                               'parameters':parameter_summary(model),'stream':stream.state_dict()['spec']},indent=2)+'\n')
    # A fresh initializer is immediately recoverable, before any expensive pass.
    if not args.resume:save_view_checkpoint(out/'checkpoint.pt',model,optimizer,state,stream,0,protocol,totals)
    recent=[]
    with (out/'history.jsonl').open('a' if args.resume else 'w') as log:
        for step in range(start+1,stop+1):
            for group in optimizer.param_groups:group['lr']=learning_rate(step,args.steps,args.lr)
            state,metrics,_=train_step(model,optimizer,stream.next_chunk(args.segment),state)
            for key in ('external_events','source_occurrences','expert_executions','optimizer_steps','seconds'):
                totals[key]+=metrics[key]
            totals['segments']+=1;recent.append(metrics['ce'])
            for name,row in metrics['by_query'].items():
                entry=totals['by_query'].setdefault(name,{'correct':0,'count':0})
                for key in entry:entry[key]+=row[key]
            for name,value in metrics['auxiliary'].items():totals['auxiliary_sum'][name]=totals['auxiliary_sum'].get(name,0.)+value
            if step%args.log_every==0 or step==stop:
                row={'step':step,'recent_query_nats':sum(recent)/len(recent),'gradient_norm':metrics['gradient_norm'],
                     'state_bytes':metrics['state_bytes'],'training':totals,
                     'utility_updates':int(model.utility.feedback_updates),
                     'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True);recent=[]
            if step%args.checkpoint_every==0 or step==stop:
                save_view_checkpoint(out/'checkpoint.pt',model,optimizer,state,stream,step,protocol,totals)
    summary={'step':stop,'training':totals,'parameters':parameter_summary(model),
             'final_tensor_sha256':tensor_fingerprint(model),'initial_tensor_sha256':protocol['initial_tensor_sha256'],
             'state_bytes':state.storage_bytes(),'utility_updates':int(model.utility.feedback_updates),
             'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
