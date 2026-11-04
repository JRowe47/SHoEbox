"""Learn and score compact delayed byte facts with controlled state interventions."""
import argparse
import json
import math
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F
from ..hat_minkron.train import optimizer_for
from .config import GraphConfig
from .episode_data import EpisodeStream, make_batch
from .model import RegionLanguageModel
from .learning import RegionLearningModel
from .objectives import DelayedUtility, PairedViewObjective, ViewObjectiveConfig
from .state import GraphState
from .train import learning_rate


def build_probe_model(config, kind, view_objective=False, utility=False):
    """All auxiliary learning remains opt-in; inference state is unchanged."""
    if view_objective and kind!='shared_private':
        raise ValueError('View objectives require shared/private episodes.')
    base=RegionLanguageModel(config)
    if not view_objective and not utility:return base
    view = None if not view_objective else PairedViewObjective(
        config.width, views=('A','C'), cfg=ViewObjectiveConfig(
            agreement_enabled=True, prediction_enabled=True, variance_enabled=True,
            covariance_enabled=True))
    delayed = None if not utility else DelayedUtility(
        tuple(base.connections.edges),tuple(base.nodes),config.laces,enabled=True)
    return RegionLearningModel(base,view_objective=view,utility=delayed,
                               baseline_view='A' if kind=='shared_private' else 'text')


def parameter_summary(model):
    base=model.model if isinstance(model,RegionLearningModel) else model
    return {**base.parameter_summary(),
            'parameters_including_auxiliary_heads':sum(p.numel() for p in model.parameters())}


def auxiliary_metrics(result):
    """Scalar reporting only; no retained examples or autograd graphs."""
    values={}
    if result.get('baseline_ce') is not None:values['baseline_nats']=float(result['baseline_ce'].detach())
    view=result.get('view_objective')
    if view is not None:
        for key in ('loss','agreement','prediction','variance','covariance','private','separation'):
            values['view_'+key]=float(view[key].detach())
    return values


@torch.no_grad()
def evaluate_probe(model, kind='correction', gap=16, count=512, batch_size=32, segment=16,
                   seed=0, split='validation', query_every=4, controls=None):
    if min(count,batch_size,segment,query_every)<1 or gap<0:raise ValueError('Invalid evaluation dimensions.')
    if controls is None:
        controls = ('carried','reset_query','withhold_update') if kind=='correction' else (
            ('carried','reset_query','a_only','b_only','c_only','no_private') if kind=='shared_private' else ('carried','reset_query'))
    old_mode=model.training; model.eval();results={}
    try:
        for control in controls:
            correct=old_correct=total=calls=0;ce_sum=0.;predictions=[];identities=[];types={};distances=[]
            started=time.perf_counter()
            for start in range(0,count,batch_size):
                batch=make_batch(kind,split,start,min(batch_size,count-start),seed,gap,query_every,control)
                state=None;last=None
                for pos in range(0,batch.length,segment):
                    chunk=batch.slice(pos,min(pos+segment,batch.length))
                    last=model(chunk.inputs,state=state,reset=chunk.reset,availability=chunk.availability)
                    state=last['state'];calls+=last['expert_executions']
                logits=last['logits'][:,-1];predicted=logits.argmax(-1)
                good=predicted.eq(batch.answers);correct+=int(good.sum());old_correct+=int(predicted.eq(batch.old_answers).sum())
                total+=len(predicted);ce_sum+=float(F.cross_entropy(logits,batch.answers,reduction='sum'))
                predictions.extend(predicted.tolist());identities.extend(batch.identities);distances.extend(batch.distances)
                for name,hit in zip(batch.query_types,good.tolist()):
                    row=types.setdefault(name,{'correct':0,'episodes':0});row['correct']+=int(hit);row['episodes']+=1
            for row in types.values():row['exact_accuracy']=row['correct']/row['episodes']
            results[control]={'episodes':total,'exact_correct':correct,'exact_accuracy':correct/total,
                              'old_value_accuracy':old_correct/total,'answer_nats_per_byte':ce_sum/total,
                              'answer_bits_per_byte':ce_sum/total/math.log(2),
                              'expert_executions':calls,'seconds':time.perf_counter()-started,'by_query':types,
                              'minimum_observation_distance':min(distances),'maximum_observation_distance':max(distances),
                              'predictions':predictions,'episode_ids':identities}
        return {'kind':kind,'split':split,'gap':gap,'segment':segment,'seed':seed,'query_every':query_every,
                'answer_vocabulary':'ABCD','chance_accuracy':.25,'scoring':'argmax over all 256 bytes at final query; one byte exact answer',
                'query_protocol':'Compact shaping fixture: intermediate query markers are identical in training/evaluation and all controls; evaluation ignores intermediate labels. Query answers are never fed back. This is not the longer natural-language distractor fixture.',
                'controls':results}
    finally:model.train(old_mode)


def save_probe(path,model,optimizer,state,streams,step,protocol):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    torch.save({'schema_version':2,'config':model.cfg.to_dict(),'weights':model.state_dict(),
                'learner_config':model.get_config() if isinstance(model,RegionLearningModel) else None,
                'optimizer':optimizer.state_dict(),'graph_state':None if state is None else state.to_dict(),
                'streams':streams.state_dict(),'step':step,'protocol':protocol,
                'torch_rng':torch.get_rng_state(),'python_rng':random.getstate()},temporary)
    temporary.replace(path)


def load_probe(path,streams=None,restore_rng=True):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['schema_version'] not in (1,2):raise ValueError('Unknown probe checkpoint schema.')
    learner=saved.get('learner_config')
    model=RegionLearningModel.from_config(learner) if learner is not None else RegionLanguageModel(GraphConfig.from_dict(saved['config']))
    model.load_state_dict(saved['weights'])
    optimizer=optimizer_for(model,saved['protocol']['lr']);optimizer.load_state_dict(saved['optimizer'])
    if streams is not None:streams.load_state_dict(saved['streams'])
    if restore_rng:torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng'])
    state=None if saved['graph_state'] is None else GraphState.from_dict(saved['graph_state'])
    return model,optimizer,state,saved


def warm_start_probe(path,model):
    """Load only matching weights; retain the caller's fresh RNG and stream."""
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['schema_version'] not in (1,2):raise ValueError('Unknown probe checkpoint schema.')
    def signature(config,learner):
        config=GraphConfig.from_dict(config).to_dict();config.pop('seed',None)
        learner=None if learner is None else {key:value for key,value in learner.items() if key!='model'}
        return config,learner
    current=model.get_config() if isinstance(model,RegionLearningModel) else None
    if signature(saved['config'],saved.get('learner_config'))!=signature(model.cfg.to_dict(),current):
        raise ValueError('Warm start requires the same architecture and auxiliary-learning configuration; a new seed is allowed.')
    model.load_state_dict(saved['weights'])
    # Utility buffers encode learned associations and remain part of model
    # weights. The optimizer, graph state, episode cursor and RNG are fresh.
    return {'checkpoint_step':saved['step'],'source':str(Path(path))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=('recall','correction','shared_private'),default='correction')
    p.add_argument('--steps',type=int,default=100);p.add_argument('--stop-after',type=int)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--segment',type=int,default=16)
    p.add_argument('--gap',type=int,default=16);p.add_argument('--query-every',type=int,default=4)
    p.add_argument('--rounds',type=int,default=1);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--lr',type=float,default=.002);p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--eval-episodes',type=int,default=512);p.add_argument('--eval-batch',type=int,default=32)
    p.add_argument('--eval-gap',type=int);p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--eval-split',choices=('validation','test'),default='validation')
    p.add_argument('--view-objective',action='store_true',help='A/C shared-code agreement, prediction, variance and covariance losses; shared_private only.')
    p.add_argument('--utility',action='store_true',help='Train a source-only baseline and explicitly commit eligible utility after each optimizer step.')
    p.add_argument('--full',action='store_true')
    checkpoint_mode=p.add_mutually_exclusive_group()
    checkpoint_mode.add_argument('--resume',type=Path)
    checkpoint_mode.add_argument('--warm-start',type=Path)
    p.add_argument('--checkpoint-every',type=int,default=25)
    p.add_argument('--out',type=Path,default=Path('artifacts/probes'))
    args=p.parse_args()
    if min(args.steps,args.batch,args.segment,args.query_every,args.rounds,args.threads,args.eval_episodes,args.eval_batch,args.log_every,args.checkpoint_every)<1 or args.gap<0 or args.lr<=0 or (args.eval_gap is not None and args.eval_gap<0):p.error('Invalid positive count, gap, or learning rate.')
    if args.stop_after is not None and not 1<=args.stop_after<=args.steps:p.error('Invalid stopping step.')
    if args.view_objective and args.kind!='shared_private':p.error('--view-objective requires --kind shared_private.')
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);random.seed(args.seed)
    protocol={k:v for k,v in vars(args).items() if k not in ('out','resume','warm_start','checkpoint_every','stop_after','eval_episodes','eval_batch','eval_gap','eval_split','log_every')}
    protocol['supervision']='Only query-marker outputs. Local query checkpoints train writes/retention; query answers are never fed back as observations. Fact values are observed in the input.'
    protocol['difficulty']='one named fact; four values; one-byte exact answer' if args.kind!='shared_private' else 'one shared value, dedicated A/C private values; random shared/left/right query'
    streams=EpisodeStream(args.kind,args.batch,args.gap,args.query_every,args.seed)
    state=None;start=0
    if args.resume:
        model,optimizer,state,saved=load_probe(args.resume,streams)
        schedule = lambda data: {k:v for k,v in {'view_objective':False,'utility':False,**data}.items() if k not in ('supervision','difficulty','warm_start')}
        if schedule(saved['protocol'])!=schedule(protocol):raise ValueError('Resume requires an identical training protocol and schedule horizon.')
        if 'warm_start' in saved['protocol']:protocol['warm_start']=saved['protocol']['warm_start']
        start=saved['step']
    else:
        factory=GraphConfig if args.full else GraphConfig.tiny
        model=build_probe_model(factory(context=args.segment,rounds=args.rounds,recurrent_feedback=args.rounds>1,seed=args.seed),args.kind,args.view_objective,args.utility)
        if args.warm_start:protocol['warm_start']=warm_start_probe(args.warm_start,model)
        optimizer=optimizer_for(model,args.lr)
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    learner_config=model.get_config() if isinstance(model,RegionLearningModel) else None
    (out/'protocol.json').write_text(json.dumps({**protocol,'config':model.cfg.to_dict(),'learner_config':learner_config,'parameters':parameter_summary(model)},indent=2))
    seen=calls=supervised=queries=hits=optimizer_steps=0;losses=[];seconds=0.;stop=args.stop_after or args.steps
    auxiliary_sums={};auxiliary_counts={}
    if stop<=start:raise ValueError('Checkpoint already reached requested stop.')
    with (out/'history.jsonl').open('a' if args.resume else 'w') as log:
        for step in range(start+1,stop+1):
            model.train();chunk=streams.next_chunk(args.segment);optimizer.zero_grad(set_to_none=True)
            mask=chunk.targets!=-100
            for group in optimizer.param_groups:group['lr']=learning_rate(step,args.steps,args.lr)
            tick=time.perf_counter()
            # Entirely unlabeled chunks still carry state and run all experts.
            # An explicitly enabled view objective can train them without CE.
            result=model(chunk.inputs,chunk.targets if bool(mask.any()) else None,state=state,reset=chunk.reset,availability=chunk.availability)
            if bool(mask.any()) or args.view_objective:
                if not bool(torch.isfinite(result['loss'])):raise RuntimeError('Nonfinite probe loss.')
                result['loss'].backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not bool(torch.isfinite(norm)):raise RuntimeError('Nonfinite probe gradient.')
                optimizer.step();optimizer_steps+=1
                if isinstance(model,RegionLearningModel):model.commit_feedback(result.get('utility_packet'))
                if bool(mask.any()):losses.append(float(result['ce'].detach()));supervised+=int(mask.sum())
            for key,value in auxiliary_metrics(result).items():
                auxiliary_sums[key]=auxiliary_sums.get(key,0.)+value
                auxiliary_counts[key]=auxiliary_counts.get(key,0)+1
            state=result['state'].detach();seconds+=time.perf_counter()-tick;seen+=result['external_events'];calls+=result['expert_executions']
            if bool(chunk.final.any()):
                pred=result['logits'].detach().argmax(-1);hits+=int((pred[chunk.final]==chunk.targets[chunk.final]).sum());queries+=int(chunk.final.sum())
            if step%args.log_every==0 or step==stop:
                row={'step':step,'query_nats_recent':sum(losses)/len(losses) if losses else None,'training_final_accuracy':hits/queries if queries else None,
                     'final_queries':queries,'supervised_queries':supervised,'optimizer_steps_this_invocation':optimizer_steps,'external_events_this_invocation':seen,'expert_executions_this_invocation':calls,
                     'training_seconds_this_invocation':seconds,'state_bytes':state.storage_bytes(),
                     'auxiliary_means_this_invocation':{key:value/auxiliary_counts[key] for key,value in auxiliary_sums.items()},
                     'utility_feedback_updates':int(model.utility.feedback_updates) if isinstance(model,RegionLearningModel) and model.utility is not None else 0}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True);losses=[]
            del result
            if step%args.checkpoint_every==0:
                save_probe(out/'checkpoint.pt',model,optimizer,state,streams,step,protocol)
    save_probe(out/'checkpoint.pt',model,optimizer,state,streams,stop,protocol)
    evaluation=evaluate_probe(model,args.kind,args.gap if args.eval_gap is None else args.eval_gap,args.eval_episodes,args.eval_batch,args.segment,args.seed,split=args.eval_split,query_every=args.query_every)
    (out/'evaluation.json').write_text(json.dumps(evaluation,indent=2))
    summary={'step':stop,'training_seconds_this_invocation':seconds,'external_events_this_invocation':seen,'expert_executions_this_invocation':calls,
             'supervised_queries':supervised,'optimizer_steps_this_invocation':optimizer_steps,'training_final_queries':queries,'training_final_accuracy':hits/queries if queries else None,
             'evaluation':{k:{key:value for key,value in v.items() if key not in ('predictions','episode_ids')} for k,v in evaluation['controls'].items()},
             'parameters':parameter_summary(model),
             'auxiliary_means_this_invocation':{key:value/auxiliary_counts[key] for key,value in auxiliary_sums.items()},
             'utility_feedback_updates':int(model.utility.feedback_updates) if isinstance(model,RegionLearningModel) and model.utility is not None else 0}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
