"""Joint next-byte and fresh memory-episode training of one eleven-node graph."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import torch
from torch.nn import functional as F
from . import byte_stream
from .composition_train import atomic_save,source_hashes
from .config import GraphConfig
from .data import OrderedByteStreams,split_bytes
from .model import RegionLanguageModel
from .multifact import backward,table_batches,trainable,COUNTERS
from .multifact_data import make_batch,KINDS,OWNERS
from .multifact_eval import validation
from .retained_composition import teacher_inputs
from .state import GraphState
from .train import optimizer_bytes


def chunk_digest(chunk):
    digest=hashlib.sha256()
    for tensor in (chunk.values,chunk.targets,chunk.reset):digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_for(model,core_lr=.0003,readout_lr=.002,producer_lr=.0001):
    groups=dict(core=[],readout=[],producer=[])
    for name,p in model.named_parameters():
        # Preserve the established memory-edge context/gate interface.
        memory_edge=name.startswith('connections.edges.') and any('.'+o+'__' in name for o in OWNERS)
        p.requires_grad_(not memory_edge or trainable(name))
        if p.requires_grad:
            group='producer' if trainable(name) else ('readout' if name.startswith('readout.') else 'core')
            groups[group].append(p)
    return torch.optim.AdamW([dict(params=groups[k],lr=lr,name=k) for k,lr in
        (('core',core_lr),('readout',readout_lr),('producer',producer_lr))],betas=(.9,.95),weight_decay=.01)


@torch.no_grad()
def evaluate_text(model,raw,batch=8,context=16,chunks=8,rounds=2,feedback=True,reset_chunks=False):
    streams=OrderedByteStreams.from_bytes(raw,batch);state=None
    rows=[];nll=0.;correct=count=0;costs=dict.fromkeys(COUNTERS,0)
    was_training=model.training;model.eval();tick=time.perf_counter()
    try:
        for _ in range(chunks):
            chunk=streams.next_chunk(context)
            r=byte_stream.forward(model,chunk.values,chunk.targets,state=None if reset_chunks else state,
                reset=chunk.reset,rounds=rounds,feedback=feedback)
            logits=r['logits'];pred=logits.argmax(-1)
            losses=F.cross_entropy(logits.reshape(-1,256),chunk.targets.reshape(-1),reduction='none').reshape_as(chunk.targets)
            rows.append(dict(inputs=chunk.values.tolist(),targets=chunk.targets.tolist(),predictions=pred.tolist(),
                nll=losses.tolist(),reset=chunk.reset.tolist()))
            nll+=float(losses.double().sum());correct+=int(pred.eq(chunk.targets).sum());count+=chunk.targets.numel()
            state=r['state'].detach()
            for k in costs:costs[k]+=r[k]
        return dict(nats=nll/count,bits=nll/count/math.log(2),accuracy=correct/count,bytes=count,
            state_bytes=state.storage_bytes(),seconds=time.perf_counter()-tick,rounds=rounds,feedback=feedback,
            reset_chunks=reset_chunks,batch=batch,context=context,chunks=chunks,rows=rows,**costs)
    finally:model.train(was_training)


def selection(text,scores):
    accuracy=[f['accuracy'] for s in scores for f in s['fields']]
    floor=min(accuracy)
    # Accuracy is a guard. Among eligible checkpoints language validation selects.
    return (int(floor>=.9),-text['nats'] if floor>=.9 else floor, -text['nats'])


def update(model,opt,chunk,state,batch,tables,reference,memory_weight=1.):
    opt.zero_grad(set_to_none=True)
    text=byte_stream.forward(model,chunk.values,chunk.targets,state,chunk.reset)
    text['loss'].backward();state=text['state'].detach()
    metrics=dict(text_loss=float(text['loss'].detach()),text_nats=float(text['ce'].detach()),
        text_bytes=chunk.values.numel(),text_batch_sha256=chunk_digest(chunk),**{k:text[k] for k in COUNTERS})
    del text
    memory=backward(model,batch,tables,memory_weight,reference)
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    opt.step()
    for k in COUNTERS:metrics[k]+=memory[k]
    metrics.update(memory=memory,gradient_norm=float(norm))
    return state,metrics


def load(path,streams=None):
    s=torch.load(path,map_location='cpu',weights_only=True)
    if s['schema']!='shoebox-joint-text-v1':raise ValueError('Unknown joint-text checkpoint.')
    model=RegionLanguageModel(GraphConfig.from_dict(s['config']));model.load_state_dict(s['weights'])
    p=s['protocol'];opt=optimizer_for(model,p['core_lr'],p['readout_lr'],p['producer_lr']);opt.load_state_dict(s['optimizer'])
    if streams is not None:streams.load_state_dict(s['streams'])
    torch.set_rng_state(s['torch_rng'])
    state=GraphState.from_dict(s['graph_state']) if s['graph_state'] is not None else None
    return model,opt,state,s


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--steps',type=int,default=120);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--context',type=int,default=16);p.add_argument('--memory-batch',type=int,default=16)
    p.add_argument('--core-lr',type=float,default=.0003);p.add_argument('--readout-lr',type=float,default=.002)
    p.add_argument('--producer-lr',type=float,default=.0001);p.add_argument('--memory-weight',type=float,default=1.)
    p.add_argument('--validate-every',type=int,default=20);p.add_argument('--validation-count',type=int,default=64)
    p.add_argument('--validation-chunks',type=int,default=8);p.add_argument('--stop-after',type=int)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    if min(a.steps,a.batch,a.context,a.memory_batch,a.validate_every,a.validation_count,a.validation_chunks)<1:p.error('Positive dimensions required.')
    if any(not math.isfinite(v) or v<=0 for v in (a.core_lr,a.readout_lr,a.producer_lr,a.memory_weight)):p.error('Finite positive rates/weights required.')
    if a.stop_after is not None and not 1<=a.stop_after<=a.steps:p.error('Invalid stop step.')
    raw=a.data.read_bytes();train,val,_=split_bytes(raw);streams=OrderedByteStreams.from_bytes(train,a.batch)
    protocol={k:v for k,v in vars(a).items() if k not in ('parent','data','out','resume','stop_after')}
    protocol.update(parent_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(raw).hexdigest(),data_bytes=len(raw),train_bytes=len(train),validation_bytes=len(val),
        source_hashes=source_hashes(),gaps=[0,4,8,16],segment=16,rounds=2,
        selection='Minimum memory validation accuracy >= 90%, then lower text NLL; update zero eligible',
        objective='Next-byte CE + routing losses + weighted fresh multifact objective; one optimizer step',
        alignment='Four fixed continuous ingress targets captured from the parent before joint training',
        text_interface='Same bytes in A/B/C, three separately counted HAT encodings, memory producers/edges inactive',
        source_interface='Original explicit shared/left/right field address and latest-replica selection',
        optimizer='AdamW beta=(.9,.95), weight decay .01, global clip 1; linear five-update warmup')
    a.out.mkdir(parents=True,exist_ok=True)
    if a.resume:
        model,opt,state,s=load(a.out/'last.pt',streams)
        if s['protocol']!=protocol:raise ValueError('Resume requires identical protocol and source.')
        step,history,best,totals,reference=s['step'],s['history'],s['best'],s['totals'],s['reference']
    else:
        if (a.out/'last.pt').exists():raise ValueError('Use --resume or a new output directory.')
        torch.manual_seed(a.seed)
        parent=torch.load(a.parent,map_location='cpu',weights_only=True)
        model=RegionLanguageModel(GraphConfig.from_dict(parent['config']));model.load_state_dict(parent['weights'])
        opt=optimizer_for(model,a.core_lr,a.readout_lr,a.producer_lr)
        reference=teacher_inputs(model,torch.arange(65,69)).clone()
        step=0;state=None;history=[];best=[-1.,-1e30,-1e30]
        totals=dict.fromkeys((*COUNTERS,'text_bytes','training_seconds'),0)
    def save(name):
        atomic_save(a.out/name,dict(schema='shoebox-joint-text-v1',config=model.cfg.to_dict(),weights=model.state_dict(),
            optimizer=opt.state_dict(),graph_state=state.to_dict() if state is not None else None,
            streams=streams.state_dict(),torch_rng=torch.get_rng_state(),reference=reference,protocol=protocol,
            step=step,history=history,best=best,totals=totals))
    def validate(row):
        text=evaluate_text(model,val,a.batch,a.context,a.validation_chunks)
        key,scores=validation(model,a.seed,a.validation_count)
        row.update(text_validation={k:v for k,v in text.items() if k!='rows'},memory_validation=scores)
        return list(selection(text,scores)),key[0]
    if not a.resume:
        row=dict(step=0);best,floor=validate(row);history.append(row)
        save('initial.pt');save('best.pt');save('last.pt')
        print(json.dumps(dict(step=0,text_nats=row['text_validation']['nats'],memory_min=floor,parameters=model.parameter_summary())),flush=True)
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for step in range(step+1,(a.stop_after or a.steps)+1):
        model.train();kind=KINDS[(step-1)%5];gap=protocol['gaps'][((step-1)//5)%4];offset=(step-1)*a.memory_batch
        batch=make_batch('train',offset,a.memory_batch,a.seed,gap,kind);tb=table_batches(batch,'train',offset,a.seed)
        chunk=streams.next_chunk(a.context)
        for group,lr in zip(opt.param_groups,(a.core_lr,a.readout_lr,a.producer_lr)):group['lr']=lr*min(1.,step/5)
        tick=time.perf_counter();state,row=update(model,opt,chunk,state,batch,tb,reference,a.memory_weight)
        row.update(step=step,kind=kind,gap=gap,training_seconds=time.perf_counter()-tick,
            state_bytes=state.storage_bytes(),optimizer_bytes=optimizer_bytes(opt),
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        history.append(row)
        for k in totals:totals[k]+=row[k]
        if step%a.validate_every==0 or step==a.steps:
            candidate,floor=validate(row)
            if tuple(candidate)>tuple(best):best=candidate;save('best.pt')
            print(json.dumps(dict(step=step,text_nats=row['text_validation']['nats'],memory_min=floor,best=best)),flush=True)
        if step==1 or step%10==0:print(json.dumps({k:v for k,v in row.items() if k not in ('memory','memory_validation','text_validation')}),flush=True)
        if step%10==0 or step==(a.stop_after or a.steps):
            save('last.pt');(a.out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(dict(complete=step==a.steps,step=step,best=best,**totals)),flush=True)


if __name__=='__main__':main()
