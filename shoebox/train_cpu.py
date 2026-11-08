"""Resumable CPU byte training with periodic validation and atomic checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import time
import torch
from . import CorticalModel, cortical_cpu
from .checkpoint import load, save
from .cpu_run import stream, timed_step, routing_summary
from .diagnostics import PortMonitor
from .fused_kron import native_status
from .train_fineweb import atomic_json, evaluate, optimizer_for, sample, source_digest


def file_sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


@torch.no_grad()
def diagnostics(model, data, context):
    training=model.training
    model.eval()
    monitor=PortMonitor(model)
    result=model(data[:,:context],observer=monitor)
    model.train(training)
    return {**monitor.snapshot(),'routing':routing_summary(result['routing'],data.shape[0])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    start=p.add_mutually_exclusive_group()
    start.add_argument('--init',type=Path,help='Continue weights/AdamW; start new streams with empty runtime state.')
    start.add_argument('--resume',type=Path,help='Continue exactly from this run checkpoint, including streams and RNG.')
    p.add_argument('--targets',type=int,default=100000)
    p.add_argument('--validation-targets',type=int,default=4000)
    p.add_argument('--batch',type=int,default=50)
    p.add_argument('--context',type=int,default=32)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--warmup-targets',type=int,default=3200)
    p.add_argument('--checkpoint-every',type=int,default=8)
    p.add_argument('--validation-every',type=int,default=16)
    p.add_argument('--sample-bytes',type=int,default=128)
    p.add_argument('--max-updates',type=int,help='Stop this invocation early with an exact resumable checkpoint.')
    args=p.parse_args()
    if min(args.targets,args.validation_targets,args.batch,args.context,args.threads,args.workers,
           args.checkpoint_every,args.validation_every)<1 or args.lr<=0 or args.warmup_targets<0 or args.sample_bytes<0:
        p.error('Invalid training budget or runtime setting.')
    if args.max_updates is not None and args.max_updates<1:p.error('max-updates must be positive.')
    if args.out.exists():p.error('Use a new output directory, including for resume.')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    native=native_status()
    if not native['available']:raise RuntimeError(native)
    data,data_sha=stream(args.data,args.targets,args.batch)
    validation,validation_sha=stream(args.validation,args.validation_targets,args.batch)
    digest=source_digest()
    protocol={'targets':args.targets,'validation_targets':args.validation_targets,'batch':args.batch,
              'context':args.context,'threads':args.threads,'workers':args.workers,'lr':args.lr,
              'warmup_targets':args.warmup_targets,'weight_decay':.01,
              'validation_every':args.validation_every,'data_sha256':data_sha,
              'validation_sha256':validation_sha,'source_sha256':digest}
    cursor=step=base_targets=0
    elapsed=0.
    history=[];validations=[];parent=None;initial=None;best=None
    if args.resume or args.init:
        path=args.resume or args.init
        model,loaded_state,_,checkpoint=load(path)
        optimizer=optimizer_for(model,args.lr,.01)
        if checkpoint['optimizer'] is not None:optimizer.load_state_dict(checkpoint['optimizer'])
        if args.resume:
            metadata=checkpoint['metadata']
            if metadata.get('format')!='shoebox-cpu-training-v1' or metadata.get('protocol')!=protocol:
                raise ValueError('Resume protocol differs from the saved run.')
            state=loaded_state
            cursor,step=metadata['cursor'],checkpoint['step']
            old=metadata['summary']
            base_targets=old['base_trained_targets'];parent=old['initial_checkpoint']
            history=old['history'];validations=old['validation'];initial=old['initial_validation'];best=old['best_validation']
            elapsed=old['training_seconds']
            if state.batch!=args.batch or state.event!=cursor or cursor*args.batch!=old['trained_targets']:
                raise ValueError('Saved cursor, state and target count disagree.')
        else:
            prior=checkpoint['metadata'].get('summary',checkpoint['metadata'].get('protocol',{}))
            base_targets=prior.get('lifetime_trained_targets',prior.get('trained_targets',0))
            parent={'sha256':file_sha(path),'filename':path.name,'trained_targets':base_targets,
                    'initialization':'weights and optimizer continued; runtime streams reset'}
            state=model.initial_state(args.batch)
        del checkpoint,loaded_state
    else:
        cfg=cortical_cpu();torch.manual_seed(cfg.seed)
        model=CorticalModel(cfg);optimizer=optimizer_for(model,args.lr,.01)
        state=model.initial_state(args.batch)
    model.set_execution_backend('cpu').set_cpu_workers(args.workers)
    parameters=sum(p.numel() for p in model.parameters())
    args.out.mkdir(parents=True)
    atomic_json(args.out/'config.json',model.cfg.to_dict())
    if args.resume and (args.resume.parent/'best.pt').exists():
        shutil.copyfile(args.resume.parent/'best.pt',args.out/'best.pt')
    for filename,records in [('history.jsonl',history),('validation.jsonl',validations)]:
        (args.out/filename).write_text(''.join(json.dumps(r)+'\n' for r in records))
    started=time.perf_counter();last_checkpoint_targets=cursor*args.batch
    if initial is None:
        initial=evaluate(model,validation,context=args.context)
        best={'trained_targets':0,**initial}
        validations.append(best)
        with (args.out/'validation.jsonl').open('a') as f:f.write(json.dumps(best)+'\n')

    def summary(complete=False):
        return {'format':'shoebox-cpu-training-v1','complete':complete,'parameters':parameters,
                'config':model.cfg.to_dict(),'protocol':protocol,'initial_checkpoint':parent,
                'base_trained_targets':base_targets,'trained_targets':cursor*args.batch,
                'lifetime_trained_targets':base_targets+cursor*args.batch,'updates':step,
                'training_seconds':elapsed,'training_bytes_per_second':cursor*args.batch/elapsed if elapsed else 0.,
                'initial_validation':initial,'best_validation':best,'validation':validations,'history':history,
                'checkpoint_trained_targets':last_checkpoint_targets,'native':native,'torch':str(torch.__version__),
                'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                'memory_writes':[s.writes for s in state.memory.streams] if model.cfg.region_topk else state.memory.writes,
                'session_wall_seconds':time.perf_counter()-started}

    def snapshot(filename,complete=False):
        nonlocal last_checkpoint_targets
        if filename=='last.pt':last_checkpoint_targets=cursor*args.batch
        record=summary(complete)
        save(args.out/filename,model,state,optimizer,step=step,
             metadata={'format':'shoebox-cpu-training-v1','protocol':protocol,'cursor':cursor,'summary':record})
        atomic_json(args.out/'results.json',record)

    if not args.resume:snapshot('best.pt')
    snapshot('last.pt')
    print(json.dumps({'event':'start','targets':args.targets,'trained_targets':cursor*args.batch,
                      'base_trained_targets':base_targets,'batch':args.batch,'context':args.context,
                      'initial_validation':initial}),flush=True)
    starting_step=step
    while cursor<data.shape[1]-1:
        length=min(args.context,data.shape[1]-1-cursor)
        x=data[:,cursor:cursor+length];y=data[:,cursor+1:cursor+length+1]
        for group in optimizer.param_groups:
            group['lr']=args.lr*min(1.,(cursor*args.batch+x.numel())/max(1,args.warmup_targets))
        result,row=timed_step(model,optimizer,x,y,state)
        state=result['state'].detach(compact=True)
        row['routing']=routing_summary(result['routing'],args.batch)
        step+=1;cursor+=length
        row.update(step=step,trained_targets=cursor*args.batch,lr=optimizer.param_groups[0]['lr'])
        elapsed+=sum(row[k] for k in ('forward_seconds','backward_seconds','optimizer_seconds'))
        history.append(row)
        del result
        with (args.out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        complete=cursor*args.batch==args.targets
        if step%args.validation_every==0 or complete:
            record={'trained_targets':cursor*args.batch,**evaluate(model,validation,context=args.context)}
            validations.append(record)
            with (args.out/'validation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if record['nll']<best['nll']:
                best=record;snapshot('best.pt',complete)
            print(json.dumps({'event':'validation',**record}),flush=True)
        if step%args.checkpoint_every==0 or complete:snapshot('last.pt',complete)
        else:atomic_json(args.out/'results.json',summary())
        print(json.dumps({'event':'update',**{k:row[k] for k in ('step','trained_targets','nll','gradient_norm','training_bytes_per_second')},
                          'elapsed_training_seconds':elapsed}),flush=True)
        if args.max_updates is not None and step-starting_step>=args.max_updates and not complete:
            snapshot('last.pt');print(json.dumps({'event':'paused','trained_targets':cursor*args.batch}),flush=True);return
    assert cursor*args.batch==args.targets
    if source_digest()!=digest:raise RuntimeError('Training source changed during the run.')
    final=summary(True)
    final['final_validation']=validations[-1]
    final['diagnostics']=diagnostics(model,validation,args.context)
    final['samples']=[sample(model,prompt,count=args.sample_bytes) for prompt in ('The ','In the ','A ')] if args.sample_bytes else []
    final['runtime_state_bytes']=state.storage_bytes()
    final['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    final['session_wall_seconds']=time.perf_counter()-started
    atomic_json(args.out/'results.json',final)
    print(json.dumps({'event':'complete','trained_targets':args.targets,'lifetime_trained_targets':base_targets+args.targets,
                      'training_bytes_per_second':final['training_bytes_per_second'],'final_validation':validations[-1]}),flush=True)


if __name__=='__main__':main()
