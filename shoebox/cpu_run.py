"""CPU sparse graph timing and one-pass FineWeb shakedowns.

Example: python -m shoebox.cpu_run --mode train --data data/cortical_fineweb/train.bin
  --validation data/cortical_fineweb/validation.bin --out artifacts/cpu_run
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import time
import torch
from . import CorticalModel, cortical_wide, cortical_cpu
from .checkpoint import save
from .config import NAMES
from .fused_kron import native_status
from .train_fineweb import atomic_json, optimizer_for, source_digest, evaluate


def stream(path, targets, batch):
    raw = path.read_bytes()[:targets+batch]
    if targets % batch or len(raw) != targets+batch:
        raise ValueError('Need a divisible target budget and one prefix byte per stream.')
    return torch.tensor(list(raw),dtype=torch.long).reshape(batch,-1), hashlib.sha256(raw).hexdigest()


def routing_summary(records, batch):
    if not records:
        return None
    count = len(records)*batch
    selections = torch.stack([r['selected'].cpu() for r in records])
    p = torch.stack([r['probabilities'].cpu() for r in records])
    return {'region_evaluations_per_byte':sum(sum(r['region_rows'].values()) for r in records)/count,
            'hat_byte_steps_per_byte':sum(r['hat_byte_steps'] for r in records)/count,
            'cohort_calls':sum(r['region_calls'] for r in records),
            'blocked_region_rows':sum(r['blocked_rows'] for r in records),
            'selected_fraction':dict(zip(NAMES,selections.float().mean((0,1)).tolist())),
            'executed_rows':{n:sum(r['region_rows'][n] for r in records) for n in NAMES},
            'route_patterns':len(set(tuple(row) for row in selections.flatten(0,1).tolist())),
            'probability_entropy_nats':float(-(p*p.clamp_min(1e-20).log()).sum(-1).mean()),
            'selection_change_fraction':float((selections[1:]!=selections[:-1]).any(-1).float().mean()) if len(records)>1 else 0.}


def timed_step(model, optimizer, x, y, state, update=True):
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    result = model(x,targets=y,state=state)
    forward = time.perf_counter()-start
    start = time.perf_counter()
    result['loss'].backward()
    backward = time.perf_counter()-start
    start = time.perf_counter()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=True)
    if update:
        optimizer.step()
    optimizer_seconds = time.perf_counter()-start
    return result, {'nll':float(result['ce'].detach()),'gradient_norm':float(norm),
                    'forward_seconds':forward,'backward_seconds':backward,
                    'optimizer_seconds':optimizer_seconds,
                    'training_bytes_per_second':x.numel()/(forward+backward+optimizer_seconds),
                    'forward_backward_bytes_per_second':x.numel()/(forward+backward)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('benchmark','train'),required=True)
    p.add_argument('--preset',choices=('wide','cpu'),default='cpu')
    p.add_argument('--backend',choices=('fused','cpu'),default='cpu')
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--validation',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--targets',type=int,default=2048)
    p.add_argument('--validation-targets',type=int,default=256)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--context',type=int,default=32)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--region-topk',type=int,default=3)
    p.add_argument('--edge-topk',type=int,default=2)
    p.add_argument('--lr',type=float,default=.001)
    args=p.parse_args()
    if min(args.targets,args.validation_targets,args.batch,args.context,args.threads,args.workers,args.repeats)<1 or args.lr<=0:
        p.error('Counts and learning rate must be positive.')
    if args.workers>1 and args.backend!='cpu':p.error('Regional workers require the CPU backend.')
    if args.out.exists():p.error('Choose a new output directory.')
    if args.mode=='train' and args.validation is None:p.error('Training requires a disjoint validation file.')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    kernel=native_status()
    if not kernel['available']:raise RuntimeError(kernel)
    cfg=cortical_cpu(region_topk=args.region_topk,edge_topk=args.edge_topk) if args.preset=='cpu' else cortical_wide()
    torch.manual_seed(cfg.seed)
    model=CorticalModel(cfg).set_execution_backend(args.backend).set_cpu_workers(args.workers)
    optimizer=optimizer_for(model,args.lr,.01)
    data,checksum=stream(args.data,args.targets,args.batch)
    args.out.mkdir(parents=True)
    summary={'format':'shoebox-cpu-overhaul-v1','complete':False,'mode':args.mode,
             'preset':args.preset,'backend':args.backend,'threads':args.threads,'workers':args.workers,
             'batch':args.batch,'context':args.context,'parameters':sum(p.numel() for p in model.parameters()),
             'config':cfg.to_dict(),'data_sha256':checksum,'source_sha256':source_digest(),
             'torch':str(torch.__version__),'python':platform.python_version(),'native':kernel}
    atomic_json(args.out/'config.json',cfg.to_dict())
    if args.mode=='benchmark':
        if data.shape[1]<9+args.context:raise ValueError('Benchmark needs eight warmup observations plus a full horizon.')
        with torch.no_grad():state=model(data[:,:8])['state'].detach(compact=True)
        x,y=data[:,8:8+args.context],data[:,9:9+args.context]
        rows=[]
        for repeat in range(args.repeats):
            torch.manual_seed(cfg.seed+1)
            result,row=timed_step(model,optimizer,x,y,state,update=False)
            row['routing']=routing_summary(result['routing'],args.batch)
            # This benchmark excludes parameter updates; the optimizer field is
            # just gradient clipping. End-to-end training is measured separately.
            row.pop('training_bytes_per_second')
            row['clip_seconds']=row.pop('optimizer_seconds')
            rows.append(row)
            print(json.dumps({'repeat':repeat+1,'bytes_per_second':row['forward_backward_bytes_per_second'],
                              'forward_seconds':row['forward_seconds'],'backward_seconds':row['backward_seconds']}),flush=True)
            del result
            gc.collect()
        summary.update(runs=rows,target_positions=x.numel(),
                       median_bytes_per_second=statistics.median(r['forward_backward_bytes_per_second'] for r in rows))
        model.eval()
        with torch.no_grad():
            # Reset state so inference has the same warmup/data horizon.
            state=model(data[:,:8])['state'].detach(compact=True)
            start=time.perf_counter();result=model(x,state=state);seconds=time.perf_counter()-start
        summary['inference']={'bytes_per_second':x.numel()/seconds,'seconds':seconds,
                              'routing':routing_summary(result['routing'],args.batch)}
    else:
        validation,vchecksum=stream(args.validation,args.validation_targets,args.batch)
        summary['validation_sha256']=vchecksum
        summary['initial_validation']=evaluate(model,validation,context=args.context)
        history=[];state=None;elapsed=0.;router_before={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('graph_router.')}
        for cursor in range(0,data.shape[1]-1,args.context):
            x=data[:,cursor:cursor+args.context];y=data[:,cursor+1:cursor+args.context+1]
            x=x[:,:y.shape[1]]
            for group in optimizer.param_groups:group['lr']=args.lr*min(1.,(cursor*args.batch+x.numel())/256)
            result,row=timed_step(model,optimizer,x,y,state)
            state=result['state'].detach(compact=True)
            row['trained_targets']=(cursor+x.shape[1])*args.batch
            row['routing']=routing_summary(result['routing'],args.batch)
            elapsed+=sum(row[k] for k in ('forward_seconds','backward_seconds','optimizer_seconds'))
            history.append(row)
            with (args.out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps({'trained_targets':row['trained_targets'],'nll':row['nll'],
                              'training_bytes_per_second':row['training_bytes_per_second']}),flush=True)
            del result
        summary.update(history=history,trained_targets=args.targets,training_seconds=elapsed,
                       training_bytes_per_second=args.targets/elapsed,
                       final_validation=evaluate(model,validation,context=args.context),
                       router_parameter_change_l1=sum(float((p.detach()-router_before[n]).abs().sum()) for n,p in model.named_parameters() if n in router_before),
                       regional_row_clocks={n:s.clock.tolist() if isinstance(s.clock,torch.Tensor) else s.clock for n,s in state.nodes.items()},
                       memory_writes=[s.writes for s in state.memory.streams] if cfg.region_topk else state.memory.writes)
        save(args.out/'last.pt',model,state,optimizer,step=len(history),metadata={'protocol':summary,'execution_backend':args.backend})
    summary['complete']=True
    summary['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if summary['source_sha256']!=source_digest():raise RuntimeError('Source changed during measurement.')
    atomic_json(args.out/'results.json',summary)
    print(json.dumps({k:summary[k] for k in ('complete','parameters','peak_rss_mib','median_bytes_per_second','training_bytes_per_second') if k in summary}),flush=True)


if __name__=='__main__':main()
