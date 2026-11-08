"""Reproducible 1,000-byte-token profile of the widened cortical model.

Modes run in separate processes so profiler overhead and allocator high-water
marks do not contaminate the training/benchmark memory figures.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import time
import torch
from torch.nn import functional as F
from . import CorticalModel, cortical_wide
from .checkpoint import save, load
from .diagnostics import PortMonitor, SavedTensorAccounting
from .fused_kron import native_status
from .train_fineweb import atomic_json, optimizer_for, source_digest


def rss_mib():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])/1024
    return None


def corpus(args):
    raw = args.data.read_bytes()
    if args.targets % args.batch or len(raw)<args.targets+args.batch:
        raise ValueError('Need a divisible target budget plus one prefix byte per stream.')
    raw = raw[:args.targets+args.batch]
    return torch.tensor(list(raw),dtype=torch.long).reshape(args.batch,-1),raw


def new_model(args):
    cfg = cortical_wide()
    torch.manual_seed(cfg.seed)
    return CorticalModel(cfg).set_execution_backend(args.backend)


@torch.no_grad()
def probe(model,data):
    monitor = PortMonitor(model)
    start = time.perf_counter()
    result = model(data[:,:-1],targets=data[:,1:],observer=monitor)
    return {'nll':float(result['ce']),'accuracy':float((result['logits'].argmax(-1)==data[:,1:]).float().mean()),
            'seconds':time.perf_counter()-start,**monitor.snapshot()}


def benchmark(args,data):
    model = new_model(args)
    # Warm runtime state without changing parameters. Both backends see the
    # same eight observations, followed by the same full 32-observation graph.
    with torch.no_grad():
        state = model(data[:,:8])['state'].detach(compact=True)
    x,y = data[:,8:8+args.context],data[:,9:9+args.context]
    before = rss_mib()
    accounting = SavedTensorAccounting()
    start = time.perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(accounting.pack,accounting.unpack):
        result = model(x,targets=y,state=state)
    forward = time.perf_counter()-start
    forward_rss = rss_mib()
    start = time.perf_counter()
    result['loss'].backward()
    backward = time.perf_counter()-start
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True))
    torch.save(result['logits'].detach(),args.out/f'benchmark_{args.backend}_logits.pt')
    summary = {'backend':args.backend,'batch':args.batch,'context':args.context,
               'target_positions':x.numel(),'parameters':sum(p.numel() for p in model.parameters()),
               'nll':float(result['ce'].detach()),'gradient_norm':norm,
               'forward_seconds':forward,'backward_seconds':backward,
               'bytes_per_second':x.numel()/(forward+backward),'rss_before_mib':before,
               'rss_after_forward_mib':forward_rss,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
               **accounting.summary()}
    atomic_json(args.out/f'benchmark_{args.backend}.json',summary)
    print(json.dumps(summary),flush=True)


def train(args,data,raw):
    if args.backend not in ('optimized','fused'):
        raise ValueError('Train with optimized or fused; benchmark reference separately.')
    if any((args.out/name).exists() for name in ('last.pt','history.jsonl','ports.jsonl')):
        raise ValueError('Refuse to overwrite an existing or partial profile run.')
    model = new_model(args)
    optimizer = optimizer_for(model,args.lr,.01)
    state = model.initial_state(args.batch)
    (args.out/'profile_train.bin').write_bytes(raw)
    summary = {'format':'shoebox-wide-profile-v1','complete':False,'parameters':sum(p.numel() for p in model.parameters()),
               'config':model.cfg.to_dict(),'targets':args.targets,'batch':args.batch,'context':args.context,
               'lr':args.lr,'warmup_targets':args.warmup_targets,'lr_after_warmup':'constant',
               'source_sha256':source_digest(),'data_sha256':hashlib.sha256(raw).hexdigest(),
               'threads':args.threads,'torch':str(torch.__version__),'python':platform.python_version(),
               'backend':args.backend,'native':native_status() if args.backend=='fused' else None}
    summary['allocation'] = {name:{'parameters':sum(p.numel() for p in region.parameters()),
                                  'width':region.hat.cfg.width,'experts':region.hat.cfg.experts,
                                  'expert_hidden':region.hat.cfg.expert_hidden,'hat_symbols':region.hat.cfg.hat_symbols,
                                  'hat_active_symbols':region.hat.cfg.hat_active_symbols,
                                  'output_port_bytes':list(map(int,region.outputs))}
                             for name,region in model.regions.items()}
    summary['hippocampus_parameters'] = sum(p.numel() for p in model.hippocampus.parameters())
    atomic_json(args.out/'topology.json',[asdict(spec) for spec in model.specs])
    summary['initial_probe'] = probe(model,data[:,:args.context+1])
    if args.old_checkpoint:
        old,_,_,_ = load(args.old_checkpoint)
        summary['old_100k_probe'] = probe(old,data[:,:args.context+1])
        del old
    print(json.dumps({'event':'start','parameters':summary['parameters'],'initial_nll':summary['initial_probe']['nll']}),flush=True)
    history,metrics,lag_gradients = [],[],[]
    cursor,step,elapsed,diagnostic_seconds = 0,0,0.,0.
    while cursor<data.shape[1]-1:
        length = min(args.context,data.shape[1]-1-cursor)
        x,y = data[:,cursor:cursor+length],data[:,cursor+1:cursor+length+1]
        tap_events = (0,15,23,29,31) if step==0 else ()
        monitor = PortMonitor(model,tap_events=tap_events)
        accounting = SavedTensorAccounting()
        for group in optimizer.param_groups:
            group['lr'] = args.lr*min(1.,(cursor*args.batch+x.numel())/max(1,args.warmup_targets))
        optimizer.zero_grad(set_to_none=True)
        tick = time.perf_counter()
        with torch.autograd.graph.saved_tensors_hooks(accounting.pack,accounting.unpack):
            result = model(x,targets=y,state=state,observer=monitor)
        forward = time.perf_counter()-tick
        forward_rss = rss_mib()
        if step==0:
            # Isolate temporal credit from the final prediction, with no aux
            # loss. This pass is measured separately from normal backward.
            start = time.perf_counter()
            last_loss = F.cross_entropy(result['logits'][:,-1],y[:,-1])
            entries = list(monitor.taps.items())
            gradients = torch.autograd.grad(last_loss,[value for _,value in entries],
                                             retain_graph=True,allow_unused=True)
            for ((name,event),activation),gradient in zip(entries,gradients):
                lag_gradients.append({'region':name,'event':event,'lag':length-1-event,
                                      'gradient_l1':0. if gradient is None else float(gradient.abs().sum()),
                                      'gradient_rms':0. if gradient is None else float(gradient.double().square().mean().sqrt()),
                                      'finite':gradient is None or bool(torch.isfinite(gradient).all())})
            diagnostic_seconds += time.perf_counter()-start
            if not all(g['finite'] for g in lag_gradients):
                raise RuntimeError('Nonfinite temporal-credit gradient.')
            monitor.taps.clear()
            del gradients,entries,last_loss,activation,gradient
        start = time.perf_counter()
        result['loss'].backward()
        backward = time.perf_counter()-start
        start = time.perf_counter()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=True)
        optimizer.step()
        update = time.perf_counter()-start
        before_compaction = result['state'].storage_bytes()
        state = result['state'].detach(compact=True)
        step += 1
        cursor += length
        duration = forward+backward+update
        elapsed += duration
        record = {'step':step,'trained_targets':cursor*args.batch,'observation_steps':length,
                  'nll':float(result['ce'].detach()),'accuracy':float((result['logits'].detach().argmax(-1)==y).float().mean()),
                  'forward_seconds':forward,'backward_seconds':backward,'optimizer_seconds':update,
                  'monitor_seconds':monitor.seconds,'gradient_norm':float(norm),'lr':optimizer.param_groups[0]['lr'],
                  'rss_after_forward_mib':forward_rss,'state_bytes_before_compaction':before_compaction,
                  'state_bytes_after_compaction':state.storage_bytes(),**accounting.summary()}
        history.append(record)
        metrics.append({'trained_targets':cursor*args.batch,**monitor.snapshot()})
        del result,monitor,accounting
        record['rss_after_backward_mib'] = rss_mib()
        with (args.out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        with (args.out/'ports.jsonl').open('a') as f:f.write(json.dumps(metrics[-1])+'\n')
        print(json.dumps({'event':'update',**{k:record[k] for k in ('step','trained_targets','nll','forward_seconds','backward_seconds','rss_after_forward_mib')}}),flush=True)
    assert cursor*args.batch==args.targets
    save(args.out/'last.pt',model,state,optimizer,step=step,
         metadata={'format':'cortical-wide-profile','complete':True,'trained_targets':args.targets,'cursor':cursor,
                   'protocol':{k:summary[k] for k in ('config','targets','batch','context','lr','warmup_targets',
                                                     'source_sha256','data_sha256','threads','torch','python')},
                   'execution_backend':args.backend})
    summary.update(complete=True,trained_targets=args.targets,updates=step,training_seconds=elapsed,
                   training_bytes_per_second=args.targets/elapsed,temporal_diagnostic_seconds=diagnostic_seconds,
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                   history=history,port_history=metrics,temporal_gradients=lag_gradients,
                   final_state_bytes=state.storage_bytes())
    summary['final_probe'] = probe(model,data[:,:args.context+1])
    atomic_json(args.out/'results.json',summary)
    print(json.dumps({'event':'complete','targets':args.targets,'seconds':elapsed,
                      'bytes_per_second':args.targets/elapsed,'final_probe_nll':summary['final_probe']['nll']}),flush=True)


def operators(args,data):
    model,state,_,_ = load(args.out/'last.pt')
    model.set_execution_backend(args.backend)
    handles,stacks = [],{}
    for name,module in model.regions.items():
        def before(module,inputs,name=name):
            scope = torch.profiler.record_function('region:'+name);scope.__enter__()
            stacks[name] = scope
        def after(module,inputs,output,name=name):
            stacks.pop(name).__exit__(None,None,None)
        handles += [module.register_forward_pre_hook(before),module.register_forward_hook(after)]
    optimizer = optimizer_for(model,args.lr,.01)
    optimizer.zero_grad(set_to_none=True)
    x,y = data[:,:1],data[:,1:2]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                record_shapes=True,profile_memory=True,with_stack=False) as profile:
        result = model(x,targets=y,state=state)
        result['loss'].backward()
    rows = []
    for entry in profile.key_averages(group_by_input_shape=True):
        rows.append({'operator':entry.key,'calls':entry.count,'self_cpu_seconds':entry.self_cpu_time_total/1e6,
                     'inclusive_cpu_seconds':entry.cpu_time_total/1e6,
                     'self_cpu_memory_bytes':entry.self_cpu_memory_usage,
                     'input_shapes':str(entry.input_shapes)})
    rows.sort(key=lambda r:r['self_cpu_seconds'],reverse=True)
    summary = {'backend':args.backend,'scope':'one observation per stream at the final checkpoint; backward only; no optimizer update',
               'scored_positions':x.numel(),'nll':float(result['ce'].detach()),'operators':rows,
               'total_self_cpu_seconds':sum(r['self_cpu_seconds'] for r in rows),
               'positive_self_allocated_bytes':sum(max(0,r['self_cpu_memory_bytes']) for r in rows),
               'profiler_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    atomic_json(args.out/f'operators_{args.backend}.json',summary)
    for handle in handles:handle.remove()
    print(json.dumps({'backend':args.backend,'top_operators':rows[:8],
                      'positive_self_allocated_bytes':summary['positive_self_allocated_bytes']}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('benchmark','train','operators'),required=True)
    parser.add_argument('--backend',choices=('reference','optimized','fused'),default='optimized')
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--old-checkpoint',type=Path)
    parser.add_argument('--targets',type=int,default=1000)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--context',type=int,default=32)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--warmup-targets',type=int,default=256)
    args = parser.parse_args()
    if min(args.targets,args.batch,args.context,args.threads)<1:
        parser.error('Budgets, batch, horizon and thread count must be positive.')
    torch.set_num_threads(args.threads)
    if args.backend=='fused' and not native_status()['available']:
        parser.error('Fused profiling requires an available native kernel: '+str(native_status()))
    args.out.mkdir(parents=True,exist_ok=True)
    data,raw = corpus(args)
    if args.mode=='benchmark':benchmark(args,data)
    elif args.mode=='train':train(args,data,raw)
    else:operators(args,data)


if __name__=='__main__':
    main()
