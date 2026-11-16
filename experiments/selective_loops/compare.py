"""Matched short Tiny Shakespeare screen and serial CPU cost measurements.

python -m experiments.selective_loops.compare --data /path/input.txt --out /path/results.json
Compilation, evaluation, reset and checkpoint I/O are excluded from training
curve timers; dedicated throughput probes include reset and final retirement.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

from shoe_multiscale import Model


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--targets',type=int,default=1_000_000)
    p.add_argument('--seeds',type=int,nargs='+',default=[1,7,19]);p.add_argument('--checkpoint-dir',type=Path)
    args=p.parse_args()
    if hasattr(os,'sched_getaffinity'):
        cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    else:cpu=None
    data=args.data.read_bytes();split=int(len(data)*.9);train=data[:split];valid=data[split:]
    if args.targets<1 or args.targets>=len(train):raise ValueError('target budget must fit the unique training prefix')
    result=dict(base_commit='e2a2e93',dataset='Tiny Shakespeare',dataset_url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
                sha256=hashlib.sha256(data).hexdigest(),split_byte=split,validation_targets=len(valid)-1,
                targets=args.targets,seeds=args.seeds,nominal_parameters=1_000_000,cpu=cpu,platform=platform.platform(),
                loop_config=dict(steps=[0,1,2,2],strength=.25,tolerance=.01),
                schedule='cosine 0.008 to 0.0004 fast/query; slow=10x, head=20x; 50K-target rate blocks',runs=[],benchmarks=[])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    def save():args.out.write_text(json.dumps(result,indent=2)+'\n')
    for seed in args.seeds:
        models={name:Model(seed=seed,loops=None if name=='C' else result['loop_config']) for name in ('C','loops')}
        rows={name:dict(seed=seed,arm=name,parameters=m.parameter_count,curve=[dict(targets=0,nll=m.evaluate(valid),cpu_seconds=0,seconds=0)],
                       cpu_seconds=0.,seconds=0.) for name,m in models.items()}
        for start in range(0,args.targets,50_000):
            end=min(start+50_000,args.targets)
            rate=.008*(.05+.95*.5*(1+math.cos(math.pi*start/args.targets)))
            order=('C','loops') if (start//50_000)%2==0 else ('loops','C')
            for name in order:
                m=models[name];m.configure(fast_rate=rate,slow_rate=10*rate,query_rate=rate,head_rate=20*rate)
                t=time.perf_counter();c=time.process_time();loss=m.run(train[start:end],train[start+1:end+1])
                if end==args.targets:m.finish()
                rows[name]['cpu_seconds']+=time.process_time()-c;rows[name]['seconds']+=time.perf_counter()-t
                if end%250_000==0 or end==args.targets:
                    rows[name]['curve'].append(dict(targets=end,nll=m.evaluate(valid),train_nll_last_block=loss,
                                                   cpu_seconds=rows[name]['cpu_seconds'],seconds=rows[name]['seconds']))
                    print(seed,name,end,rows[name]['curve'][-1]['nll'],flush=True)
        for name,m in models.items():
            rows[name]['loop_stats']=m.loop_stats();rows[name]['final_nll']=rows[name]['curve'][-1]['nll']
            result['runs'].append(rows[name])
            if args.checkpoint_dir:
                args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
                m.save(args.checkpoint_dir/f'{name}_{seed}.shoe')
        if seed==args.seeds[0]:
            probe=train[:100_001]
            for rep in range(3):
                for name in (('C','loops') if rep%2==0 else ('loops','C')):
                    with models[name].clone() as m:
                        t=time.perf_counter();c=time.process_time();m.reset_context()
                        m.train(probe);m.finish()
                        cpu_elapsed=time.process_time()-c;elapsed=time.perf_counter()-t
                        result['benchmarks'].append(dict(arm=name,mode='train',rep=rep,targets=len(probe)-1,
                                                        cpu_seconds=cpu_elapsed,seconds=elapsed,
                                                        cpu_bytes_per_second=(len(probe)-1)/cpu_elapsed,bytes_per_second=(len(probe)-1)/elapsed))
                    with models[name].inference_copy() as m:
                        t=time.perf_counter();c=time.process_time()
                        m.run(probe[:-1],probe[1:]);cpu_elapsed=time.process_time()-c;elapsed=time.perf_counter()-t
                        result['benchmarks'].append(dict(arm=name,mode='inference',rep=rep,targets=len(probe)-1,
                                                        cpu_seconds=cpu_elapsed,seconds=elapsed,
                                                        cpu_bytes_per_second=(len(probe)-1)/cpu_elapsed,bytes_per_second=(len(probe)-1)/elapsed))
        for m in models.values():m.close()
        save()
    result['summary']={name:dict(mean_nll=statistics.mean(r['final_nll'] for r in result['runs'] if r['arm']==name),
        train_cpu_bps=statistics.median(r['cpu_bytes_per_second'] for r in result['benchmarks'] if r['arm']==name and r['mode']=='train'),
        inference_cpu_bps=statistics.median(r['cpu_bytes_per_second'] for r in result['benchmarks'] if r['arm']==name and r['mode']=='inference')) for name in ('C','loops')}
    save();print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
