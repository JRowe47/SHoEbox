"""Matched reference/optimized continuation; also tests CPU row-team settings."""
import argparse
import hashlib
import json
import os
import resource
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

TRAIN=(b'The river runs through the valley. The king crossed the bridge. '
       b'A bird flew over the water. The queen opened the gate. ')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def worker(a):
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    sys.path.insert(0,str(a.runtime.resolve()))
    import numpy as np
    from runtime import Model,fresh_model,core
    from reference_acceptance import collect,check_integrity
    n=a.warmup+a.targets+1
    raw=a.corpus.read_bytes()[:n] if a.corpus else (TRAIN*(n//len(TRAIN)+1))[:n]
    if len(raw)!=n:raise ValueError('Corpus is shorter than the measured workload')
    data=np.frombuffer(raw,np.uint8)
    if a.initialize:
        os.sched_setaffinity(0,{a.cpu});m=fresh_model(seed=7)
        m.run(data[:a.warmup],data[1:a.warmup+1],mode='periodic',lr=.04,lp=.04/3)
        m.save(a.checkpoint);m.close();return
    m=Model.load(a.checkpoint)
    if a.workers>1:
        available=sorted(os.sched_getaffinity(0));cpus=[a.cpu]+[c for c in available if c!=a.cpu][:a.workers-1]
        if len(cpus)!=a.workers:raise ValueError('Insufficient CPUs for requested workers')
        m.configure_content_workers(a.workers,cpus)
    os.sched_setaffinity(0,{a.cpu})
    before=time.perf_counter();cpu_before=time.process_time()
    nll,logits=m.run(data[a.warmup:-1],data[a.warmup+1:],mode=a.mode,lr=.04,lp=.04/3,logits=True)
    seconds=time.perf_counter()-before;cpu_seconds=time.process_time()-cpu_before
    row=dict(experts=core.E,parameters=m.parameter_count,workers=a.workers,targets=a.targets,
             seconds=seconds,cpu_seconds=cpu_seconds,bytes_per_second=a.targets/seconds,
             nll=float(nll),logits_sha256=digest(logits.tobytes()),fingerprint=m.fingerprint(),
             base_fingerprint=m.base_fingerprint(),routes_sha256=digest(m.codes().tobytes()),
             content_state_sha256=digest(m.content_expert_state().tobytes()),
             paired_state_sha256=digest(m.paired_memory_state().tobytes()),
             pattern_state_sha256=digest(m.pattern_state().tobytes()),
             main_sha256=digest(m.flat.tobytes()),extra_sha256=digest(m.extra_flat.tobytes()),
             composition_sha256=digest(m.composition_state(fitted=True).tobytes()),
             patterns_sha256=digest(m.pattern_state(fitted=True).tobytes()),
             data_sha256=digest(raw),capabilities=m.capabilities())
    check_integrity(collect(m));row['credit_integrity']=True
    row['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if hasattr(m,'completion_storage'):row['storage']=m.completion_storage()
    if hasattr(m,'publication_storage'):row['publication_storage']=m.publication_storage()
    if a.save_after:m.save(a.save_after)
    m.close();a.output.write_text(json.dumps(row,indent=2)+'\n')


def source_hash(path):
    h=hashlib.sha256()
    for file in sorted(path.rglob('*')):
        if file.suffix in ('.py','.cpp','.inc'):
            h.update(str(file.relative_to(path)).encode());h.update(file.read_bytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-runtime',type=Path)
    p.add_argument('--runtime',type=Path,default=Path(__file__).resolve().parents[1]/'runtime')
    p.add_argument('--targets',type=int,default=1024);p.add_argument('--warmup',type=int,default=1024)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
    p.add_argument('--mode',choices=('periodic','eval'),default='periodic')
    p.add_argument('--corpus',type=Path);p.add_argument('--worker-counts',type=int,nargs='+',default=[1])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--initialize',action='store_true')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--save-after',type=Path,help='Worker-only checkpoint for cross-runtime continuation checks')
    a=p.parse_args()
    if a.worker:return worker(a)
    if a.baseline_runtime is None:p.error('--baseline-runtime is required')
    if a.targets<1 or a.warmup<1 or a.repeats<1:p.error('Positive target, warmup and repeat counts required')
    variants=[('reference',a.baseline_runtime,1)]+[(f'optimized_{n}',a.runtime,n) for n in a.worker_counts]
    result=dict(protocol='Identical actual baseline checkpoint before every continuation; alternating process order; compilation, loading and hashes excluded.',
                mode=a.mode,workload='natural' if a.corpus else 'repeated',targets=a.targets,warmup=a.warmup,repeats=a.repeats,cpu=a.cpu,
                source_sha256={name:source_hash(path) for name,path,_ in variants},runs={name:[] for name,_,_ in variants})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='shoebox-opt-') as temporary:
        checkpoint=Path(temporary)/'baseline.npz';out=Path(temporary)/'result.json'
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--checkpoint',str(checkpoint),
                 '--targets',str(a.targets),'--warmup',str(a.warmup),'--cpu',str(a.cpu),'--mode',a.mode,'--output',str(out)]
        if a.corpus:command+=['--corpus',str(a.corpus.resolve())]
        subprocess.run(command+['--runtime',str(a.baseline_runtime.resolve()),'--initialize'],check=True)
        for i in range(a.repeats):
            for name,runtime,workers in variants[::1 if i%2==0 else -1]:
                subprocess.run(command+['--runtime',str(runtime.resolve()),'--workers',str(workers)],check=True)
                row=json.loads(out.read_text());result['runs'][name].append(row)
                print(name,i+1,round(row['bytes_per_second'],2),flush=True)
                a.output.write_text(json.dumps(result,indent=2)+'\n')
    rows=[row for group in result['runs'].values() for row in group]
    for key in ('nll','logits_sha256','main_sha256','extra_sha256','composition_sha256','patterns_sha256','data_sha256',
                'base_fingerprint','routes_sha256','content_state_sha256','paired_state_sha256','pattern_state_sha256'):
        if len({row[key] for row in rows})!=1:raise AssertionError('Continuation mismatch: '+key)
    result['bitwise_logits_and_fitted_state']=True
    result['physical_state_identical']=len({row['fingerprint'] for row in rows})==1
    result['medians']={name:statistics.median(r['bytes_per_second'] for r in rows) for name,rows in result['runs'].items()}
    result['speedups']={name:rate/result['medians']['reference'] for name,rate in result['medians'].items()}
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
