"""Compare exact v39 continuation and observational cost against a base checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')


def worker(args):
    sys.path.insert(0,str(args.runtime))
    import numpy as np
    from runtime import Model,fresh_model
    from validate_content_experts import TRAIN
    os.sched_setaffinity(0,{args.cpu})
    data=np.frombuffer(TRAIN*((max(1024,args.targets)+1)//len(TRAIN)+1),np.uint8)
    if args.initialize:
        m=fresh_model(seed=7)
        m.run(data[:1024],data[1:1025],mode='periodic',lr=.04,lp=.04/3)
        m.save(args.checkpoint);m.close();return
    m=Model.load(args.checkpoint)
    start=time.perf_counter();cpu=time.process_time()
    loss,logits=m.run(data[:args.targets],data[1:args.targets+1],mode='periodic',lr=.04,lp=.04/3,logits=True)
    seconds=time.perf_counter()-start;cpu_seconds=time.process_time()-cpu
    result=dict(nll=float(loss),seconds=seconds,cpu_seconds=cpu_seconds,
                bytes_per_second=args.targets/seconds,cpu_bytes_per_second=args.targets/cpu_seconds,
                fingerprint=m.fingerprint(),logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest())
    m.close();args.output.write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-runtime',type=Path)
    p.add_argument('--runtime',type=Path,default=Path(__file__).resolve().parents[1]/'runtime')
    p.add_argument('--targets',type=int,default=512)
    p.add_argument('--repeats',type=int,default=4)
    p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results/acceptance_runtime_comparison.json')
    p.add_argument('--worker',action='store_true');p.add_argument('--initialize',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    args=p.parse_args()
    if args.targets<1 or args.repeats<1:p.error('Positive targets/repeats required')
    if args.worker:return worker(args)
    if args.baseline_runtime is None:p.error('--baseline-runtime must name the base checkout runtime directory')
    result=dict(targets=args.targets,repeats=args.repeats,cpu=args.cpu,
                protocol='Same actual baseline v39 checkpoint before every timed segment; alternating process order. Compilation, load and initialization excluded.',
                runs={'baseline':[],'acceptance_observations':[]})
    result['sources']={}
    for name,path in (('baseline',args.baseline_runtime),('acceptance_observations',args.runtime)):
        digest=hashlib.sha256()
        for file in sorted(path.rglob('*')):
            if file.suffix in ('.py','.cpp','.inc'):
                digest.update(str(file.relative_to(path)).encode());digest.update(file.read_bytes())
        head=subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],capture_output=True,text=True)
        result['sources'][name]=dict(runtime_sha256=digest.hexdigest(),
                                    checkout_head=head.stdout.strip() if head.returncode==0 else None)
    with tempfile.TemporaryDirectory(prefix='shoebox-acceptance-') as tmp:
        checkpoint=Path(tmp)/'baseline.npz';out=Path(tmp)/'result.json'
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--checkpoint',str(checkpoint),
                 '--targets',str(args.targets),'--cpu',str(args.cpu),'--output',str(out)]
        subprocess.run(command+['--runtime',str(args.baseline_runtime),'--initialize'],check=True)
        for i in range(args.repeats):
            jobs=[('baseline',args.baseline_runtime),('acceptance_observations',args.runtime)]
            for name,runtime in jobs[::1 if i%2==0 else -1]:
                subprocess.run(command+['--runtime',str(runtime)],check=True)
                row=json.loads(out.read_text());result['runs'][name].append(row)
                print(name,i+1,row['bytes_per_second'],flush=True)
        rows=[r for values in result['runs'].values() for r in values]
        for key in ('fingerprint','logits_sha256','nll'):
            if len({r[key] for r in rows})!=1:raise AssertionError('Baseline continuation mismatch: '+key)
        import statistics
        result['bitwise_exact_baseline_continuation']=True
        result['medians']={name:{key:statistics.median(r[key] for r in rows)
                                 for key in ('bytes_per_second','cpu_bytes_per_second')}
                           for name,rows in result['runs'].items()}
        result['throughput_change_percent']=100*(result['medians']['acceptance_observations']['bytes_per_second']/result['medians']['baseline']['bytes_per_second']-1)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
