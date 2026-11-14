"""Isolated Action-4 ablations/tuning; fixed data, common seeds and NLL target."""
import argparse,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime'),str(ROOT/'src')]
from measure_current_baseline import fixtures,atomic_json,source_manifest,rss_kib
POLICIES={'baseline':{},'predictor':dict(compact=True,rank=16),
          'y':dict(slow_stride=4,rank=16),'combined':dict(compact=True,slow_stride=4,rank=16)}

def worker(a):
    import numpy as np
    from runtime import fresh_model
    from memory_inspection import read_memory_index
    policy=dict(POLICIES[a.variant])
    if policy:policy.update(rank=a.rank,slow_scale=a.slow_scale)
    if a.variant in ('y','combined'):policy['slow_stride']=a.stride
    m=fresh_model(seed=a.seed,profile=a.profile,prediction=policy)
    os.sched_setaffinity(0,{a.cpu})
    docs,val,data=fixtures(a.workload,a.targets,validation_bytes=a.validation)
    x=np.frombuffer(docs[0],np.uint8);v=np.frombuffer(val,np.uint8)
    points=[];seconds=cpu=weighted=0.
    def heldout():
        e=m.evaluation_copy(memory='cold_frozen')
        try:return float(e.run(v[:-1],v[1:]))
        finally:e.close()
    # Compilation and construction finish before the measured training clock.
    for lo in range(0,a.targets,a.chunk):
        hi=min(a.targets,lo+a.chunk);start=time.perf_counter();c0=time.process_time()
        loss=m.run(x[lo:hi],x[lo+1:hi+1],mode='periodic',lr=a.lr,lp=a.lr/3,logits=a.logits)
        cpu+=time.process_time()-c0;seconds+=time.perf_counter()-start
        weighted+=(loss[0] if a.logits else loss)*(hi-lo)
        points.append(dict(bytes=hi,seconds=seconds,cpu_seconds=cpu,heldout_nll=heldout()))
    live=read_memory_index(m)
    for name,row in live.items():
        if row['invalid_retained_spans'] or row['missing_content_records']:raise AssertionError((name,row))
    info=m.inspect_prediction();info.pop('features');info.pop('regions')
    crossing=next((p for p in points if p['heldout_nll']<=a.nll_target),None)
    result=dict(variant=a.variant,seed=a.seed,policy=m.prediction,profile=a.profile,lr=a.lr,local_lr=a.lr/3,
                targets=a.targets,workload=a.workload,validation_bytes=a.validation,data=data,logits=a.logits,chunk=a.chunk,
                cpu=a.cpu,seconds=seconds,process_cpu_seconds=cpu,bytes_per_second=a.targets/seconds,
                heldout_nll=points[-1]['heldout_nll'],train_nll=weighted/a.targets,quality_curve=points,
                nll_target=a.nll_target,time_to_target=None if crossing is None else crossing['seconds'],
                bytes_to_target=None if crossing is None else crossing['bytes'],
                fitted_parameter_count=m.parameter_count,parameter_accounting=m.prediction_parameter_accounting(),
                rss_kib=rss_kib(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                prediction_state=info,integrity=True,source={k:v for k,v in source_manifest().items() if k!='files'})
    m.close();atomic_json(a.output,result)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=POLICIES,default='baseline');p.add_argument('--variants',default='baseline,predictor,y,combined')
    p.add_argument('--seed',type=int,default=7);p.add_argument('--seeds',default='7,11,19')
    p.add_argument('--rates',default='.04');p.add_argument('--lr',type=float,default=.04)
    p.add_argument('--rank',type=int,default=16);p.add_argument('--stride',type=int,default=4);p.add_argument('--slow-scale',type=float,default=1.)
    p.add_argument('--profile',choices=('event','pipeline','reference'),default='event')
    p.add_argument('--targets',type=int,default=17408);p.add_argument('--chunk',type=int,default=2048)
    p.add_argument('--validation',type=int,default=1024);p.add_argument('--workload',choices=('natural','repeated'),default='natural')
    p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)));p.add_argument('--nll-target',type=float,default=3.33)
    p.add_argument('--logits',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args()
    if a.worker:worker(a);return
    rows=[];variants=a.variants.split(',')
    for trial,seed in enumerate(map(int,a.seeds.split(','))):
        for rate in map(float,a.rates.split(',')):
            order=variants[trial%len(variants):]+variants[:trial%len(variants)]
            for variant in order:
                out=a.output.with_suffix('.worker.json')
                cmd=[sys.executable,__file__,'--worker','--output',str(out),'--variant',variant,'--seed',str(seed),'--lr',str(rate),
                     '--rank',str(a.rank),'--stride',str(a.stride),'--slow-scale',str(a.slow_scale),'--profile',a.profile,
                     '--targets',str(a.targets),'--chunk',str(a.chunk),'--validation',str(a.validation),'--workload',a.workload,
                     '--cpu',str(a.cpu),'--nll-target',str(a.nll_target),'--logits' if a.logits else '--no-logits']
                subprocess.run(cmd,check=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','SHOEBOX_INLINE_DIAGNOSTICS':'0'})
                row=json.loads(out.read_text());rows.append(row);out.unlink()
                atomic_json(a.output,dict(rows=rows,completed=False))
                print(variant,seed,rate,round(row['bytes_per_second'],1),round(row['heldout_nll'],5),row['time_to_target'],flush=True)
    atomic_json(a.output,dict(rows=rows,completed=True))
if __name__=='__main__':main()
