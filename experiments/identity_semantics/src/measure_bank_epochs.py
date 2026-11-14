"""Action 6 isolated bank epoch quality, worker and storage comparisons.

No counters or timers are installed in training. Each row is a fresh process;
only run/document-drain calls are timed. Quality and memory reads are outside.
"""
import argparse,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime'),str(ROOT/'src')]
from measure_current_baseline import fixtures,segments,atomic_json,source_manifest,rss_kib


def make_model(a):
    from runtime import fresh_model
    if a.variant=='event':profile='event';epochs=None
    elif a.variant=='legacy':profile='pipeline';epochs=None
    else:
        profile='epochs';parts=a.variant.split('-');base=parts[0];epoch,lag=map(int,base[1:].split('l'))
        epochs=dict(epoch_bytes=epoch,lag_epochs=lag,storage='dense' if 'dense' in parts else 'sparse')
    m=fresh_model(seed=a.seed,profile=profile,bank_epochs=epochs,
                  event=dict(learning='ancestry') if 'ancestry' in a.variant else None,
                  prediction=dict(compact=True,rank=16,slow_stride=8))
    if profile!='event':m.configure_pipeline_workers(2 if 'w2' in a.variant else 1,cpu=a.worker_cpu if 'w2' in a.variant else None)
    os.sched_setaffinity(0,{a.cpu})
    return m


def worker(a):
    import numpy as np
    from memory_inspection import read_memory_index
    m=make_model(a);docs,val,data=fixtures(a.workload,a.targets,validation_bytes=a.validation)
    v=np.frombuffer(val,np.uint8);seconds=cpu=weighted=0.;points=[]
    for lo,xb,yb,end in segments(docs,a.chunk):
        x=np.frombuffer(xb,np.uint8);y=np.frombuffer(yb,np.uint8)
        t0=time.perf_counter();c0=time.process_time()
        loss=m.run(x,y,mode='periodic',lr=a.lr,lp=a.lr/3,logits=a.logits)
        if a.workload=='documents' and end:m.finish_document(mode='periodic',lr=a.lr,lp=a.lr/3,reset=True)
        cpu+=time.process_time()-c0;seconds+=time.perf_counter()-t0
        weighted+=(loss[0] if a.logits else loss)*len(x)
        e=m.evaluation_copy(memory='cold_frozen')
        try:nll=float(e.run(v[:-1],v[1:],mode='eval'))
        finally:e.close()
        points.append(dict(bytes=lo+len(x),seconds=seconds,process_cpu_seconds=cpu,heldout_nll=nll))
    live=read_memory_index(m)
    assert all(not r['invalid_retained_spans'] and not r['missing_content_records'] for r in live.values())
    p=m.inspect_pipeline()
    for key in ('banks','building_buffer','pending_buffer'):p.pop(key,None)
    crossing=next((p for p in points if p['heldout_nll']<=a.nll_target),None)
    source=source_manifest()
    result=dict(variant=a.variant,seed=a.seed,lr=a.lr,local_lr=a.lr/3,bank_epochs=m.bank_epochs,event=m.event,prediction=m.prediction,
                targets=a.targets,chunk=a.chunk,validation=a.validation,workload=a.workload,data=data,logits=a.logits,
                cpu=a.cpu,worker_cpu=a.worker_cpu,seconds=seconds,process_cpu_seconds=cpu,bytes_per_second=a.targets/seconds,
                train_nll=weighted/a.targets,heldout_nll=points[-1]['heldout_nll'],quality_curve=points,nll_target=a.nll_target,
                time_to_target=None if crossing is None else crossing['seconds'],bytes_to_target=None if crossing is None else crossing['bytes'],
                rss_kib=rss_kib(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                pipeline=p,epochs=m.inspect_bank_epochs(),fitted_parameters=m.parameter_count,integrity=True,
                source={k:v for k,v in source.items() if k!='files'})
    m.close();atomic_json(a.output,result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variants',default='event,legacy,e1l0,e1l1,e4l0,e4l1,e8l0,e8l1,e16l0,e16l1')
    p.add_argument('--variant',default='event');p.add_argument('--seeds',default='7');p.add_argument('--seed',type=int,default=7)
    p.add_argument('--rates',default='.08,.16,.32');p.add_argument('--lr',type=float,default=.16)
    p.add_argument('--targets',type=int,default=4096);p.add_argument('--chunk',type=int,default=2048)
    p.add_argument('--validation',type=int,default=512);p.add_argument('--workload',choices=('repeated','natural','documents'),default='natural')
    allowed=sorted(os.sched_getaffinity(0));p.add_argument('--cpu',type=int,default=allowed[0]);p.add_argument('--worker-cpu',type=int,default=allowed[1])
    p.add_argument('--nll-target',type=float,default=3.25);p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--logits',action=argparse.BooleanOptionalAction,default=False)
    a=p.parse_args()
    if a.worker:return worker(a)
    rows=[];variants=a.variants.split(',');trial=0
    for seed in map(int,a.seeds.split(',')):
        for rate in map(float,a.rates.split(',')):
            for repeat in range(a.repeats):
                order=variants[trial%len(variants):]+variants[:trial%len(variants)];trial+=1
                for variant in order:
                    out=a.output.with_suffix('.worker.json')
                    cmd=[sys.executable,__file__,'--worker','--output',str(out),'--variant',variant,'--seed',str(seed),'--lr',str(rate),
                         '--targets',str(a.targets),'--chunk',str(a.chunk),'--validation',str(a.validation),'--workload',a.workload,
                         '--cpu',str(a.cpu),'--worker-cpu',str(a.worker_cpu),'--nll-target',str(a.nll_target),'--logits' if a.logits else '--no-logits']
                    subprocess.run(cmd,check=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','SHOEBOX_INLINE_DIAGNOSTICS':'0'})
                    row=json.loads(out.read_text());row['repeat']=repeat;rows.append(row);out.unlink()
                    atomic_json(a.output,dict(completed=False,rows=rows))
                    print(variant,seed,rate,round(row['bytes_per_second'],1),round(row['heldout_nll'],5),row['time_to_target'],flush=True)
    atomic_json(a.output,dict(completed=True,rows=rows))
if __name__=='__main__':main()
