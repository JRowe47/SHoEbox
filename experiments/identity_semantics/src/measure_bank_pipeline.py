"""Matched stage-5 / stage-6 schedules, one stream, one or two pinned CPUs.

Worker creation, compilation, warmup, evaluation and IO are excluded. Training
timings include API quiescence and requested logit export. No other test or
benchmark process should run during this suite. Each subprocess has fresh
seed-7 parameters. Execution variants must have identical final fingerprints.
"""
import argparse,hashlib,json,os,resource,statistics,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'));sys.path.insert(0,str(ROOT/'src'))
from compare_optimized_runtime import TRAIN,source_hash

def worker(a):
    import numpy as np
    from runtime import fresh_model
    from reference_acceptance import collect,check_integrity
    from run_reference_acceptance import generate
    cfg=json.loads(a.policy);profile=cfg['profile'];workers=cfg.get('workers',1);scale=cfg.get('scale',1.)
    # Configure the background thread's affinity before pinning the caller.
    os.sched_setaffinity(0,{a.cpu,a.learning_cpu})
    m=fresh_model(seed=7,profile=profile)
    if profile=='pipeline':m.configure_pipeline_workers(workers,cpu=a.learning_cpu if workers==2 else None,fused=cfg.get('fused',True))
    os.sched_setaffinity(0,{a.cpu})
    n=a.warmup+a.targets+1
    raw=a.corpus.read_bytes()[:n] if a.corpus else (TRAIN*(n//len(TRAIN)+1))[:n]
    valraw=a.validation.read_bytes()[:1025] if a.validation else (b'The bird crossed the valley. The queen walked through the gate. '*20)[:1025]
    if len(raw)!=n:raise ValueError('Insufficient training data')
    x=np.frombuffer(raw,np.uint8);val=np.frombuffer(valraw,np.uint8)
    m.run(x[:a.warmup],x[1:a.warmup+1],mode='periodic',lr=.04*scale,lp=.04*scale/3)
    for method in ('reset_pipeline_metrics','reset_event_metrics','reset_reverse_metrics'):getattr(m,method)()
    points=[];seconds=0.;weighted=0.
    for offset in range(0,a.targets,512):
        lo=a.warmup+offset;k=min(512,a.targets-offset);start=time.perf_counter()
        result=m.run(x[lo:lo+k],x[lo+1:lo+k+1],mode='periodic',lr=.04*scale,lp=.04*scale/3,logits=a.logits)
        elapsed=time.perf_counter()-start;nll=result[0] if a.logits else result
        seconds+=elapsed;weighted+=float(nll)*k
        points.append(dict(targets=offset+k,seconds=seconds,train_nll=float(nll)))
    check_integrity(collect(m));evaluation=m.evaluation_copy(memory='cold_frozen')
    heldout=float(evaluation.run(val[:-1],val[1:],mode='eval'));evaluation.close()
    row=dict(config=cfg,targets=a.targets,warmup=a.warmup,seconds=seconds,bytes_per_second=a.targets/seconds,
             train_nll=weighted/a.targets,heldout_nll=heldout,curve=points,pipeline_metrics=m.pipeline_metrics(),
             event_metrics=m.event_metrics(),reverse_metrics=m.reverse_metrics(),credit_integrity=True,
             fingerprint=m.fingerprint(),generation=generate(m,b'The river ',count=64),
             pending_state_bytes=m.pipeline_state().size,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             train_sha256=hashlib.sha256(raw).hexdigest(),validation_sha256=hashlib.sha256(valraw).hexdigest())
    m.close();a.output.write_text(json.dumps(row,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus',type=Path);p.add_argument('--validation',type=Path)
    p.add_argument('--warmup',type=int,default=1024);p.add_argument('--targets',type=int,default=1024)
    cpus=sorted(os.sched_getaffinity(0));p.add_argument('--cpu',type=int,default=cpus[0]);p.add_argument('--learning-cpu',type=int,default=cpus[1])
    p.add_argument('--policy');p.add_argument('--worker',action='store_true');p.add_argument('--logits',action='store_true')
    p.add_argument('--suite',choices=['paired','sweep'],default='paired');p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.worker:return worker(a)
    controls=[dict(name='stage5',profile='event'),dict(name='serial_unfused',profile='pipeline',fused=False),
              dict(name='serial_fused',profile='pipeline'),dict(name='two_workers',profile='pipeline',workers=2)]
    if a.suite=='sweep':configs=[dict(name=f'{profile}_{scale:g}',profile=profile,scale=scale) for scale in (.5,1.,2.) for profile in ('event','pipeline')]
    else:
        configs=[]
        for repeat in range(a.repeats):
            order=controls[repeat%4:]+controls[:repeat%4]
            if repeat%2:order=list(reversed(order))
            configs += [dict(c,repeat=repeat) for c in order]
    result=dict(protocol=__doc__,cpu=a.cpu,learning_cpu=a.learning_cpu,experts=int(os.environ.get('SHOEBOX_EXPERTS','12')),
                source_sha256=source_hash(ROOT/'runtime'),logits=a.logits,runs=[])
    with tempfile.TemporaryDirectory(prefix='shoebox-stage6-') as tmp:
        for i,cfg in enumerate(configs):
            output=Path(tmp)/'run.json'
            cmd=[sys.executable,__file__,'--worker','--output',str(output),'--policy',json.dumps(cfg),'--warmup',str(a.warmup),
                 '--targets',str(a.targets),'--cpu',str(a.cpu),'--learning-cpu',str(a.learning_cpu)]
            if a.corpus:cmd+=['--corpus',str(a.corpus)]
            if a.validation:cmd+=['--validation',str(a.validation)]
            if a.logits:cmd+=['--logits']
            subprocess.run(cmd,check=True);row=json.loads(output.read_text());result['runs'].append(row)
            a.output.write_text(json.dumps(result,indent=2)+'\n')
            print(i+1,len(configs),cfg['name'],round(row['bytes_per_second'],1),'B/s',round(row['heldout_nll'],6),'heldout',flush=True)
    if a.suite=='paired':
        fingerprints={r['fingerprint'] for r in result['runs'] if r['config']['profile']=='pipeline'}
        if len(fingerprints)!=1:raise AssertionError('Execution variants changed the training schedule')
        result['bitwise_pipeline_parity']=True
        result['medians']={c['name']:statistics.median(r['bytes_per_second'] for r in result['runs'] if r['config']['name']==c['name']) for c in controls}
    a.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
