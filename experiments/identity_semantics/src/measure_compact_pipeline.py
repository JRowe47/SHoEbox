"""Matched Stage-6 vs compact pipeline timing, with inspections outside run().

One stream, CPU-pinned, rotated execution order. Timed chunks include logits
and API quiescence; compilation, warmup, memory inspection, evaluation and IO
are excluded. No validation/benchmark processes should overlap this suite.
"""
import argparse,hashlib,json,os,resource,statistics,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def worker(a):
    import numpy as np
    sys.path.insert(0,str(a.runtime));from runtime import fresh_model,core
    cfg=json.loads(a.policy);m=fresh_model(seed=7,profile='pipeline')
    cpus=sorted(os.sched_getaffinity(0));m.configure_pipeline_workers(cfg.get('workers',1),cpu=cpus[1] if cfg.get('workers',1)==2 else None)
    os.sched_setaffinity(0,{cpus[0]})
    n=a.warmup+a.targets+1;text=b'The river runs through the valley. The king crossed the bridge. '
    raw=a.corpus.read_bytes()[:n] if a.corpus else (text*(n//len(text)+1))[:n]
    if len(raw)!=n:raise ValueError('Insufficient corpus')
    x=np.frombuffer(raw,np.uint8);m.run(x[:a.warmup],x[1:a.warmup+1],mode='periodic',lr=.04,lp=.04/3)
    points=[];seconds=0.;parts=[];weighted=0.
    for lo in range(a.warmup,n-1,512):
        hi=min(lo+512,n-1);start=time.perf_counter()
        nll,logits=m.run(x[lo:hi],x[lo+1:hi+1],mode='periodic',lr=.04,lp=.04/3,logits=True)
        elapsed=time.perf_counter()-start;seconds+=elapsed;weighted+=nll*(hi-lo);parts.append(logits)
        points.append(dict(bytes=hi-a.warmup,seconds=seconds,nll=nll))
    train_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory=m.inspect_pipeline() if hasattr(m,'inspect_pipeline') else None
    if memory:
        memory={k:v for k,v in memory.items() if k not in ('banks','building_buffer','pending_buffer')}
    valraw=a.validation.read_bytes()[:1025] if a.validation else (b'The bird crossed the valley. The queen walked through the gate. '*20)[:1025]
    validation=np.frombuffer(valraw,np.uint8);v=m.evaluation_copy(memory='cold_frozen')
    heldout=v.run(validation[:-1],validation[1:],mode='eval');v.close()
    params=dict(logits=np.concatenate(parts),flat=m.flat.copy(),extra=m.extra_flat.copy(),event=m.event_state(fitted=True))
    np.savez(a.output.with_suffix('.npz'),**params)
    sha=lambda array:hashlib.sha256(array.tobytes()).hexdigest()
    row=dict(config=cfg,experts=core.E,targets=a.targets,warmup=a.warmup,seconds=seconds,bytes_per_second=a.targets/seconds,
             train_nll=weighted/a.targets,heldout_nll=heldout,curve=points,memory=memory,pending_state_bytes=m.pipeline_state().size,
             train_peak_rss_kib=train_rss,hashes={k:sha(v) for k,v in params.items()},
             data_sha256=hashlib.sha256(raw).hexdigest(),validation_sha256=hashlib.sha256(valraw).hexdigest())
    m.close();a.output.write_text(json.dumps(row,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-runtime',type=Path);p.add_argument('--runtime',type=Path)
    p.add_argument('--corpus',type=Path);p.add_argument('--validation',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--targets',type=int,default=2048);p.add_argument('--warmup',type=int,default=1024);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--policy');p.add_argument('--worker',action='store_true');a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.worker:return worker(a)
    import numpy as np
    controls=[dict(name='stage6_serial',old=True),dict(name='compact_audit',audit=True),
              dict(name='compact_serial'),dict(name='stage6_worker',old=True,workers=2),dict(name='compact_worker',workers=2)]
    result=dict(protocol=__doc__,baseline='fbf7c816446289bcd987a85028d7cab59389b7ed',experts=int(os.environ.get('SHOEBOX_EXPERTS','12')),runs=[],comparisons=[])
    with tempfile.TemporaryDirectory(prefix='shoebox-compact-') as tmp:
        refs={}
        for repeat in range(a.repeats):
            order=controls[repeat%len(controls):]+controls[:repeat%len(controls)]
            if repeat%2:order=list(reversed(order))
            for cfg in order:
                runtime=a.baseline_runtime if cfg.get('old') else ROOT/'runtime';out=Path(tmp)/'run.json'
                cmd=[sys.executable,__file__,'--worker','--runtime',str(runtime),'--output',str(out),'--policy',json.dumps(cfg),
                     '--targets',str(a.targets),'--warmup',str(a.warmup)]
                for name in ('corpus','validation'):
                    if getattr(a,name):cmd+=['--'+name,str(getattr(a,name))]
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','SHOEBOX_INLINE_DIAGNOSTICS':'1' if cfg.get('audit') else '0'}
                subprocess.run(cmd,check=True,env=env)
                row=json.loads(out.read_text());row['repeat']=repeat;result['runs'].append(row)
                with np.load(out.with_suffix('.npz')) as arrays:
                    data={k:arrays[k].copy() for k in arrays.files}
                if cfg['name'] not in refs:refs[cfg['name']]=data
                else:
                    for k in data:np.testing.assert_array_equal(refs[cfg['name']][k],data[k])
                a.output.write_text(json.dumps(result,indent=2)+'\n');print(repeat,cfg['name'],round(row['bytes_per_second'],1),'B/s',flush=True)
        for name in refs:
            changes={}
            for key in refs[name]:
                aa=refs['stage6_serial'][key];bb=refs[name][key];diff=np.abs(aa.astype(float)-bb.astype(float))
                changes[key]=dict(bitwise_equal=bool(np.array_equal(aa,bb)),max_abs=float(diff.max(initial=0)))
            result['comparisons'].append(dict(variant=name,vs_stage6=changes))
        for pair in [('stage6_serial','stage6_worker'),('compact_serial','compact_worker')]:
            for key in refs[pair[0]]:np.testing.assert_array_equal(refs[pair[0]][key],refs[pair[1]][key])
        result['serial_worker_bitwise']=True
        result['medians']={c['name']:statistics.median(r['bytes_per_second'] for r in result['runs'] if r['config']['name']==c['name']) for c in controls}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
