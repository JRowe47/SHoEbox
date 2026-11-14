"""Matched stage-3/4 continuations with numerical drift and routing comparisons.

Timings exclude compilation, checkpoints, verification, route tracing and hashes.
Snapshots stay in a temporary directory; only compact measurements are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from compare_optimized_runtime import TRAIN,source_hash


def digest(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def worker(a):
    import numpy as np
    sys.path.insert(0,str(a.runtime.resolve()))
    from runtime import Model,fresh_model,core
    from reference_acceptance import collect,check_integrity
    os.sched_setaffinity(0,{a.cpu})
    length=a.warmup+a.targets+1
    raw=a.corpus.read_bytes()[:length] if a.corpus else (TRAIN*(length//len(TRAIN)+1))[:length]
    if len(raw)!=length:raise ValueError('Insufficient corpus bytes')
    data=np.frombuffer(raw,np.uint8)
    if a.initialize:
        m=fresh_model(seed=7)
        m.run(data[:a.warmup],data[1:a.warmup+1],mode='periodic',lr=.04,lp=.04/3)
        m.save(a.checkpoint);m.close();return
    m=Model.load(a.checkpoint)
    if hasattr(m,'configure_reverse') and a.new_policy:m.configure_reverse(dict(policy='union'))
    if a.roundtrip:
        m.save(a.save_after);m.close();return
    initial=m.flat.copy()
    start=time.perf_counter();cpu_start=time.process_time()
    nll,logits=m.run(data[a.warmup:-1],data[a.warmup+1:],mode=a.mode,lr=.04,lp=.04/3,logits=True)
    seconds=time.perf_counter()-start;cpu_seconds=time.process_time()-cpu_start
    check_integrity(collect(m))
    arrays=dict(logits=logits,main=m.flat.copy(),extra=m.extra_flat.copy(),routes=m.codes(),
                initial=initial,composition=m.composition_state(fitted=True))
    content=m.content_expert_state()
    version,count=struct.unpack_from('<QQ',content)
    if version!=1 or len(content)!=16+count*808:raise AssertionError('Unexpected content record ABI')
    dtype=np.dtype({'names':['serial','region','index','expert','runner','mask'],
                    'formats':['<u8','<i4','<i4','<i4','<i4','<u8'],
                    'offsets':[0,8,12,16,20,24],'itemsize':808})
    records=np.frombuffer(content,dtype=dtype,offset=16)
    arrays['content_routes']=np.column_stack([records[name].astype(np.uint64) for name in dtype.names])
    row=dict(experts=core.E,parameters=m.parameter_count,targets=a.targets,nll=float(nll),
             seconds=seconds,cpu_seconds=cpu_seconds,bytes_per_second=a.targets/seconds,
             data_sha256=hashlib.sha256(raw).hexdigest(),fingerprint=m.fingerprint(),
             credit_integrity=True,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             hashes={k:digest(v) for k,v in arrays.items()},
             factor_metrics=m.content_factor_metrics() if hasattr(m,'content_factor_metrics') else {},
             reverse_metrics=m.reverse_metrics() if hasattr(m,'reverse_metrics') else {})
    if a.trace:
        # Untimed replay validates every byte's current regional expert and
        # masks; this includes held decisions for regions that did not execute.
        expected=m.fingerprint();m.close();m=Model.load(a.checkpoint);routes=[];trace_logits=[]
        if hasattr(m,'configure_reverse') and a.new_policy:m.configure_reverse(dict(policy='union'))
        for i in range(a.warmup,len(data)-1):
            _,l=m.run(data[i:i+1],data[i+1:i+2],mode=a.mode,lr=.04,lp=.04/3,logits=True)
            routes.append(m.codes());trace_logits.append(l)
        np.testing.assert_array_equal(np.concatenate(trace_logits),logits)
        if m.fingerprint()!=expected:raise AssertionError('Chunk boundary changed state')
        arrays['route_trace']=np.stack(routes);row['route_trace_sha256']=digest(arrays['route_trace'])
        row['chunk_parity']=True
    if a.save_after:m.save(a.save_after)
    np.savez(a.snapshot,**arrays)
    m.close();a.output.write_text(json.dumps(row,indent=2)+'\n')


def difference(a,b):
    import numpy as np
    delta=b.astype(np.float64)-a.astype(np.float64)
    return dict(max_abs=float(np.max(np.abs(delta),initial=0)),rms=float(np.sqrt(np.mean(delta*delta))),
                relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(a.astype(float)),1e-30)))


def compare(old,new,oldrow,newrow):
    import numpy as np
    result={k:difference(old[k],new[k]) for k in ('logits','main','extra')}
    result['nll_delta']=newrow['nll']-oldrow['nll']
    result['main_update_drift_relative_l2']=float(np.linalg.norm(new['main'].astype(float)-old['main'])/
        max(np.linalg.norm(old['main'].astype(float)-old['initial']),1e-30))
    result['top_byte_disagreements']=int(np.count_nonzero(old['logits'].argmax(-1)!=new['logits'].argmax(-1)))
    for key in ('routes','route_trace'):
        if key in old and key in new:
            result[key]=dict(expert_disagreements=int(np.count_nonzero(old[key][...,2]!=new[key][...,2])),
                             mask_disagreements=int(np.count_nonzero(old[key][...,:2]!=new[key][...,:2])),
                             regional_decisions=int(old[key][...,2].size))
    left={tuple(row[:3]):tuple(row[3:]) for row in old['content_routes']}
    right={tuple(row[:3]):tuple(row[3:]) for row in new['content_routes']}
    common=left.keys()&right.keys()
    result['retained_content_routes']=dict(common_handles=len(common),old_only=len(left.keys()-right.keys()),
        new_only=len(right.keys()-left.keys()),expert_disagreements=int(sum(left[k][0]!=right[k][0] for k in common)),
        runner_disagreements=int(sum(left[k][1]!=right[k][1] for k in common)),mask_disagreements=int(sum(left[k][2]!=right[k][2] for k in common)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-runtime',type=Path)
    p.add_argument('--runtime',type=Path,default=Path(__file__).resolve().parents[1]/'runtime')
    p.add_argument('--warmup',type=int,default=1024);p.add_argument('--targets',type=int,default=1024)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--mode',choices=('periodic','eval'),default='periodic')
    p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
    p.add_argument('--corpus',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--initialize',action='store_true')
    p.add_argument('--trace',action='store_true');p.add_argument('--checkpoint',type=Path)
    p.add_argument('--snapshot',type=Path);p.add_argument('--save-after',type=Path)
    p.add_argument('--roundtrip',action='store_true');p.add_argument('--new-policy',action='store_true');p.add_argument('--root-control',action='store_true')
    a=p.parse_args()
    if a.worker:return worker(a)
    if a.baseline_runtime is None:p.error('--baseline-runtime required')
    if min(a.warmup,a.targets,a.repeats)<1:p.error('Positive counts required')
    import numpy as np
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    result=dict(protocol=__doc__,warmup=a.warmup,targets=a.targets,repeats=a.repeats,cpu=a.cpu,
                mode=a.mode,workload='natural' if a.corpus else 'repeated',runs={'stage3':[],'stage4':[]},
                source_sha256={k:source_hash(v) for k,v in [('stage3',a.baseline_runtime),('stage4',a.runtime)]})
    result['cpu_model']=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                              if line.startswith('model name')),'unknown')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='shoebox-union-') as temporary:
        folder=Path(temporary);checkpoint=folder/'base.npz';output=folder/'row.json'
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--checkpoint',str(checkpoint),
                 '--warmup',str(a.warmup),'--targets',str(a.targets),'--cpu',str(a.cpu),'--mode',a.mode,'--output',str(output)]
        if a.corpus:command+=['--corpus',str(a.corpus.resolve())]
        subprocess.run(command+['--runtime',str(a.baseline_runtime.resolve()),'--initialize'],env=env,check=True)
        for repeat in range(a.repeats):
            variants=[('stage3',a.baseline_runtime),('stage4',a.runtime)]
            for name,runtime in variants[::1 if repeat%2==0 else -1]:
                snapshot=folder/(name+'.npz')
                args=command+['--runtime',str(runtime.resolve()),'--snapshot',str(snapshot)]
                if name=='stage4' and not a.root_control:args+=['--new-policy']
                if repeat==a.repeats-1:args+=['--trace']
                subprocess.run(args,env=env,check=True)
                row=json.loads(output.read_text());result['runs'][name].append(row)
                print(name,repeat+1,round(row['bytes_per_second'],2),'B/s',flush=True)
                a.output.write_text(json.dumps(result,indent=2)+'\n')
        with np.load(folder/'stage3.npz') as old,np.load(folder/'stage4.npz') as new:
            result['comparison']=compare(old,new,result['runs']['stage3'][-1],result['runs']['stage4'][-1])
        for name,rows in result['runs'].items():
            if len({r['fingerprint'] for r in rows})!=1:raise AssertionError(name+' nondeterministic')
        result['medians']={name:statistics.median(r['bytes_per_second'] for r in rows) for name,rows in result['runs'].items()}
        result['speedup']=result['medians']['stage4']/result['medians']['stage3']
        result['deterministic']=True
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'speedup':result['speedup'],'comparison':result['comparison'],
                          'factors':result['runs']['stage4'][-1]['factor_metrics']},indent=2),flush=True)


if __name__=='__main__':main()
