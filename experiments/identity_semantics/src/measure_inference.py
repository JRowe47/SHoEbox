"""Isolated frozen-runtime vs forward-only timing; no inline instrumentation."""
import argparse,ctypes,gc,hashlib,json,os,platform,statistics,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime'),str(ROOT/'src')]
from measure_current_baseline import fixtures,source_manifest,rss_kib,atomic_json

def worker(a):
 import numpy as np
 from runtime import fresh_model,core
 os.sched_setaffinity(0,{a.cpu})
 kw={} if a.profile=='reference' else dict(profile='epochs',bank_epochs=dict(epoch_bytes=4,lag_epochs=1),prediction=dict(compact=True,rank=16,slow_stride=8))
 source=fresh_model(seed=7,**kw)
 docs,val,meta=fixtures('repeated' if a.workload=='repeated' else 'natural',2048+1024+a.targets,validation_bytes=1024)
 text=np.frombuffer(docs[0],np.uint8)
 source.run(text[:2048],text[1:2049],mode='periodic',lr=.04,lp=.04/3)
 source.finish_training()
 m=source.inference_copy(memory='warm_adaptive') if a.runtime=='forward' else source.evaluation_copy(memory='warm_adaptive')
 source.close();del source;gc.collect()
 ctypes.CDLL(None).malloc_trim(0)
 if m.capabilities()['inline_diagnostics']:raise RuntimeError('Production runtime required')
 m.run(text[2048:3072],text[2049:3073],logits=True)
 samples=[dict(bytes=0,rss_kib=rss_kib(),storage=m.inspect_inference())]
 digest=hashlib.sha256();elapsed=cpu=loss=0.;count=0
 if a.workload=='generation':
  rng=np.random.default_rng(876)
  start=time.perf_counter();cstart=time.process_time()
  generated=[]
  for i in range(a.targets):
   byte=m.draw(rng,temperature=.8);generated.append(byte);m.run([byte])
  elapsed=time.perf_counter()-start;cpu=time.process_time()-cstart;digest.update(bytes(generated));count=a.targets
 else:
  for lo in range(3072,len(text)-1,1024):
   hi=min(lo+1024,len(text)-1)
   start=time.perf_counter();cstart=time.process_time()
   nll,out=m.run(text[lo:hi],text[lo+1:hi+1],logits=True)
   elapsed+=time.perf_counter()-start;cpu+=time.process_time()-cstart
   count+=hi-lo;loss+=nll*(hi-lo);digest.update(out.tobytes())
   if count in (8192,16384,32768):samples.append(dict(bytes=count,rss_kib=rss_kib(),storage=m.inspect_inference()))
 samples.append(dict(bytes=count,rss_kib=rss_kib(),storage=m.inspect_inference()))
 # A separate held-out frozen copy measures values, outside every timed call.
 ev=m.inference_copy(memory='cold_frozen') if a.runtime=='forward' else m.evaluation_copy(memory='cold_frozen')
 v=np.frombuffer(val,np.uint8);heldout=float(ev.run(v[:-1],v[1:]));ev.close()
 result=dict(profile=a.profile,runtime=a.runtime,workload=a.workload,targets=count,experts=core.E,
             fitted_parameters=m.parameter_count,elapsed=elapsed,process_cpu_seconds=cpu,bytes_per_second=count/elapsed,
             continuation_nll=loss/count if a.workload!='generation' else None,heldout_nll=heldout,
             output_sha256=digest.hexdigest(),memory=samples,data=meta,cpu=a.cpu,repeat=a.repeat,
             protocol=dict(training_targets=2048,warmup_targets=1024,logits=a.workload!='generation',memory='warm_adaptive',generation_temperature=.8))
 m.close();atomic_json(a.output,result)

def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--runtime',choices=['frozen','forward'],default='forward')
 p.add_argument('--profile',choices=['reference','selected'],default='selected');p.add_argument('--workload',choices=['natural','repeated','generation'],default='natural')
 p.add_argument('--targets',type=int,default=8192);p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
 p.add_argument('--repeat',type=int,default=0);p.add_argument('--repeats',type=int,default=3);p.add_argument('--profiles',nargs='+',default=['reference','selected'])
 p.add_argument('--workloads',nargs='+',default=['natural','repeated','generation']);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.worker:return worker(a)
 rows=[]
 with tempfile.TemporaryDirectory() as d:
  for repeat in range(a.repeats):
   for profile in a.profiles:
    for workload in a.workloads:
     for runtime in (['frozen','forward'] if repeat%2==0 else ['forward','frozen']):
      path=Path(d)/'row.json';targets=min(1024,a.targets) if workload=='generation' else a.targets
      subprocess.run([sys.executable,__file__,'--worker','--profile',profile,'--runtime',runtime,'--workload',workload,
        '--targets',str(targets),'--cpu',str(a.cpu),'--repeat',str(repeat),'--output',str(path)],check=True)
      row=json.loads(path.read_text());rows.append(row)
      atomic_json(a.output,dict(completed=False,rows=rows))
      print(profile,workload,runtime,repeat,round(row['bytes_per_second'],1),round(row['memory'][-1]['rss_kib']/1024,1),flush=True)
 summary=[]
 for profile in a.profiles:
  for workload in a.workloads:
   pair=[r for r in rows if r['profile']==profile and r['workload']==workload]
   for repeat in range(a.repeats):
    ab=[r for r in pair if r['repeat']==repeat]
    assert ab[0]['output_sha256']==ab[1]['output_sha256'],(profile,workload,repeat)
    assert ab[0]['heldout_nll']==ab[1]['heldout_nll']
   med=lambda runtime,fn:statistics.median(fn(r) for r in pair if r['runtime']==runtime)
   before=med('frozen',lambda r:r['bytes_per_second']);after=med('forward',lambda r:r['bytes_per_second'])
   bm=med('frozen',lambda r:r['memory'][-1]['rss_kib']/1024);am=med('forward',lambda r:r['memory'][-1]['rss_kib']/1024)
   summary.append(dict(profile=profile,workload=workload,frozen_bps=before,forward_bps=after,speedup=after/before,
                       frozen_rss_mib=bm,forward_rss_mib=am,rss_reduction_percent=100*(1-am/bm),
                       outputs_identical=True,heldout_nll=pair[0]['heldout_nll']))
 environment=dict(cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown'),
                  compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],python=platform.python_version(),flags='-O3 -march=native -std=c++17 -pthread')
 atomic_json(a.output,dict(completed=True,rows=rows,summary=summary,environment=environment,source=source_manifest()))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
