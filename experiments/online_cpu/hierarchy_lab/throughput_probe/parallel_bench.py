import concurrent.futures,json,os,threading,time
from pathlib import Path
import numpy as np
import model
ROOT=Path(__file__).resolve().parent;size='1m' if model.core.E==12 else '5m'
source=ROOT.parent/'utility_probe/results'/size/'seed1/center_split/last.npz'
data=np.frombuffer((ROOT.parent/'scale_probe/data/fresh.bin').read_bytes(),np.uint8)[2000000:]
CPUS=[0,2,4,6,8,1,3,5];targets=65536;warm=4096;records=[]
# Each thread owns one complete independent trainer; ctypes releases the GIL.
# This measures aggregate experiment throughput, never shared-model batching.
def worker(kernel,cpu,barrier):
 os.sched_setaffinity(0,{cpu})
 m=model.Model.load(source)  # Global kernel is fixed for this entire cohort.
 m.run(data[:warm],data[1:warm+1],mode='periodic',lr=.008,lp=.008/3,route=.002)
 barrier.wait(timeout=60);start=time.perf_counter();cpu_start=time.thread_time()
 nll=m.run(data[warm:warm+targets],data[warm+1:warm+targets+1],mode='periodic',lr=.008,lp=.008/3,route=.002)
 end=time.perf_counter();cpu_seconds=time.thread_time()-cpu_start;m.close()
 return dict(cpu=cpu,start=start,end=end,cpu_seconds=cpu_seconds,nll=nll)
for rep in range(2):
 configurations=[('baseline',1),('baseline',5),('baseline',8),('cached',1),('cached',2),('cached',4),('cached',5),('cached',8)]
 if rep:configurations=configurations[::-1]
 for kernel,count in configurations:
  model.KERNEL=kernel;model.build();barrier=threading.Barrier(count)
  with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:rows=list(pool.map(lambda cpu:worker(kernel,cpu,barrier),CPUS[:count]))
  assert len(set(r['nll'] for r in rows))==1
  seconds=max(r['end'] for r in rows)-min(r['start'] for r in rows)
  item=dict(kernel=kernel,workers=count,repeat=rep,targets_per_worker=targets,seconds=seconds,aggregate_bytes_per_second=count*targets/seconds,rows=rows);records.append(item)
  print(size,kernel,count,rep,round(item['aggregate_bytes_per_second']),flush=True)
assert len(set(r['rows'][0]['nll'] for r in records))==1
summary={f'{k}_{n}':float(np.median([r['aggregate_bytes_per_second'] for r in records if r['kernel']==k and r['workers']==n])) for k,n in configurations}
(ROOT/'results'/f'parallel_{size}.json').write_text(json.dumps(dict(size=size,cpus=CPUS,records=records,summary=summary,independent_models=True),indent=2)+'\n');print(json.dumps(summary),flush=True)
