import ctypes as ct,json,os,time
from pathlib import Path
import numpy as np
import model
ROOT=Path(__file__).resolve().parent;os.sched_setaffinity(0,{0});model.FLAGS=['-pthread']
size='1m' if model.core.E==12 else '5m';source=ROOT.parent/'utility_probe/results'/size/'seed1/center_split/last.npz'
data=np.frombuffer((ROOT.parent/'scale_probe/data/fresh.bin').read_bytes(),np.uint8)[2000000:]
variants=[('baseline',1),('mask_radix',1),('router',1),('cached',1),('threaded',1),('threaded',2),('threaded',4)]
for k,n in variants:model.KERNEL=k;model.build()
results={f'{k}_{n}':[] for k,n in variants};targets=32768;warm=4096
for rep in range(5):
 order=variants[rep%len(variants):]+variants[:rep%len(variants)]
 if rep%2:order=order[::-1]
 for k,n in order:
  model.KERNEL=k;m=model.Model.load(source)
  if k=='threaded':
   cpus=np.array([0,2,4,6][:n],np.int32);m.lib.configure_threads.argtypes=[ct.c_void_p,ct.c_int,ct.POINTER(ct.c_int)];m.lib.configure_threads(m.handle,n,cpus.ctypes.data_as(ct.POINTER(ct.c_int)))
  m.run(data[:warm],data[1:warm+1],mode='periodic',lr=.008,lp=.008/3,route=.002)
  start=time.perf_counter();cpu=time.process_time();nll=m.run(data[warm:warm+targets],data[warm+1:warm+targets+1],mode='periodic',lr=.008,lp=.008/3,route=.002)
  elapsed=time.perf_counter()-start;cpu=time.process_time()-cpu
  results[f'{k}_{n}'].append(dict(repeat=rep,seconds=elapsed,cpu_seconds=cpu,bytes_per_second=targets/elapsed,nll=nll));m.close()
  print(size,rep,k,n,round(targets/elapsed),flush=True)
all_losses=[r['nll'] for v in results.values() for r in v];assert len(set(all_losses))==1
out=dict(size=size,targets=targets,warmup=warm,repeats=5,cpus=[0,2,4,6],samples=results,summary={k:dict(median=float(np.median([r['bytes_per_second'] for r in v])),min=min(r['bytes_per_second'] for r in v),max=max(r['bytes_per_second'] for r in v),median_cpu_per_wall=float(np.median([r['cpu_seconds']/r['seconds'] for r in v]))) for k,v in results.items()},identical_losses=True)
(ROOT/'results'/f'paired_{size}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary']),flush=True)
