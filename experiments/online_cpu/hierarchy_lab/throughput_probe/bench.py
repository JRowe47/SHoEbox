import argparse,ctypes as ct,json,os,time
from pathlib import Path
import numpy as np
import model
ROOT=Path(__file__).resolve().parent
CATEGORIES=['byte overhead','source assembly','memory lookup/write','credit dispatch','local teacher','state input','routing','expert forward','publication','prototype updates','predictor forward','decoder forward','task teaching','readout update','expert weight update','credit Jacobian','bottom state learning','router task update','bottom router task','feature mask','state publication','key materialization']
p=argparse.ArgumentParser();p.add_argument('--kernel',default='baseline');p.add_argument('--repeats',type=int,default=3);p.add_argument('--targets',type=int,default=32768);p.add_argument('--warmup',type=int,default=2048);p.add_argument('--cpu',type=int,default=0);p.add_argument('--threads',type=int,default=1);p.add_argument('--cpus',default='0,2,4,6,8,1,3,5');p.add_argument('--label');a=p.parse_args()
os.sched_setaffinity(0,{a.cpu});model.KERNEL=a.kernel
size='1m' if model.core.E==12 else '5m';source=ROOT.parent/'utility_probe/results'/size/'seed1/center_split/last.npz'
data=np.frombuffer((ROOT.parent/'scale_probe/data/fresh.bin').read_bytes(),np.uint8)[2000000:]
samples=[];profiles=[]
for rep in range(a.repeats):
 m=model.Model.load(source)
 if a.kernel=='threaded':
  cpus=np.array([int(x) for x in a.cpus.split(',')][:a.threads],dtype=np.int32);m.lib.configure_threads.argtypes=[ct.c_void_p,ct.c_int,ct.POINTER(ct.c_int)];m.lib.configure_threads(m.handle,a.threads,cpus.ctypes.data_as(ct.POINTER(ct.c_int)))
 m.run(data[:a.warmup],data[1:a.warmup+1],mode='periodic',lr=.008,lp=.008/3,route=.002)
 if a.kernel.startswith('profile'):m.lib.profile_reset()
 start=time.perf_counter();cpu=time.process_time();nll=m.run(data[a.warmup:a.warmup+a.targets],data[a.warmup+1:a.warmup+a.targets+1],mode='periodic',lr=.008,lp=.008/3,route=.002)
 elapsed=time.perf_counter()-start;cpu=time.process_time()-cpu
 samples.append(dict(seconds=elapsed,cpu_seconds=cpu,bytes_per_second=a.targets/elapsed,nll=nll))
 if a.kernel.startswith('profile'):
  stats=np.zeros(2+2*len(CATEGORIES),np.uint64);m.lib.profile_export.argtypes=[ct.POINTER(ct.c_uint64)];m.lib.profile_export(stats.ctypes.data_as(ct.POINTER(ct.c_uint64)))
  total=int(stats[2:2+len(CATEGORIES)].sum());profiles.append(dict(bytes=int(stats[0]),sampled=int(stats[1]),stages={c:dict(ns=int(stats[2+i]),calls=int(stats[2+len(CATEGORIES)+i]),share=float(stats[2+i]/total)) for i,c in enumerate(CATEGORIES)}))
 m.close()
result=dict(size=size,kernel=a.kernel,threads=a.threads,targets=a.targets,warmup=a.warmup,median_bytes_per_second=float(np.median([v['bytes_per_second'] for v in samples])),samples=samples,profiles=profiles)
path=ROOT/'results'/f'{a.label or a.kernel}_{size}.json';path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
