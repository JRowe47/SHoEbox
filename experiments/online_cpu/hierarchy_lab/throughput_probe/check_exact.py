import argparse,ctypes as ct,json,os
from pathlib import Path
import numpy as np
import model
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--kernel',default='mask_radix');p.add_argument('--targets',type=int,default=32768);p.add_argument('--threads',type=int,default=1);p.add_argument('--label');a=p.parse_args()
os.sched_setaffinity(0,{0})
size='1m' if model.core.E==12 else '5m';source=ROOT.parent/'utility_probe/results'/size/'seed1/center_split/last.npz'
model.KERNEL='baseline';reference=model.Model.load(source)
model.KERNEL=a.kernel;candidate=model.Model.load(source)
if a.kernel=='threaded':
 cpus=np.arange(a.threads,dtype=np.int32);candidate.lib.configure_threads.argtypes=[ct.c_void_p,ct.c_int,ct.POINTER(ct.c_int)];candidate.lib.configure_threads(candidate.handle,a.threads,cpus.ctypes.data_as(ct.POINTER(ct.c_int)))
data=np.frombuffer((ROOT.parent/'scale_probe/data/fresh.bin').read_bytes(),np.uint8)[2000000:]
rows=[]
for pos in range(0,a.targets,4096):
 n=min(4096,a.targets-pos);args=(data[pos:pos+n],data[pos+1:pos+n+1]);kwargs=dict(mode='periodic',lr=.008,lp=.008/3,route=.002)
 losses=[m.run(*args,**kwargs) for m in [reference,candidate]]
 equal=reference.fingerprint()==candidate.fingerprint()
 rows.append(dict(targets=pos+n,losses=losses,exact=equal))
 if not equal:
  rows[-1]['max_parameter_delta']=float(np.max(np.abs(reference.flat-candidate.flat)))
  rows[-1]['differing_parameters']=int(np.count_nonzero(reference.flat!=candidate.flat))
  for name,fn in [('state',lambda m:model.parent.legacy.base.state(m)),('laces',lambda m:m.lace_state()),('learned',lambda m:m.learned_state()),('probe',lambda m:m.probe_state()),('utility',lambda m:m.utility_state())]:
   rows[-1][name+'_different_bytes']=int(np.count_nonzero(fn(reference)!=fn(candidate)))
  break
result=dict(size=size,kernel=a.kernel,threads=a.threads,compiler_flags=model.FLAGS,requested_targets=a.targets,all_exact=all(r['exact'] for r in rows),rows=rows)
(ROOT/'results'/f'exact_{size}_{a.label or a.kernel}_{a.threads}.json').write_text(json.dumps(result,indent=2)+'\n')
reference.close();candidate.close();print(json.dumps(result));assert result['all_exact']
