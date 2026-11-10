import argparse,ctypes as ct,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
MAIN=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('phase',choices=['continued','fresh1m','fresh3m','allocation']);p.add_argument('profile');p.add_argument('--benchmark',action='store_true');a=p.parse_args()
if a.phase=='allocation':sys.path.insert(0,str(MAIN.parent/'allocation_probe'))
from model import Model,core
SIZE=os.environ.get('PROBE_SIZE','1m' if core.E==12 else '5m')
os.sched_setaffinity(0,{int(os.environ.get('PROBE_CPU','8'))})
OUT=MAIN/'results/evaluation';OUT.mkdir(exist_ok=True)
def checkpoint(phase,name):
 if phase=='continued':return MAIN/'results/continued'/SIZE/name/'last.npz'
 if phase=='allocation':return MAIN/'results/allocation'/SIZE/name/'last.npz'
 d=MAIN/'results/fresh'/SIZE/name
 if phase=='fresh1m' and (d/'at_1m.npz').exists():return d/'at_1m.npz'
 return d/'last.npz'
def document_score(model,data,documents):
 m=model.frozen_copy();rows=[];pos=0
 for d in documents:
  lo=max(0,d['start']-1);hi=min(len(data)-1,d['start']+d['bytes_used']-1);assert lo==pos
  if hi>lo:rows.append({'id':d['id'],'targets':hi-lo,'nll':m.run(data[lo:hi],data[lo+1:hi+1])})
  pos=hi
 assert pos==len(data)-1
 counts=np.array(m.stats()['expert_counts']);prob=counts/np.maximum(1,counts.sum(1,keepdims=True));effective=np.exp(-(prob*np.log(np.maximum(prob,1e-30))).sum(1))
 out={'nll':float(np.average([d['nll'] for d in rows],weights=[d['targets'] for d in rows])),'documents':rows,
  'effective_experts':dict(zip(core.NAMES,effective.tolist()))};m.close();return out
def mean_y(model):
 m=model.frozen_copy();m.lib.diagnostic_enable(m.handle,1)
 data=np.frombuffer((MAIN/'data/fresh.bin').read_bytes(),np.uint8)[:1000001][-65537:];m.run(data[:-1],data[1:])
 arr=np.zeros(18+18*4*64+18*4*64*64+18*8,np.float64);m.lib.diagnostic_export(m.handle,arr.ctypes.data_as(ct.POINTER(ct.c_double)))
 mean=(arr[18:18+18*4*64].reshape(18,4,64)[17,3]/max(1,arr[17])).astype(np.float32);m.close();return mean
def measure():
 assert (MAIN/'results/final_selection.json').exists(),'Test sealed until validation decision recorded.'
 src=checkpoint(a.phase,a.profile);m=Model.load(src);before=m.fingerprint()+hashlib.sha256(m.probe_state()).hexdigest()
 data=np.frombuffer((MAIN/'data/test.bin').read_bytes(),np.uint8);manifest=json.loads((MAIN/'data/manifest.json').read_text())
 out={'phase':a.phase,'size':SIZE,'profile':a.profile,'parameters':m.parameter_count,'checkpoint_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
      'test':document_score(m,data,manifest['heldout']['test']['documents'])}
 mean=mean_y(m);f=m.frozen_copy();f.params['rbias']+=f.params['readout'][:,64:]@mean;f.params['readout'][:,64:]=0
 out['mean_Y_test']=document_score(f,data,manifest['heldout']['test']['documents']);f.close()
 assert before==m.fingerprint()+hashlib.sha256(m.probe_state()).hexdigest();out['trainer_preserved']=True
 assert np.isfinite(m.flat).all() and np.isfinite(m.extra_flat).all();out['finite']=True;m.close()
 (OUT/f'{a.phase}_{SIZE}_{a.profile}.json').write_text(json.dumps(out,indent=2))
 print(json.dumps({'phase':a.phase,'size':SIZE,'profile':a.profile,'test':out['test']['nll'],'mean_Y_delta':out['mean_Y_test']['nll']-out['test']['nll']}),flush=True)
def benchmark():
 names=a.profile.split(',');samples={n:[] for n in names};data=np.frombuffer((MAIN/'data/train.bin').read_bytes(),np.uint8)[2000000:]
 for rep in range(5):
  for n in names[::1 if rep%2==0 else -1]:
   m=Model.load(checkpoint(a.phase,n));m.run(data[:2048],data[1:2049],mode='periodic',lr=.008,lp=.008/3,route=.002)
   t=time.perf_counter();m.run(data[2048:18432],data[2049:18433],mode='periodic',lr=.008,lp=.008/3,route=.002);samples[n].append(16384/(time.perf_counter()-t));m.close()
 out={k:{'samples':v,'median':float(np.median(v))} for k,v in samples.items()}
 (OUT/f'benchmark_{a.phase}_{SIZE}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
benchmark() if a.benchmark else measure()
