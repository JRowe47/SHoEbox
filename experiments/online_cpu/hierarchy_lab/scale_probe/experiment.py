import argparse,ctypes as ct,hashlib,json,os,time,shutil
from pathlib import Path
import numpy as np
from model import ROOT,Model,core,evaluate
SIZE='1m' if core.E==12 else '5m'
PROFILES={
 'baseline':{},'context0':{'context_gain':0},'context05':{'context_gain':.5},'context10':{'context_gain':1.0},
 'route_center':{'route_mode':1,'center_rate':.001},
 'route_center_full':{'route_mode':1,'center_rate':.001,'dense':True},
 'source_center':{'route_mode':2,'center_rate':.001},
 'route_reset':{'reset_keys':True},
 'source_center_reset':{'route_mode':2,'center_rate':.001,'reset_keys':True},
 'separated_ports':{'separate_ports':True},
 'raw_full':{'dense':True},
}
TRAIN=np.frombuffer((ROOT/'data/train.bin').read_bytes(),np.uint8)
VAL=np.frombuffer((ROOT/'data/validation.bin').read_bytes(),np.uint8)
RESULTS=ROOT/'results'
def atomic(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.pending');q.write_text(json.dumps(obj,indent=2)+'\n');q.replace(p)
def make(name,fresh=False):
 cfg=PROFILES[name].copy();dense=cfg.pop('dense',False);reset=cfg.pop('reset_keys',False);separate=cfg.pop('separate_ports',False)
 src=ROOT/'sources'/SIZE/'last.npz'
 if not fresh:m=Model.load(src,probe=cfg)
 else:
  with np.load(src) as a:
   p,_=core.initialize(seed=1)
   m=Model(config=json.loads(str(a['config_json'])),params=p,fixed={k[6:]:a[k] for k in a.files if k.startswith('fixed_')},lace=json.loads(str(a['lace_json'])),state_config=json.loads(str(a['state_config_json'])),probe=cfg)
 if dense:
  m.lace['sdr_k']=64
  a=np.array([m.lace[k] for k in __import__('model').legacy.base.FIELDS],np.float32)
  m.lib.configure_laces(m.handle,a.ctypes.data_as(ct.POINTER(ct.c_float)))
 if reset:
  rng=np.random.default_rng(1973)
  keys=rng.normal(size=m.params['keys'][2:].shape).astype(np.float32);keys/=np.linalg.norm(keys,axis=-1,keepdims=True)
  m.params['keys'][2:]=keys
 if separate:
  # Preserve each source in disjoint coordinates at L3/F4/G5/Y. The total
  # message width remains 64; source projections become narrower and larger
  # fan-ins no longer add their main publications into the same coordinates.
  # Existing signed-permutation credit uses these exact masks/scales too.
  for r in [6,11,14,17]:
   lo,hi=map(int,m.fixed['starts'][r:r+2]);edges=np.linspace(0,64,hi-lo+1,dtype=int)
   for slot,k in enumerate(range(lo,hi)):
    a,b=edges[slot:slot+2];mask=np.zeros(64,np.float32);mask[a:b]=np.sqrt(64/(b-a))
    m.fixed['signs'][k]*=mask
 return m
def run(name,total,fresh=False):
 data=np.frombuffer((ROOT/'data/fresh.bin').read_bytes(),np.uint8) if fresh else TRAIN
 assert total<len(data)
 out=RESULTS/('fresh' if fresh else 'continued')/SIZE/name;out.mkdir(parents=True,exist_ok=True)
 meta=out/'training.json'
 if meta.exists():
  result=json.loads(meta.read_text());pos=result['targets'];m=Model.load(out/'last.npz')
  if fresh and pos==1000000 and total>pos:
   shutil.copy2(out/'last.npz',out/'at_1m.npz');shutil.copy2(meta,out/'at_1m_training.json')
   result['initial_prefix_sha256']=result['training_sha256'];result['training_sha256']=hashlib.sha256(data).hexdigest()
 else:
  m=make(name,fresh);pos=0
  result={'profile':name,'size':SIZE,'fresh':fresh,'parameters':m.parameter_count,'targets':0,'seconds':0.,'validation':[],'history':[],
    'initial_validation':evaluate(m,VAL[:32769]),'training_sha256':hashlib.sha256(data).hexdigest(),'probe':m.probe,'lace':m.lace}
 while pos<total:
  n=min(5000,total-pos);lr=float(.008+.036*(1+np.cos(np.pi*min(1,(pos+n/2)/1e6)))) if fresh else .008
  start=time.perf_counter();loss=m.run(data[pos:pos+n],data[pos+1:pos+n+1],mode='periodic',lr=lr,lp=lr/3,route=.002);dt=time.perf_counter()-start;pos+=n
  result['seconds']+=dt;result['history'].append({'targets':pos,'nll':loss,'lr':lr})
  if pos%(50000 if total<=200000 else 200000)==0 or pos==total:
   score=evaluate(m,VAL[:32769] if total<=200000 else VAL)
   result['validation'].append({'training_targets':pos,**score});result['targets']=pos
   print(json.dumps({'size':SIZE,'run':name,'fresh':fresh,'targets':pos,'val':score['nll'],'effective_Y':score['effective_experts']['Y']}),flush=True)
  if fresh and pos==1000000 and total>pos:
   m.save(out/'at_1m.npz');atomic(out/'at_1m_training.json',result)
  if pos==total:
   assert np.isfinite(m.flat).all() and np.isfinite(m.extra_flat).all()
   m.save(out/'last.npz');result['lace_metrics']=m.lace_metrics();result['learned_metrics']=m.learned_metrics()
   atomic(meta,result)
 m.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('profile',choices=PROFILES);p.add_argument('--targets',type=int,default=200000);p.add_argument('--fresh',action='store_true');a=p.parse_args()
 os.sched_setaffinity(0,{int(os.environ.get('PROBE_CPU','0'))});run(a.profile,a.targets,a.fresh)
