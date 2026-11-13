"""Version-12 learned byte state, bounded temporal laces and bottom routing."""
from pathlib import Path
import ctypes as ct,fcntl,hashlib,json,os,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
# Import the parent explicitly: this file is also called model.py.
import importlib.util
spec=importlib.util.spec_from_file_location('publication_lace_model',ROOT.parent/'model.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
core=base.core
DEFAULT=dict(enabled=1,state_lr=1.,read_limit=.25,write_a=.125,write_b=.03125,
             temporal_gain=1.,credit_stride=4,temporal_features=16,
             task_gain=0.,margin_strength=.05,temperature=.5,ema=0,
             ema_mass_start=128.,ema_mass_end=512.,ema_anneal_events=100000,
             revive_every=4096,revive_after=8192,revive_threshold=.1,revive_idle=4096,
             route_features=64,initial_read=0.,task_offset_limit=.5,reserved=0.,revive_utility=.01)
FIELDS=list(DEFAULT)
EXTRA_SHAPES={'state_gates':(2,3,64),'prototypes':(2,core.E,64),'task_offsets':(2,core.E,64)}

def build():
    source=ROOT/'native.cpp';digest=hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    library=ROOT/f'_state_e{core.E}_{digest}.so'
    with open(ROOT/f'.build_e{core.E}.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not library.exists():
            temporary=library.with_suffix(f'.{os.getpid()}.tmp')
            subprocess.run(['g++','-O3','-march=native','-std=c++17',f'-DSHOEBOX_EXPERTS={core.E}','-shared','-fPIC',str(source),'-o',str(temporary)],check=True)
            os.replace(temporary,library)
    return library

class Model(base.Model):
    def __init__(self,config=None,seed=1,params=None,fixed=None,lace=None,state_config=None,extra=None):
        self.state_config={**DEFAULT,**(state_config or {})}
        c=self.state_config
        if c['credit_stride']<1 or c['temporal_features']<1 or c['temporal_features']>64 or not 1<=c['route_features']<=64:raise ValueError('Invalid sparse credit/routing budget')
        if c['read_limit']<=0 or abs(c['initial_read'])>=c['read_limit']:raise ValueError('Invalid state read range')
        if not 0<c['write_a']<1 or not 0<c['write_b']<1:raise ValueError('Write probabilities must lie inside (0,1)')
        if min(c['ema_mass_start'],c['ema_mass_end'],c['ema_anneal_events'],c['temperature'],c['task_offset_limit'])<=0:raise ValueError('Positive rates/count scales required')
        previous=base.build
        try:base.build=build;super().__init__(config,seed,params,fixed,lace)
        finally:base.build=previous
        total=sum(int(np.prod(s)) for s in EXTRA_SHAPES.values())
        self.extra_flat=np.zeros(total,np.float32) if extra is None else np.ascontiguousarray(extra,dtype=np.float32).copy()
        if self.extra_flat.shape!=(total,):raise ValueError('Learned-state parameter shape mismatch')
        self.extra={};pos=0
        for name,shape in EXTRA_SHAPES.items():
            n=int(np.prod(shape));self.extra[name]=self.extra_flat[pos:pos+n].reshape(shape);pos+=n
        if extra is None:
            for r,key in enumerate(['write_a','write_b']):
                self.extra['state_gates'][r,0]=np.arctanh(c['initial_read']/c['read_limit'])
                self.extra['state_gates'][r,2]=np.log(c[key]/(1-c[key]))
            self.extra['prototypes'][:]=self.params['keys'][:2]
        fp=ct.POINTER(ct.c_float)
        self.lib.configure_state.argtypes=[ct.c_void_p,fp,fp]
        cfg=np.array([c[k] for k in FIELDS],np.float32)
        self.lib.configure_state(self.handle,self.extra_flat.ctypes.data_as(fp),cfg.ctypes.data_as(fp))
        self.lib.learned_state_size.restype=ct.c_size_t
        for name in ['export_learned','import_learned']:getattr(self.lib,name).argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.learned_metrics.argtypes=[ct.c_void_p,ct.POINTER(ct.c_double)]
        self.lib.learned_snapshot.argtypes=[ct.c_void_p,fp]
        self.lib.force_revival.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int];self.lib.force_revival.restype=ct.c_int
    def learned_state(self):
        out=np.zeros(self.lib.learned_state_size(),np.uint8);self.lib.export_learned(self.handle,out.ctypes.data);return out
    def learned_metrics(self):
        out=np.zeros((2,12),np.float64);self.lib.learned_metrics(self.handle,out.ctypes.data_as(ct.POINTER(ct.c_double)))
        names=['state_updates','temporal_updates','missing_or_stale','task_router_updates','ema_updates','revivals','read_gradient_abs','read_contribution_abs','write_sum','publications','temporal_age_sum','reserved']
        return {n:dict(zip(names,row.tolist())) for n,row in zip(['A1','B1'],out)}
    def learned_snapshot(self):
        out=np.zeros((2,2,64),np.float32);self.lib.learned_snapshot(self.handle,out.ctypes.data_as(ct.POINTER(ct.c_float)));return out
    @property
    def parameter_count(self):return int(self.flat.size+self.extra_flat.size)
    def fingerprint(self):
        return hashlib.sha256(self.flat.tobytes()+self.extra_flat.tobytes()+base.state(self).tobytes()+self.lace_state().tobytes()+self.learned_state().tobytes()).hexdigest()
    def save(self,path):
        base.atomic_checkpoint(path,flat=self.flat,extra=self.extra_flat,state=base.state(self),lace_state=self.lace_state(),learned_state=self.learned_state(),
             version=np.array([12],np.int32),experts=np.array([core.E],np.int32),config_json=np.array(json.dumps(self.config)),
             lace_json=np.array(json.dumps(self.lace)),state_config_json=np.array(json.dumps(self.state_config)),
             kernel_sha256=np.array(hashlib.sha256((ROOT/'native.cpp').read_bytes()).hexdigest()),**{'fixed_'+k:v for k,v in self.fixed.items()})
    @classmethod
    def load(cls,path,state_config=None,reset=False):
        with np.load(path) as a:
            version=int(a['version'][0])
            if version not in (11,12) or int(a['experts'][0])!=core.E:raise ValueError('Set matching SHOEBOX_EXPERTS before import; checkpoint version must be 11 or 12')
            policy=json.loads(str(a['state_config_json'])) if version==12 else DEFAULT.copy()
            if state_config is not None:
                changed=any(policy.get(k)!=v for k,v in state_config.items())
                if version==12 and changed and not reset:raise ValueError('Changing state policy requires an explicit context reset')
                policy.update(state_config)
            if version==11 and not reset:raise ValueError('Legacy migration requires reset=True for the new temporal history')
            m=cls(config=json.loads(str(a['config_json'])),params=core.unflatten(a['flat']),fixed={k[6:]:a[k] for k in a.files if k.startswith('fixed_')},lace=json.loads(str(a['lace_json'])),state_config=policy,extra=a['extra'] if version==12 else None)
            if reset:m.reset()
            else:
                for key,method,size in [('state','import_state',m.lib.state_size()),('lace_state','import_laces',m.lib.lace_state_size()),('learned_state','import_learned',m.lib.learned_state_size())]:
                    st=np.ascontiguousarray(a[key],dtype=np.uint8)
                    if len(st)!=size:m.close();raise ValueError(f'Checkpoint {key} ABI mismatch')
                    getattr(m.lib,method)(m.handle,st.ctypes.data)
        return m

def evaluate(m,data,blocks=False):
    p=Model(config=m.config,params=m.params,fixed=m.fixed,lace=m.lace,state_config=m.state_config,extra=m.extra_flat);p.reset()
    scores=[];sizes=[]
    for i in range(0,len(data)-1,1024):
        n=min(1024,len(data)-1-i);scores.append(p.run(data[i:i+n],data[i+1:i+n+1]));sizes.append(n)
    result=dict(nll=float(np.average(scores,weights=sizes)),accuracy=p.stats()['correct']/(len(data)-1))
    if blocks:result['block_nll']=scores
    p.close();return result
