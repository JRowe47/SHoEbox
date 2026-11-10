"""Sparse publication laces, task credit and feature-SDR routing."""
import ctypes as ct,fcntl,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent;BASE=ROOT.parent
sys.path.insert(0,str(BASE))
import persistent_dual_lab as original
core=original.core
DEFAULT=dict(depth=0,credit_gain=1,sdr_k=64,router_task_gain=0,temperature=.25,reuse_exponent=.5,dfa_gain=1,credit_stride=1,sparse_from=0,credit_features=64,deep_stride=1,flush_on_execution=0)
FIELDS=list(DEFAULT)

def atomic_checkpoint(path,**arrays):
    path=Path(path);temporary=path.with_name(path.name+f'.pending.{os.getpid()}')
    try:
        with temporary.open('wb') as output:
            np.savez_compressed(output,**arrays)
            output.flush();os.fsync(output.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()

def build():
    src=ROOT/'lace_native.cpp'
    digest=hashlib.sha256(src.read_bytes()).hexdigest()[:12]
    lib=ROOT/f'_lace_e{core.E}_{digest}.so'
    with open(ROOT/f'.build_e{core.E}.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not lib.exists():
            tmp=lib.with_suffix(f'.{os.getpid()}.tmp')
            subprocess.run(['g++','-O3','-march=native','-std=c++17',f'-DSHOEBOX_EXPERTS={core.E}','-shared','-fPIC',str(src),'-o',str(tmp)],check=True)
            os.replace(tmp,lib)
    return lib

class Model(original.Variant):
    def __init__(self,config=None,seed=1,params=None,fixed=None,lace=None):
        self.lace={**DEFAULT,**(lace or {})}
        if not 0<=self.lace['depth']<=3:raise ValueError('depth must be 0..3')
        if not 1<=self.lace['sdr_k']<=64:raise ValueError('sdr_k must be 1..64')
        if self.lace['temperature']<=0:raise ValueError('positive temperature required')
        if self.lace['credit_stride']<1:raise ValueError('positive credit stride required')
        if self.lace['deep_stride']<1:raise ValueError('positive third-hop stride required')
        if self.lace['credit_features'] not in (64,self.lace['sdr_k']):raise ValueError('Sparse credit must use the routing SDR width')
        previous=original.build
        try:
            original.build=build
            super().__init__(config,seed,params,fixed)
        finally:original.build=previous
        ptr=ct.POINTER(ct.c_float)
        self.lib.configure_laces.argtypes=[ct.c_void_p,ptr]
        cfg=np.array([self.lace[k] for k in FIELDS],np.float32)
        self.lib.configure_laces(self.handle,cfg.ctypes.data_as(ptr))
        self.lib.lace_state_size.restype=ct.c_size_t
        for name in ('export_laces','import_laces'):
            getattr(self.lib,name).argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.lace_metrics.argtypes=[ct.c_void_p,ct.POINTER(ct.c_double)]
        self.lib.lace_codes.argtypes=[ct.c_void_p,ct.POINTER(ct.c_uint64)]
    def lace_state(self):
        a=np.zeros(self.lib.lace_state_size(),np.uint8);self.lib.export_laces(self.handle,a.ctypes.data);return a
    def lace_metrics(self):
        a=np.zeros((18,12),np.float64);self.lib.lace_metrics(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_double)))
        names=('publications','credit_edges','credit_updates','missing_parent','router_updates','gradient_norm_sum','feature_overlap_sum','reserved','credit_depth1','credit_depth2','credit_depth3','overwritten_credit')
        return {region:dict(zip(names,row.tolist())) for region,row in zip(core.NAMES,a)}
    def codes(self):
        a=np.zeros((18,3),np.uint64);self.lib.lace_codes(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_uint64)));return a
    def run(self,x,y=None,mode='eval',lr=.012,lp=.004,route=.002,logits=False):
        # Safe continuation defaults for the selected cosine-trained models.
        # Initial training and comparison runners pass their schedule explicitly.
        return super().run(x,y,mode,lr,lp,route,logits)
    def save(self,path):
        atomic_checkpoint(path,flat=self.flat,state=state(self),lace_state=self.lace_state(),
            version=np.array([11],np.int32),experts=np.array([core.E],np.int32),
            config_json=np.array(json.dumps(self.config)),lace_json=np.array(json.dumps(self.lace)),
            **{'fixed_'+k:v for k,v in self.fixed.items()})
    @classmethod
    def load(cls,path,lace=None,reset=False):
        with np.load(path) as a:
            version=int(a['version'][0]);assert version in (9,11)
            if int(a['experts'][0])!=core.E:raise ValueError('Set SHOEBOX_EXPERTS before import')
            lc=json.loads(str(a['lace_json'])) if version==11 else {}
            if lace is not None:lc.update(lace)
            m=cls(json.loads(str(a['config_json'])),params=core.unflatten(a['flat']),fixed={k[6:]:a[k] for k in a.files if k.startswith('fixed_')},lace=lc)
            if not reset:
                st=np.ascontiguousarray(a['state'],dtype=np.uint8);assert len(st)==m.lib.state_size()
                m.lib.import_state(m.handle,st.ctypes.data)
                if version==11:
                    if lace is not None and lc!=json.loads(str(a['lace_json'])):raise ValueError('Changing lace policy requires reset=True')
                    st=np.ascontiguousarray(a['lace_state'],dtype=np.uint8);assert len(st)==m.lib.lace_state_size();m.lib.import_laces(m.handle,st.ctypes.data)
        return m

def state(m):
    a=np.zeros(m.lib.state_size(),np.uint8);m.lib.export_state(m.handle,a.ctypes.data);return a

def evaluate(m,data,ablation=0,blocks=False):
    probe=Model(m.config,params=m.params,fixed=m.fixed,lace=m.lace);probe.ablate(ablation)
    losses=[];sizes=[]
    for i in range(0,len(data)-1,1024):
        n=min(1024,len(data)-1-i);losses.append(probe.run(data[i:i+n],data[i+1:i+n+1]));sizes.append(n)
    result=dict(nll=float(np.average(losses,weights=sizes)),accuracy=probe.stats()['correct']/(len(data)-1))
    if blocks:result['block_nll']=losses
    probe.close();return result
