from pathlib import Path
import ctypes as ct, fcntl, hashlib, importlib.util, json, os, subprocess
import numpy as np
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('utility_parent_model',ROOT.parent/'scale_probe/model.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
core=parent.core
DEFAULT=dict(center_y=0.,split_norm=0.,residual=0.,upper_local_gain=1.,innovation=0.,mean_rate=.001,reserved1=0.,reserved2=0.)
FIELDS=list(DEFAULT)
def build():
    source=ROOT/'native.cpp';digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    lib=ROOT/f'_utility_e{core.E}_{digest}.so'
    with open(ROOT/f'.build_e{core.E}.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not lib.exists():
            tmp=lib.with_suffix(f'.{os.getpid()}.tmp')
            subprocess.run(['g++','-O3','-march=native','-std=c++17',f'-DSHOEBOX_EXPERTS={core.E}','-shared','-fPIC',str(source),'-o',str(tmp)],check=True)
            os.replace(tmp,lib)
    return lib
class Model(parent.Model):
    def __init__(self,*args,utility=None,**kwargs):
        self.utility={**DEFAULT,**(utility or {})}
        if not 0<self.utility['mean_rate']<=1 or self.utility['upper_local_gain']<0:raise ValueError('invalid utility policy')
        previous=parent.build
        try:parent.build=build;super().__init__(*args,**kwargs)
        finally:parent.build=previous
        fp=ct.POINTER(ct.c_float)
        self.lib.utility_configure.argtypes=[ct.c_void_p,fp]
        self.lib.utility_state_size.restype=ct.c_size_t
        for n in ['utility_export','utility_import']:getattr(self.lib,n).argtypes=[ct.c_void_p,ct.c_void_p]
        for n in ['utility_mean','utility_set_mean']:getattr(self.lib,n).argtypes=[ct.c_void_p,fp]
        for n in ['utility_ablate','utility_observe']:getattr(self.lib,n).argtypes=[ct.c_void_p,ct.c_int]
        self.lib.utility_metrics.argtypes=[ct.c_void_p,ct.POINTER(ct.c_double)]
        cfg=np.array([self.utility[k] for k in FIELDS],np.float32)
        self.lib.utility_configure(self.handle,cfg.ctypes.data_as(fp))
        self._utility_ready=True
    def utility_state(self):
        a=np.zeros(self.lib.utility_state_size(),np.uint8);self.lib.utility_export(self.handle,a.ctypes.data);return a
    def ymean(self):
        a=np.zeros(64,np.float32);self.lib.utility_mean(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_float)));return a
    def set_ymean(self,a):
        a=np.ascontiguousarray(a,dtype=np.float32);assert a.shape==(64,)
        self.lib.utility_set_mean(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_float)))
    def reset(self):
        super().reset()
        if getattr(self,'_utility_ready',False):
            saved=self.utility_state();clean=np.zeros_like(saved);stop=(8+64)*4;clean[:stop]=saved[:stop]
            self.lib.utility_import(self.handle,clean.ctypes.data);self.lib.utility_observe(self.handle,0);self.lib.utility_ablate(self.handle,0)
    def frozen_copy(self):
        m=Model(config=self.config,params=self.params,fixed=self.fixed,lace=self.lace,state_config=self.state_config,extra=self.extra_flat,probe=self.probe,utility=self.utility)
        a=self.means();m.lib.probe_set_means(m.handle,a.ctypes.data_as(ct.POINTER(ct.c_float)));m.set_ymean(self.ymean());return m
    def fingerprint(self):
        return hashlib.sha256(super().fingerprint().encode()+self.probe_state().tobytes()+self.utility_state().tobytes()).hexdigest()
    def save(self,path):
        parent.legacy.base.atomic_checkpoint(path,flat=self.flat,extra=self.extra_flat,state=parent.legacy.base.state(self),lace_state=self.lace_state(),learned_state=self.learned_state(),probe_state=self.probe_state(),utility_state=self.utility_state(),
            version=np.array([16],np.int32),experts=np.array([core.E],np.int32),config_json=np.array(json.dumps(self.config)),lace_json=np.array(json.dumps(self.lace)),state_config_json=np.array(json.dumps(self.state_config)),probe_json=np.array(json.dumps(self.probe)),utility_json=np.array(json.dumps(self.utility)),
            kernel_sha256=np.array(hashlib.sha256((ROOT/'native.cpp').read_bytes()).hexdigest()),**{'fixed_'+k:v for k,v in self.fixed.items()})
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as a:
            assert int(a['version'][0])==16 and int(a['experts'][0])==core.E
            m=cls(config=json.loads(str(a['config_json'])),params=core.unflatten(a['flat']),fixed={k[6:]:a[k] for k in a.files if k.startswith('fixed_')},lace=json.loads(str(a['lace_json'])),state_config=json.loads(str(a['state_config_json'])),extra=a['extra'],probe=json.loads(str(a['probe_json'])),utility=json.loads(str(a['utility_json'])))
            for key,method,size in [('state','import_state',m.lib.state_size()),('lace_state','import_laces',m.lib.lace_state_size()),('learned_state','import_learned',m.lib.learned_state_size()),('probe_state','probe_import',m.lib.probe_state_size()),('utility_state','utility_import',m.lib.utility_state_size())]:
                arr=np.ascontiguousarray(a[key],dtype=np.uint8);assert arr.size==size,(key,arr.size,size)
                getattr(m.lib,method)(m.handle,arr.ctypes.data)
        return m
    def utility_metrics(self):
        a=np.zeros((18,8),np.float64);self.lib.utility_metrics(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_double)))
        return {n:dict(total_norm=v[0],total_calls=v[1],local_norm=v[2],local_calls=v[3]) for n,v in zip(core.NAMES,a)}

def evaluate(model,data,mean_y=False,documents=None):
    m=model.frozen_copy();m.lib.utility_ablate(m.handle,int(mean_y));rows=[]
    if documents is None:
        spans=[(i,min(i+2048,len(data)-1),str(i)) for i in range(0,len(data)-1,2048)]
    else:
        spans=[(max(0,d['start']-1),min(len(data)-1,d['start']+d['bytes_used']-1),d['id']) for d in documents]
    for lo,hi,ident in spans:
        if hi>lo:rows.append(dict(id=ident,targets=hi-lo,nll=m.run(data[lo:hi],data[lo+1:hi+1])))
    counts=np.array(m.stats()['expert_counts']);prob=counts/np.maximum(1,counts.sum(1,keepdims=True));eff=np.exp(-(prob*np.log(np.maximum(prob,1e-30))).sum(1))
    out=dict(nll=float(np.average([v['nll'] for v in rows],weights=[v['targets'] for v in rows])),targets=sum(v['targets'] for v in rows),effective_experts=dict(zip(core.NAMES,eff.tolist())),documents=rows)
    m.close();return out
