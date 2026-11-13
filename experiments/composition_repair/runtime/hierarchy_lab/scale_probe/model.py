from pathlib import Path
import ctypes as ct,fcntl,hashlib,importlib.util,json,os,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('probe_legacy',ROOT.parent/'learned_state/model.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
core=legacy.core
DEFAULT=dict(context_gain=.2,route_mode=0,center_rate=0,port_mode=0,port_gain=0,balance=0,reserved1=0,reserved2=0)
FIELDS=list(DEFAULT)
def build():
    source=ROOT/'native.cpp';digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    lib=ROOT/f'_probe_e{core.E}_{digest}.so'
    with open(ROOT/f'.build_e{core.E}.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not lib.exists():
            tmp=lib.with_suffix(f'.{os.getpid()}.tmp')
            subprocess.run(['g++','-O3','-march=native','-std=c++17',f'-DSHOEBOX_EXPERTS={core.E}','-shared','-fPIC',str(source),'-o',str(tmp)],check=True)
            os.replace(tmp,lib)
    return lib
class Model(legacy.Model):
    def __init__(self,*args,probe=None,**kwargs):
        self.probe={**DEFAULT,**(probe or {})}
        previous=legacy.build
        try:legacy.build=build;super().__init__(*args,**kwargs)
        finally:legacy.build=previous
        fp=ct.POINTER(ct.c_float);dp=ct.POINTER(ct.c_double)
        self.lib.probe_configure.argtypes=[ct.c_void_p,fp]
        self.lib.diagnostic_enable.argtypes=[ct.c_void_p,ct.c_int]
        self.lib.diagnostic_export.argtypes=[ct.c_void_p,dp]
        cfg=np.array([self.probe[k] for k in FIELDS],np.float32)
        self.lib.probe_configure(self.handle,cfg.ctypes.data_as(fp))
        self.lib.probe_state_size.restype=ct.c_size_t
        for name in ['probe_export','probe_import']:getattr(self.lib,name).argtypes=[ct.c_void_p,ct.c_void_p]
        for name in ['probe_get_means','probe_set_means']:getattr(self.lib,name).argtypes=[ct.c_void_p,fp]
        self.lib.probe_initial_mean.argtypes=[ct.c_void_p]
        self._probe_ready=True
    @classmethod
    def load(cls,path,probe=None,reset=False):
        with np.load(path,allow_pickle=False) as a:
            assert int(a['experts'][0])==core.E
            policy=json.loads(str(a['probe_json'])) if 'probe_json' in a.files else DEFAULT.copy()
            if probe is not None:policy.update(probe)
            m=cls(config=json.loads(str(a['config_json'])),params=core.unflatten(a['flat']),fixed={k[6:]:a[k] for k in a.files if k.startswith('fixed_')},lace=json.loads(str(a['lace_json'])),state_config=json.loads(str(a['state_config_json'])),extra=a['extra'],probe=policy)
            if not reset:
                for key,method,size in [('state','import_state',m.lib.state_size()),('lace_state','import_laces',m.lib.lace_state_size()),('learned_state','import_learned',m.lib.learned_state_size())]:
                    st=np.ascontiguousarray(a[key],dtype=np.uint8);assert len(st)==size
                    getattr(m.lib,method)(m.handle,st.ctypes.data)
                if 'probe_state' in a.files:
                    st=np.ascontiguousarray(a['probe_state']);assert st.size==m.lib.probe_state_size();m.lib.probe_import(m.handle,st.ctypes.data)
                else:m.lib.probe_initial_mean(m.handle)
            if reset and 'probe_state' in a.files:
                # Means are learned preprocessing statistics. Preserve them
                # when clearing stream state, like frozen_copy does.
                saved=np.ascontiguousarray(a['probe_state'],dtype=np.uint8)
                assert saved.size==m.lib.probe_state_size()
                clean=m.probe_state();start=8*ct.sizeof(ct.c_float);stop=start+18*64*ct.sizeof(ct.c_float)
                clean[start:stop]=saved[start:stop];m.lib.probe_import(m.handle,clean.ctypes.data)
            cfg=np.array([m.probe[k] for k in FIELDS],np.float32)
            m.lib.probe_configure(m.handle,cfg.ctypes.data_as(ct.POINTER(ct.c_float)))
            return m
    def reset(self):
        super().reset()
        if getattr(self,'_probe_ready',False):
            saved=self.probe_state();clean=np.zeros_like(saved)
            stop=(8+18*64)*ct.sizeof(ct.c_float)
            clean[:stop]=saved[:stop]
            self.lib.probe_import(self.handle,clean.ctypes.data)
            self.lib.diagnostic_enable(self.handle,0)
    def probe_state(self):
        a=np.zeros(self.lib.probe_state_size(),np.uint8);self.lib.probe_export(self.handle,a.ctypes.data);return a
    def means(self):
        a=np.zeros((18,64),np.float32);self.lib.probe_get_means(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_float)));return a
    def frozen_copy(self):
        m=Model(config=self.config,params=self.params,fixed=self.fixed,lace=self.lace,state_config=self.state_config,extra=self.extra_flat,probe=self.probe);m.reset()
        a=self.means();m.lib.probe_set_means(m.handle,a.ctypes.data_as(ct.POINTER(ct.c_float)));return m
    def save(self,path):
        legacy.base.atomic_checkpoint(path,flat=self.flat,extra=self.extra_flat,state=legacy.base.state(self),lace_state=self.lace_state(),learned_state=self.learned_state(),probe_state=self.probe_state(),
            version=np.array([15],np.int32),experts=np.array([core.E],np.int32),config_json=np.array(json.dumps(self.config)),lace_json=np.array(json.dumps(self.lace)),
            state_config_json=np.array(json.dumps(self.state_config)),probe_json=np.array(json.dumps(self.probe)),
            kernel_sha256=np.array(hashlib.sha256((ROOT/'native.cpp').read_bytes()).hexdigest()),**{'fixed_'+k:v for k,v in self.fixed.items()})
    def diagnostics(self):
        a=np.zeros(18+18*4*64+18*4*64*64+18*8,np.float64)
        self.lib.diagnostic_export(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_double)))
        pos=18;n=a[:pos];s=a[pos:pos+18*4*64].reshape(18,4,64);pos+=s.size
        prod=a[pos:pos+18*4*64*64].reshape(18,4,64,64);pos+=prod.size;ex=a[pos:].reshape(18,8)
        result={}
        for r,name in enumerate(core.NAMES):
            entry={'samples':int(n[r]),'margin':float(ex[r,0]/max(1,n[r])),'saturated_h_fraction':float(ex[r,1]/max(1,n[r]*64)),
              'context_to_bottom_norm':float(np.sqrt(ex[r,3]/max(ex[r,2],1e-30))),
              'context_energy_in_router_mask':float(ex[r,4]/max(ex[r,3],1e-30))}
            for k,label in enumerate(['bottom','context','input','output']):
                mean=s[r,k]/max(1,n[r]);moment=prod[r,k]/max(1,n[r]);cov=moment-np.outer(mean,mean)
                e=np.maximum(0,np.linalg.eigvalsh(cov));pr=e/max(e.sum(),1e-30)
                entry[label]={'effective_rank':float(np.exp(-np.sum(pr*np.log(np.maximum(pr,1e-30))))) if e.sum()>1e-14 else 0,
                  'variance':float(e.sum()),'mean_energy_fraction':float(np.dot(mean,mean)/max(np.trace(moment),1e-30))}
            result[name]=entry
        return result
def evaluate(model,data,diag=False):
    m=model.frozen_copy();m.lib.diagnostic_enable(m.handle,int(diag));loss=[];lengths=[]
    for pos in range(0,len(data)-1,1024):
        n=min(1024,len(data)-1-pos);loss.append(m.run(data[pos:pos+n],data[pos+1:pos+n+1]));lengths.append(n)
    stats=m.stats();counts=np.array(stats['expert_counts']);prob=counts/np.maximum(1,counts.sum(1,keepdims=True))
    out=dict(nll=float(np.average(loss,weights=lengths)),blocks=loss,targets=sum(lengths),effective_experts=dict(zip(core.NAMES,np.exp(-(prob*np.log(np.maximum(prob,1e-30))).sum(1)).tolist())))
    if diag:out['diagnostics']=m.diagnostics()
    m.close();return out
