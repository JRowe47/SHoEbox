"""Hippocampus sweep configuration; all access remains through ordinary EC."""
import ctypes as ct,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import importlib.util
_core_spec=importlib.util.spec_from_file_location('online_cortical_core',Path(__file__).resolve().parent/'experiments/cortical_compute/core.py')
core=importlib.util.module_from_spec(_core_spec);_core_spec.loader.exec_module(core)

ROOT=Path(__file__).parent
DEFAULT=dict(key=2,gain=1,capacity=256,temporal=1,value=1,center_value=1,credit=1,credit_scale=4,threshold=.6,stride=1,port=1,byte_cue=1,buckets=256,ways=4,write_novelty=.04,duplicate=.98,salience=1,read_stride=1,key_mix=0,wake_threshold=0,read_novelty=.02,no_returns=0,outcome_threshold=.4,competition=.02,ec_c=1,ec_x=1,mode='periodic',ec_y_gain=1)
FIELDS=list(DEFAULT)[:24]

def build():
    src=ROOT/'persistent_dual_native.cpp';lib=ROOT/f'_persistent_dual_native_e{core.E}.so'
    if not lib.exists() or src.stat().st_mtime>lib.stat().st_mtime:
        subprocess.run(['g++','-O3','-march=native','-std=c++17',f'-DSHOEBOX_EXPERTS={core.E}','-shared','-fPIC',str(src),'-o',str(lib)],check=True)
    return lib

class Variant(core.Native):
    def __init__(self,config=None,seed=1,params=None,fixed=None):
        self.config={**DEFAULT,**(config or {})}
        b=int(self.config['buckets']);w=int(self.config['ways'])
        if b<4 or b&(b-1) or w<1 or b*w>8192:raise ValueError('buckets/ways')
        if min(int(self.config[k]) for k in ('ec_c','ec_x','stride','read_stride'))<1:raise ValueError('cadence')
        if self.config['wake_threshold']>0 and self.config['credit']==2:raise ValueError('Event EC uses sampled local credit.')
        if self.config['mode'] not in ('dfa','periodic'):raise ValueError('This experiment supports DFA and periodic local teaching.')
        if params is None:params,fixed=core.initialize(seed)
        fixed={k:v.copy() for k,v in fixed.items()};fixed['cadence'][15:17]=[self.config['ec_c'],self.config['ec_x']]
        for k in range(int(fixed['starts'][17]),int(fixed['starts'][18])):
            if int(fixed['slots'][k,0])==15:fixed['slots'][k,3]=np.sqrt(.5)*self.config['ec_y_gain']
        original=core.build
        try:
            core.build=build
            super().__init__(params=params,fixed=fixed,head='tree')
        finally:core.build=original
        self.lib.configure.argtypes=[ct.c_void_p,ct.POINTER(ct.c_float)]
        arr=np.array([self.config[k] for k in FIELDS],np.float32)
        self.lib.configure(self.handle,arr.ctypes.data_as(ct.POINTER(ct.c_float)))
        self.lib.set_reference.argtypes=[ct.c_void_p,ct.POINTER(ct.c_float)]
        if self.config['key']==3:
            if 'address_center' not in self.fixed:raise ValueError('key=3 requires calibrated fixed address_center.')
            ref=np.ascontiguousarray(self.fixed['address_center'],dtype=np.float32)
            if ref.shape!=(2,64):raise ValueError('address_center shape')
            self.lib.set_reference(self.handle,ref.ctypes.data_as(ct.POINTER(ct.c_float)))
        self.lib.query_vectors.argtypes=[ct.c_void_p,ct.POINTER(ct.c_float)]
        self.lib.persistent_metrics.argtypes=[ct.c_void_p,ct.POINTER(ct.c_double)]
        self.lib.block_before.argtypes=[ct.c_void_p,ct.c_int64]
        self.lib.freeze_writes.argtypes=[ct.c_void_p,ct.c_int]
        self.lib.clear_memory.argtypes=[ct.c_void_p]
        self.lib.reset_context.argtypes=[ct.c_void_p]
    def persistent_metrics(self):
        a=np.zeros(20,np.float64);self.lib.persistent_metrics(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_double)))
        names=('queries','candidate_slots','dot_products','write_considered','merged','inserted','replaced','occupied_insertions','admission_rejected','novelty_rejected','hits','age_gt256','age_gt4096','age_gt16384','ec_skipped','ec_woken','age_sum','max_age','similarity_sum','occupied')
        return dict(zip(names,a.tolist()))
    def block_before(self,cutoff=0):self.lib.block_before(self.handle,int(cutoff))
    def freeze_writes(self,value=True):self.lib.freeze_writes(self.handle,int(value))
    def clear_memory(self):self.lib.clear_memory(self.handle)
    def reset_context(self):self.lib.reset_context(self.handle)
    def queries(self):
        a=np.zeros((2,64),np.float32);self.lib.query_vectors(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_float)));return a
    def run(self,x,y=None,mode='eval',lr=.12,lp=.04,route=.002,logits=False):
        if mode not in ('eval','dfa','periodic'):raise ValueError('Use eval, dfa or periodic with this experimental kernel.')
        return super().run(x,y,mode,lr,lp,route,logits)
    def save(self,path):
        state=np.zeros(self.lib.state_size(),np.uint8);self.lib.export_state(self.handle,state.ctypes.data)
        np.savez_compressed(path,flat=self.flat,state=state,version=np.array([9],np.int32),experts=np.array([core.E],np.int32),config_json=np.array(json.dumps(self.config)),
            **{'fixed_'+k:v for k,v in self.fixed.items()})
    @classmethod
    def load(cls,path,reset=False):
        with np.load(path) as a:
            assert int(a['version'][0]) == 9
            if int(a['experts'][0]) != core.E: raise ValueError('Set SHOEBOX_EXPERTS before importing the model.')
            m=cls(json.loads(str(a['config_json'])),params=core.unflatten(a['flat']),fixed={k[6:]:a[k] for k in a.files if k.startswith('fixed_')})
            if not reset:
                if int(a['version'][0])!=9:raise ValueError('Dual bucket layout requires reset=True for earlier weights.')
                state=np.ascontiguousarray(a['state'],dtype=np.uint8)
                if len(state)!=m.lib.state_size():raise ValueError('Use reset=True for earlier experiment weights; state layout changed.')
                m.lib.import_state(m.handle,state.ctypes.data)
        return m

def evaluate(model,data,ablation=0,blocks=False):
    m=Variant(model.config,params=model.params,fixed=model.fixed);m.ablate(ablation)
    losses=[];sizes=[]
    for i in range(0,len(data)-1,1024):
        n=min(1024,len(data)-1-i);losses.append(m.run(data[i:i+n],data[i+1:i+n+1]));sizes.append(n)
    out=dict(nll=float(np.average(losses,weights=sizes)),accuracy=m.stats()['correct']/(len(data)-1))
    if blocks:out['block_nll']=losses
    m.close();return out
