"""One wide sparse operation per regional event, without a global training tape.

The transport recipes are copied from shoebox.config.cortical_map at 3bd9d96.
Widths below describe the original relative port capacities; transport is now
continuous 64D and uses distinct fixed signed permutations for every slot.
"""
from pathlib import Path
import ctypes as ct
import hashlib
import json
import subprocess
import numpy as np

import os
D, E, N, V, CAP = 64, int(os.environ.get("SHOEBOX_EXPERTS", "12")), 18, 256, 64
if not 2 <= E <= 4096: raise ValueError("Compiled expert capacity must be 2..4096.")
NAMES = ('A1','B1','L2','S2','R2','F2','L3','S3','R3','F3','E4','F4','V4','M4','G5','EC-C','EC-X','Y')
# source, original port width, occurrence count, delayed context
RECIPES = (
 (), (),
 (('A1',4,2,0),('L3',1,1,1)),
 (('A1',4,1,0),('B1',4,1,0),('S3',1,1,1)),
 (('A1',4,2,0),('B1',4,1,0),('R3',1,1,1)),
 (('B1',4,2,0),('A1',2,1,0),('F4',1,1,1)),
 (('L2',4,2,0),('S2',2,1,0),('F3',1,1,1),('EC-C',2,1,1)),
 (('S2',4,2,0),('L2',2,1,0),('E4',1,1,1),('EC-C',2,1,1)),
 (('R2',4,2,0),('S2',4,1,0),('F3',1,1,1)),
 (('F2',4,2,0),('L2',4,1,0),('R2',4,1,0),('G5',1,1,1),('M4',1,1,1),('EC-X',2,1,1)),
 (('S3',4,2,0),('R3',4,2,0),('L3',4,1,0),('G5',1,1,1),('EC-X',2,1,1)),
 (('F3',4,2,0),('R3',4,1,0),('L3',4,1,0),('G5',1,1,1),('M4',1,1,1)),
 (('S3',4,2,0),('F3',4,1,0),('B1',2,1,0),('G5',1,1,1)),
 (('R3',4,1,0),('F3',4,1,0),('B1',2,1,0),('F4',1,1,1)),
 (('E4',4,2,0),('F4',4,2,0),('V4',4,1,0)),
 (('L3',4,1,0),('S3',4,1,0),('S2',2,1,0),('EC-X',1,1,1)),
 (('E4',4,1,0),('F3',4,1,0),('G5',1,1,1),('M4',1,1,1),('EC-C',1,1,1)),
 (('L3',4,2,0),('F4',4,1,0),('S2',2,1,0),('EC-C',2,1,0)),
)
CADENCE = np.array([1,1,2,2,2,2,4,4,4,4,8,8,8,8,16,8,16,1],np.int32)
SHAPES = {'W':(N,E,D,D),'bias':(N,E,D),'keys':(N,E,D),
          'P':(N,D,D),'pbias':(N,D),'readout':(V,2*D),'rbias':(V,)}
MODES = {'eval':0,'local':1,'dfa':2,'hybrid':3,'readout':4,'spatial':5,'periodic':6}

def initialize(seed=1):
    rng=np.random.default_rng(seed)
    p={k:np.zeros(s,np.float32) for k,s in SHAPES.items()}
    # Generate one region at a time to avoid an extra ~800 MB float64 temporary
    # when initializing the uniform 100M (1312 experts/region) configuration.
    for r in range(N):p['W'][r]=rng.normal(0,.25/np.sqrt(D),p['W'][r].shape)
    p['keys'][:]=rng.normal(0,1,p['keys'].shape)
    p['keys']/=np.linalg.norm(p['keys'],axis=-1,keepdims=True)
    p['P'][:]=np.eye(D,dtype=np.float32)[None]*.25
    p['readout'][:]=rng.normal(0,.02,p['readout'].shape)
    embedding=rng.choice([-1.,1.],(V,D)).astype(np.float32)*.5
    feedback=rng.normal(0,1/np.sqrt(D),(V,D)).astype(np.float32)
    slots=[]; starts=[0]
    for r,recipe in enumerate(RECIPES):
        for src,width,count,delay in recipe:
            for age in range(count-1,-1,-1):
                slots.append((NAMES.index(src),age,delay,float(np.sqrt(width/4))))
        starts.append(len(slots))
    slots=np.array(slots,np.float32)
    perm=np.stack([rng.permutation(D) for _ in slots]).astype(np.int32)
    signs=rng.choice([-1.,1.],perm.shape).astype(np.float32)
    fperm=np.stack([rng.permutation(D) for _ in range(N)]).astype(np.int32)
    fsign=rng.choice([-1.,1.],(N,D)).astype(np.float32)
    fixed=dict(embedding=embedding,feedback=feedback,slots=slots,
               starts=np.array(starts,np.int32),perm=perm,signs=signs,
               fperm=fperm,fsign=fsign,cadence=CADENCE.copy())
    return p,fixed

def flat_parameters(p):
    return np.concatenate([p[k].ravel() for k in SHAPES]).astype(np.float32,copy=False)

def unflatten(a):
    out={}; pos=0
    for name,shape in SHAPES.items():
        size=int(np.prod(shape)); out[name]=a[pos:pos+size].reshape(shape);pos+=size
    return out

def build():
    root=Path(__file__).parent; src=root/'native.cpp'; lib=root/'_native.so'
    if not lib.exists() or lib.stat().st_mtime<src.stat().st_mtime:
        subprocess.run(['g++','-O3','-march=native','-std=c++17','-shared','-fPIC',str(src),'-o',str(lib)],check=True)
    return lib

class Native:
    def __init__(self,seed=1,all_ticks=False,params=None,fixed=None,head=None):
        if params is None:params,fixed=initialize(seed)
        if head not in (None,'softmax','tree'):raise ValueError('Unknown output head')
        if any(np.shape(params[k])!=shape for k,shape in SHAPES.items()):raise ValueError('Parameter shape mismatch')
        self.fixed={k:np.ascontiguousarray(v) for k,v in fixed.items()}
        self.fixed.setdefault('head',np.array([0],np.int32))
        if head is not None:self.fixed['head']=np.array([int(head=='tree')],np.int32)
        self.flat=flat_parameters(params);self.params=unflatten(self.flat)
        self.lib=ct.CDLL(str(build())); fptr=ct.POINTER(ct.c_float);iptr=ct.POINTER(ct.c_int)
        self.lib.create.restype=ct.c_void_p
        self.lib.create.argtypes=[fptr,fptr,fptr,fptr,iptr,iptr,fptr,iptr,fptr,iptr,ct.c_int]
        def fp(a):return a.ctypes.data_as(fptr)
        def ip(a):return a.ctypes.data_as(iptr)
        f=self.fixed
        if all_ticks:f['cadence']=np.ones(N,np.int32)
        self.handle=self.lib.create(fp(self.flat),fp(f['embedding']),fp(f['feedback']),fp(f['slots']),ip(f['starts']),ip(f['perm']),fp(f['signs']),ip(f['fperm']),fp(f['fsign']),ip(f['cadence']),int(f['head'][0]))
        self.lib.destroy.argtypes=[ct.c_void_p];self.lib.reset.argtypes=[ct.c_void_p]
        self.lib.run.argtypes=[ct.c_void_p,ct.POINTER(ct.c_ubyte),ct.POINTER(ct.c_ubyte),ct.c_int,ct.c_int,ct.c_float,ct.c_float,ct.c_float,ct.c_int,fptr]
        self.lib.run.restype=ct.c_double
        self.lib.info.argtypes=[ct.c_void_p,fptr]
        self.lib.state_size.restype=ct.c_size_t
        self.lib.export_state.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.import_state.argtypes=[ct.c_void_p,ct.c_void_p]
        self.lib.set_ablation.argtypes=[ct.c_void_p,ct.c_int]
        self.lib.publications.argtypes=[ct.c_void_p,fptr]
        self.lib.draw.argtypes=[ct.c_void_p,ct.c_float,fptr];self.lib.draw.restype=ct.c_int
    def close(self):
        if getattr(self,'handle',None):self.lib.destroy(self.handle);self.handle=None
    def __del__(self):self.close()
    def reset(self):self.lib.reset(self.handle)
    def ablate(self,mode=0):self.lib.set_ablation(self.handle,mode)
    def publications(self):
        a=np.zeros((N,D),np.float32);self.lib.publications(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_float)));return a
    def draw(self,rng,temperature=1.):
        if not self.fixed['head'][0]:raise ValueError('Sparse draw is for the tree head')
        if not np.isfinite(temperature) or temperature<=0:raise ValueError('Temperature must be positive')
        u=rng.random(8).astype(np.float32);return self.lib.draw(self.handle,temperature,u.ctypes.data_as(ct.POINTER(ct.c_float)))
    def run(self,x,y=None,mode='eval',lr=.12,lp=.04,route=.002,logits=False):
        x=np.ascontiguousarray(x,dtype=np.uint8);y=x.copy() if y is None else np.ascontiguousarray(y,dtype=np.uint8)
        if len(x)!=len(y):raise ValueError('Input and target lengths differ')
        out=np.empty((len(x),V) if logits else 1,np.float32)
        ptr=ct.POINTER(ct.c_ubyte)
        loss=self.lib.run(self.handle,x.ctypes.data_as(ptr),y.ctypes.data_as(ptr),len(x),MODES[mode],lr,lp,route,int(logits),out.ctypes.data_as(ct.POINTER(ct.c_float)))
        return (loss,out) if logits else loss
    def stats(self):
        a=np.zeros(N*(E+2)+5,np.float32);self.lib.info(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_float)))
        return dict(executions=a[:N].tolist(),local_mse=a[N:2*N].tolist(),expert_counts=a[2*N:2*N+N*E].reshape(N,E).tolist(),hc_reads=float(a[-5]),hc_writes=float(a[-4]),correct=float(a[-3]),observations=float(a[-2]),state_bytes=int(a[-1]))
    def save(self,path):
        state=np.zeros(self.lib.state_size(),np.uint8);self.lib.export_state(self.handle,state.ctypes.data)
        np.savez_compressed(path,flat=self.flat,state=state,version=np.array([1],np.int32),
            state_size=np.array([len(state)],np.int32),**{'fixed_'+k:v for k,v in self.fixed.items()})
    @classmethod
    def load(cls,path):
        with np.load(path) as a:
            m=cls(params=unflatten(a['flat']),fixed={k[6:]:a[k] for k in a.files if k.startswith('fixed_')})
            # Keep ownership through the C call. A temporary array's .ctypes.data
            # is only an integer address and does not keep its allocation alive.
            state=np.ascontiguousarray(a['state'],dtype=np.uint8)
            if int(a['version'][0])!=1 or len(state)!=m.lib.state_size():raise ValueError('Incompatible native state layout')
            m.lib.import_state(m.handle,state.ctypes.data)
        return m

def topology_manifest():
    return dict(names=NAMES,width=D,experts_per_region=E,active_experts=1,
                cadence=CADENCE.tolist(),recipes=RECIPES,
                parameters={k:int(np.prod(s)) for k,s in SHAPES.items()},
                total_parameters=sum(int(np.prod(s)) for s in SHAPES.values()))
