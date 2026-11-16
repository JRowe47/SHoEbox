"""Isolated native research adapter; production sources and ABI are unchanged."""
import ctypes as C
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ARMS={'final_only':0,'head_only':1,'state_aux':2}

def source():
    text=(ROOT/'multiscale/native.cpp').read_text()
    anchors=[('for(int i=0;i<D;++i)dh[i]+=language_credit[scale]*task[i];',
              '\n        task_credit(this,scale,t,dh);'),
             ('s.count=0;++s.updates;','\n    task_retired(this,scale);')]
    for anchor,extra in anchors:
        if text.count(anchor)!=1:raise RuntimeError('native hook changed; review the research adapter')
        text=text.replace(anchor,anchor+extra)
    return '#include "adapter.hpp"\n'+text+'\n'+(HERE/'adapter_api.cpp').read_text()

def library():
    text=source();compiler=os.environ.get('CXX','g++')
    flags=['-std=c++17','-O3','-march=native','-ffp-contract=off','-shared','-fPIC']
    digest=hashlib.sha256(str(flags).encode()+subprocess.check_output([compiler,'--version'])+text.encode()+(HERE/'adapter.hpp').read_bytes())
    for p in sorted((ROOT/'multiscale').glob('*.hpp')):digest.update(p.read_bytes())
    cache=Path(tempfile.gettempdir())/'shoebox-structured-state';cache.mkdir(exist_ok=True)
    path=cache/(digest.hexdigest()[:20]+'.so')
    if not path.exists():
        cpp=path.with_suffix('.cpp');cpp.write_text(text)
        temporary=path.with_suffix(f'.{os.getpid()}.so')
        subprocess.run([compiler,*flags,'-I'+str(ROOT/'multiscale'),'-I'+str(HERE),str(cpp),'-o',str(temporary)],check=True)
        temporary.replace(path)
    lib=C.CDLL(str(path));void=C.c_void_p;fptr=C.POINTER(C.c_float);dptr=C.POINTER(C.c_double)
    signatures={'st_error':([],C.c_char_p),'st_new':([C.c_int,C.c_uint32,C.c_int,C.c_uint64],void),
        'st_free':([void],None),'st_run':([void,C.c_char_p,C.c_char_p,C.c_size_t,dptr],C.c_int),
        'st_supervise':([void,C.POINTER(C.c_int),C.c_int,dptr],C.c_int),
        'st_read':([void,fptr,fptr,fptr],C.c_int),'st_reset':([void],C.c_int),'st_finish':([void],C.c_int),
        'st_clone':([void,C.c_int],void),'st_info':([void,dptr],C.c_int),
        'st_permute':([void,C.c_int,C.c_uint32,C.POINTER(C.c_int)],C.c_int),
        'st_native_probs':([void,fptr],C.c_int),
        'st_save':([void,void,C.c_size_t,C.POINTER(C.c_size_t)],C.c_int),'st_load':([C.c_char_p,C.c_size_t],void)}
    for name,(args,result) in signatures.items():f=getattr(lib,name);f.argtypes=args;f.restype=result
    return lib
_LIB=None
class Adapter:
    def __init__(self,parameters=1000000,seed=7,arm='final_only',budget=3000000,*,_ptr=None):
        global _LIB
        if _LIB is None:_LIB=library()
        self.lib=_LIB;self.ptr=_ptr or self.lib.st_new(parameters,seed,ARMS[arm],budget)
        if not self.ptr:raise RuntimeError(self.lib.st_error().decode())
    def _check(self,code):
        if code:raise RuntimeError(self.lib.st_error().decode())
    def close(self):
        if self.ptr:self.lib.st_free(self.ptr);self.ptr=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def run(self,x,y):
        if not x or len(x)!=len(y):raise ValueError('nonempty equal-length input and targets required')
        loss=C.c_double();self._check(self.lib.st_run(self.ptr,x,y,len(x),C.byref(loss)));return loss.value
    def supervise(self,labels,intermediate=False):
        if len(labels)!=4:raise ValueError('four register labels required')
        y=(C.c_int*4)(*labels);loss=C.c_double();self._check(self.lib.st_supervise(self.ptr,y,intermediate,C.byref(loss)));return loss.value
    def read(self,override=None):
        x=(C.c_float*128)();p=(C.c_float*32)();over=None
        if override is not None:
            over=np.asarray(override,dtype=np.float32)
            if over.shape!=(128,) or not np.isfinite(over).all():raise ValueError('expected 128 finite features')
            over=(C.c_float*128)(*over)
        self._check(self.lib.st_read(self.ptr,x,p,over));return np.array(x),np.array(p).reshape(4,8)
    def native_probs(self):
        out=(C.c_float*256)();self._check(self.lib.st_native_probs(self.ptr,out));return np.exp(np.array(out,dtype=np.float64))
    def reset(self):self._check(self.lib.st_reset(self.ptr))
    def finish(self):self._check(self.lib.st_finish(self.ptr))
    def clone(self,frozen=False):
        ptr=self.lib.st_clone(self.ptr,frozen)
        if not ptr:raise RuntimeError(self.lib.st_error().decode())
        return Adapter(_ptr=ptr)
    def info(self):
        out=(C.c_double*15)();self._check(self.lib.st_info(self.ptr,out));return list(out)
    def permute(self,scale,seed=23):
        count=C.c_int();self._check(self.lib.st_permute(self.ptr,scale,seed,C.byref(count)));return count.value
    def dumps(self):
        n=C.c_size_t();self._check(self.lib.st_save(self.ptr,None,0,C.byref(n)));buf=C.create_string_buffer(n.value)
        self._check(self.lib.st_save(self.ptr,buf,n.value,C.byref(n)));payload=buf.raw
        return hashlib.sha256(payload).digest()+payload
    @classmethod
    def loads(cls,raw):
        if len(raw)<32 or hashlib.sha256(raw[32:]).digest()!=raw[:32]:raise ValueError('task checkpoint checksum mismatch')
        obj=cls();ptr=obj.lib.st_load(raw[32:],len(raw)-32)
        if not ptr:obj.close();raise RuntimeError(obj.lib.st_error().decode())
        obj.close();obj.ptr=ptr;return obj
