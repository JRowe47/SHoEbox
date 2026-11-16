"""Pinned E8 arithmetic; public bridge adds lifecycle and format validation."""
import ctypes as C
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PAIRED_PARAMETERS=5*65+5*16*65+4*8*65
PARAMETERS=PAIRED_PARAMETERS+65
FEATURES=5*64+5*16+1
KINDS={'native':0,'paired':1,'paired_latest':2,'paired_gated':3,'paired_local':4}
LESIONS={None:0,'no_head':1,'no_query':2,'no_state':3,'uniform':4,'reverse':5,'unpair':6,'gate_off':7,'gate_on':8}
_LIB=None


def source():
    text=(ROOT/'multiscale/native.cpp').read_text()
    changes=[('float Model::teach(uint8_t target){','\n    query_prepare(this);query_target(this,target);'),
             ('void Model::log_probs(float* out)const{','\n    query_prepare(this);'),
             ('loss+=softplus(z)-bit*z;',''),
             ('node=2*node+1+bit;\n    }','\n    query_backward(this,recall_grad);')]
    for anchor,addition in changes:
        if text.count(anchor)!=1:raise RuntimeError('query hook changed: '+anchor)
        text=text.replace(anchor,anchor+addition)
    text=text.replace('loss+=softplus(z)-bit*z;',
                      'z+=query_logit(this,node);\n        query_error(this,node,sigmoid(z)-bit);\n        loss+=softplus(z)-bit*z;')
    anchor='p[2*n+1]=p[n]-softplus(z);'
    if text.count(anchor)!=1:raise RuntimeError('query inference hook changed')
    text=text.replace(anchor,'z+=query_logit(this,n);\n        '+anchor)
    text=text.replace('float error=sigmoid(z)-bit;', 'float error=(sigmoid(z)-bit)*query_native_credit(this);')
    text=text.replace('return loss;\n}\nvoid Model::log_probs', 'return query_loss(this,loss);\n}\nvoid Model::log_probs')
    text=text.replace('std::copy(p.begin()+255,p.end(),out);', 'std::copy(p.begin()+255,p.end(),out);query_mix_probs(this,out);')
    return '#include "readout.hpp"\n'+text+'\n'+(HERE/'api.cpp').read_text()+'\n'+(HERE/'public.cpp').read_text()


def library():
    global _LIB
    if _LIB is not None:return _LIB
    text=source();compiler=os.environ.get('CXX','g++')
    flags=['-std=c++17','-O3','-march=native','-ffp-contract=off','-shared','-fPIC']
    h=hashlib.sha256(text.encode()+str(flags).encode()+subprocess.check_output([compiler,'--version']))
    for p in [*sorted(HERE.glob('*.hpp')),*sorted((ROOT/'multiscale').glob('*.hpp'))]:h.update(p.read_bytes())
    cache=Path(tempfile.gettempdir())/'shoebox-public-e8';cache.mkdir(exist_ok=True)
    path=cache/(h.hexdigest()[:20]+'.so')
    if not path.exists():
        cpp=path.with_suffix(f'.{os.getpid()}.cpp');cpp.write_text(text)
        temporary=path.with_suffix(f'.{os.getpid()}.so')
        subprocess.run([compiler,*flags,'-I'+str(ROOT/'multiscale'),'-I'+str(HERE),str(cpp),'-o',str(temporary)],check=True)
        temporary.replace(path);cpp.unlink()
    lib=C.CDLL(str(path));p=C.c_void_p;i=C.c_int;f=C.c_float;n=C.c_size_t;s=C.c_char_p
    fp=C.POINTER(f);dp=C.POINTER(C.c_double)
    specs={'ms_error':([],s),'qr_new':([i,C.c_uint32,i,i,i],p),'qr_free':([p],None),
        'qr_query_policy':([p,i],i),'qr_auxiliary':([p,f],i),'qr_configure':([p,f,f,f],i),'qr_reset':([p],i),'qr_finish':([p],i),'qr_observe':([p,C.c_uint8],i),
        'qr_probs':([p,fp],i),'qr_train':([p,s,s,s,C.POINTER(C.c_int32),n,dp,C.POINTER(n)],i),
        'qr_score':([p,s,s,n,i,i,i,dp],i),'qr_clone':([p,i,i],p),'qr_adopt':([p,p],i),
        'qr_diagnostic':([p,fp],i),'qr_info':([p,dp],i),'qr_benchmark':([p,s,n,n,dp],i),
        'qr_stream_score':([p,s,n,dp],i),'qr_gate':([p,fp],i),'qr_features':([p,fp],i),'qr_credit':([p,fp],i),'qr_core_save':([p,p,n,C.POINTER(n)],i),'qr_save':([p,p,n,C.POINTER(n)],i),'qr_load':([s,n],p)}
    for name,(args,result) in specs.items():fn=getattr(lib,name);fn.argtypes=args;fn.restype=result
    from multiscale.model import _bind_library
    _bind_library(lib, "addressed")
    extra = {"pe_core": ([p], p), "pe_load": ([s,n], p),
             "pe_run": ([p,p,p,p,p,n,dp,C.POINTER(n)], i),
             "pe_clone": ([p,i],p), "pe_observe": ([p,C.c_uint8,fp],i)}
    for name,(args,result) in extra.items():
        fn=getattr(lib,name);fn.argtypes=args;fn.restype=result
    _LIB=lib;return lib
