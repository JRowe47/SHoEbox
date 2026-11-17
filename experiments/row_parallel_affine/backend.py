"""Explicit execution backend for the unchanged A9 numerical policy.

No original module is edited. Scalar A9 remains the reference. Same magic,
same bytes, same cursor; backend selection is external metadata only.
"""
import ctypes as C
import hashlib
import subprocess
import tempfile
from pathlib import Path
from experiments.selective_record_key_credit import backend as scalar

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FLAGS=['-std=c++17','-O3','-march=native','-ffp-contract=off','-shared','-fPIC']
_LIB={}

CHANGES={
 'for(int r=0;r<QRANK;++r)c.query[r]=std::tanh(affine(p.key[4*QRANK+r],c.query_raw));':
 'a10_affine<QRANK>(p.key.data()+4*QRANK,c.query_raw,c.query.data());\n    for(int r=0;r<QRANK;++r)c.query[r]=std::tanh(c.query[r]);',
 'for(int r=0;r<QRANK;++r){m.key[r]=std::tanh(affine(p.key[ph*QRANK+r],s.x));':
 'a10_affine<QRANK>(p.key.data()+ph*QRANK,s.x,m.key.data());\n            for(int r=0;r<QRANK;++r){m.key[r]=std::tanh(m.key[r]);',
 'for(int r=0;r<QRANK;++r)c.keys[i][r]=std::tanh(affine(p.key[c.phases[i]*QRANK+r],c.raw[i]));':
 'a10_affine<QRANK>(p.key.data()+c.phases[i]*QRANK,c.raw[i],c.keys[i].data());\n            for(int r=0;r<QRANK;++r)c.keys[i][r]=std::tanh(c.keys[i][r]);',
 'for(int k=0;k<2;++k)for(int r=0;r<VRANK;++r)\n        c.values[k][r]=std::tanh(affine(p.value[c.phases[k]*VRANK+r],c.raw[k]));':
 'for(int k=0;k<2;++k){a10_affine<VRANK>(p.value.data()+c.phases[k]*VRANK,c.raw[k],c.values[k].data());\n        for(int r=0;r<VRANK;++r)c.values[k][r]=std::tanh(c.values[k][r]);}',
 'for(int r=0;r<QRANK;++r)z[r]=std::tanh(affine(rows[r],x));':
 'a10_affine<QRANK>(rows,x,z.data());\n    for(int r=0;r<QRANK;++r)z[r]=std::tanh(z[r]);',
 'for(int r=0;r<VRANK;++r)reconstruct(head.value[phase*VRANK+r],x,value_code(value,r),lr*4);':
 'ValueCode a10_z; a10_affine<VRANK>(head.value.data()+phase*VRANK,x,a10_z.data());\n                    for(int r=0;r<VRANK;++r){float z=std::tanh(a10_z[r]);affine_step(head.value[phase*VRANK+r],x,(z-value_code(value,r))*(1-z*z),lr*4);}',
}

def header():
    text=scalar.header('a9')
    for before,after in CHANGES.items():
        assert text.count(before)==1,before
        text=text.replace(before,after)
    anchor='// Reconstruction targets only.'
    assert text.count(anchor)==1
    return '#include <immintrin.h>\n'+text.replace(anchor,(HERE/'affine.hpp').read_text()+'\n'+anchor)

def source():
    return scalar.source('a9')

def library():
    if 'row_parallel' in _LIB:return _LIB['row_parallel']
    head=header();text=source();compiler=subprocess.check_output(['g++','--version'])
    digest=hashlib.sha256(text.encode()+head.encode()+str(FLAGS).encode()+compiler)
    for p in sorted((ROOT/'multiscale').glob('*.hpp')):digest.update(p.read_bytes())
    cache=Path(tempfile.gettempdir())/'shoebox-exp0305a10'/digest.hexdigest()[:20]
    cache.mkdir(parents=True,exist_ok=True);so=cache/'model.so'
    if not so.exists():
        (cache/'readout.hpp').write_text(head);(cache/'model.cpp').write_text(text)
        subprocess.run(['g++',*FLAGS,'-I'+str(cache),'-I'+str(ROOT/'multiscale'),str(cache/'model.cpp'),'-o',str(so)],check=True)
    reference=scalar.library('a9');lib=C.CDLL(str(so))
    for name in dir(reference):
        if isinstance(getattr(reference,name),C._CFuncPtr):
            a=getattr(reference,name);b=getattr(lib,name);b.argtypes=a.argtypes;b.restype=a.restype
    _LIB['row_parallel']=lib
    return lib

class Model(scalar.Model):
    def __init__(self,policy='a9',parameters=500000,seed=1,kind='paired_local',_ptr=None):
        if policy not in ('e8','a8','a9'):raise ValueError(policy)
        self.policy=policy
        self.lib=library() if policy=='a9' else scalar.library(policy)
        self.ptr=_ptr or self.lib.qr_new(parameters,seed,1,scalar.e8.KINDS[kind],1)
        if not self.ptr:raise RuntimeError(self.lib.ms_error().decode())
        if _ptr is None:self.call('qr_configure',.16,1.,.1)
    def clone(self,frozen=True,intervention=None):
        ptr=self.lib.qr_clone(self._alive(),frozen,scalar.e8.LESIONS[intervention])
        if not ptr:raise RuntimeError(self.lib.ms_error().decode())
        return type(self)(self.policy,_ptr=ptr)
    @classmethod
    def loads(cls,raw,policy='a9'):
        if len(raw)<32 or hashlib.sha256(raw[32:]).digest()!=raw[:32]:raise ValueError('checksum mismatch')
        lib=library() if policy=='a9' else scalar.library(policy)
        ptr=lib.qr_load(raw[32:],len(raw)-32)
        if not ptr:raise RuntimeError(lib.ms_error().decode())
        return cls(policy,_ptr=ptr)
