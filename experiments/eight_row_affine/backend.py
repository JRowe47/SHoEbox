"""A11 explicitly selected fixed eight-row backend for unchanged A9.

No original module is edited. Scalar A9 remains the reference. Same magic,
same bytes, same cursor; backend selection is external metadata only.
"""
import ctypes as C
import hashlib
import subprocess
import tempfile
from pathlib import Path
from experiments.selective_record_key_credit import backend as scalar
from experiments.row_parallel_affine.backend import CHANGES as A10_CHANGES

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FLAGS=['-std=c++17','-O3','-march=native','-ffp-contract=off','-shared','-fPIC']
_LIB={}

CHANGES={before:after.replace('a10_affine','a11_affine') for before,after in A10_CHANGES.items()}

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
    if 'eight_row' in _LIB:return _LIB['eight_row']
    if 'avx2' not in Path('/proc/cpuinfo').read_text().split():raise RuntimeError('A11 requires AVX2')
    head=header();text=source();compiler=subprocess.check_output(['g++','--version'])
    digest=hashlib.sha256(text.encode()+head.encode()+str(FLAGS).encode()+compiler)
    for p in sorted((ROOT/'multiscale').glob('*.hpp')):digest.update(p.read_bytes())
    cache=Path(tempfile.gettempdir())/'shoebox-exp0305a11'/digest.hexdigest()[:20]
    cache.mkdir(parents=True,exist_ok=True);so=cache/'model.so'
    if not so.exists():
        (cache/'readout.hpp').write_text(head);(cache/'model.cpp').write_text(text)
        subprocess.run(['g++',*FLAGS,'-I'+str(cache),'-I'+str(ROOT/'multiscale'),str(cache/'model.cpp'),'-o',str(so)],check=True)
    reference=scalar.library('a9');lib=C.CDLL(str(so))
    for name in dir(reference):
        if isinstance(getattr(reference,name),C._CFuncPtr):
            a=getattr(reference,name);b=getattr(lib,name);b.argtypes=a.argtypes;b.restype=a.restype
    _LIB['eight_row']=lib
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
