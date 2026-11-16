"""Compile instrumentation around the unchanged E8 kernel; bounded research policies."""
import ctypes as C, hashlib, os, subprocess, tempfile
from pathlib import Path
import numpy as np
from experiments.joint_cold_gate import adapter as e8
from experiments.query_first_retention import backend as a2
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
_LIB={}

def header(policy):
    if policy in ('e8','a2'):return a2.header(policy)
    if policy!='a3':raise ValueError(policy)
    text=e8.HERE.joinpath('readout.hpp').read_text().replace('0x38455041514853ULL','0x3341353050584553ULL')
    start=text.index('    for(int k=0;k<records.count;++k)')
    end=text.index('    if(lesion==2||lesion==4)',start)
    text=text[:start]+'''    a3_select(p,records,c,lesion);
    for(int k=0;k<2;++k)for(int r=0;r<VRANK;++r)
        c.values[k][r]=std::tanh(affine(p.value[c.phases[k]*VRANK+r],c.raw[k]));
'''+text[end:]
    anchor='inline QueryCache query_forward'
    assert text.count(anchor)==1
    text=text.replace(anchor,(HERE/'selection.hpp').read_text()+'\n'+anchor)
    return text+'\n'+(ROOT/'experiments/interference_retention/admission.hpp').read_text()

def source(policy='e8'):
    return a2.source('a2' if policy=='a3' else policy)

def library(policy='e8'):
    if policy in _LIB:return _LIB[policy]
    # Reuse E8's complete checked ABI declarations, replacing only library bytes.
    original=e8._LIB
    text=source(policy);head=header(policy);flags=['-std=c++17','-O3','-march=native','-ffp-contract=off','-shared','-fPIC']
    h=hashlib.sha256((text+head+str(flags)).encode())
    for p in sorted((ROOT/'multiscale').glob('*.hpp')):h.update(p.read_bytes())
    h.update(subprocess.check_output(['g++','--version']))
    cache=Path(tempfile.gettempdir())/'shoebox-exp0305a3'/h.hexdigest()[:20];cache.mkdir(parents=True,exist_ok=True)
    so=cache/'model.so'
    if not so.exists():
        (cache/'readout.hpp').write_text(head);(cache/'model.cpp').write_text(text)
        subprocess.run(['g++',*flags,'-I'+str(cache),'-I'+str(ROOT/'multiscale'),str(cache/'model.cpp'),'-o',str(so)],check=True)
    lib=C.CDLL(str(so));base=e8.library()
    for name in ['ms_error','qr_new','qr_free','qr_query_policy','qr_auxiliary','qr_configure','qr_reset','qr_finish','qr_observe','qr_probs','qr_train','qr_score','qr_clone','qr_adopt','qr_diagnostic','qr_info','qr_benchmark','qr_stream_score','qr_gate','qr_features','qr_credit','qr_core_save','qr_save','qr_load']:
        a=getattr(base,name);b=getattr(lib,name);b.argtypes=a.argtypes;b.restype=a.restype
    for name in ['ir_state','ir_decode']:
        fn=getattr(lib,name);fn.argtypes=[C.c_void_p,C.POINTER(C.c_double)];fn.restype=C.c_int
    lib.ir_frozen_span.argtypes=[C.c_void_p,C.c_char_p,C.c_size_t];lib.ir_frozen_span.restype=C.c_int
    if policy in ('a2','a3'):
        lib.a2_probe.argtypes=[C.c_void_p,C.POINTER(C.c_double)];lib.a2_probe.restype=C.c_int
    e8._LIB=original or base;_LIB[policy]=lib;return lib

class Model(e8.Adapter):
    def __init__(self,policy='e8',parameters=500000,seed=1,kind='paired_local',_ptr=None):
        self.policy=policy;self.lib=library(policy);self.ptr=_ptr or self.lib.qr_new(parameters,seed,1,e8.KINDS[kind],1)
        if not self.ptr:raise RuntimeError(self.lib.ms_error().decode())
        if _ptr is None:self.call('qr_configure',.16,1.,.1)
    def clone(self,frozen=True,intervention=None):
        ptr=self.lib.qr_clone(self._alive(),frozen,e8.LESIONS[intervention])
        if not ptr:raise RuntimeError(self.lib.ms_error().decode())
        return Model(self.policy,_ptr=ptr)
    @classmethod
    def loads(cls,raw,policy='e8'):
        if hashlib.sha256(raw[32:]).digest()!=raw[:32]:raise ValueError('checksum mismatch')
        lib=library(policy);ptr=lib.qr_load(raw[32:],len(raw)-32)
        if not ptr:raise RuntimeError(lib.ms_error().decode())
        return cls(policy,_ptr=ptr)
    def state(self):
        out=(C.c_double*1106)();self.call('ir_state',out);a=np.array(out)
        return int(a[0]),int(a[1]),a[2:66],a[66:].reshape(16,65)
    def decoded(self):
        out=(C.c_double*324)();self.call('ir_decode',out);a=np.array(out)
        return a[:4].astype(int).reshape(2,2),a[4:].reshape(16,4,5)
    def freeze_span(self,data):self.call('ir_frozen_span',data,len(data))

    def probe(self):
        out=(C.c_double*10000)();self.call('a2_probe',out);return np.array(out)
