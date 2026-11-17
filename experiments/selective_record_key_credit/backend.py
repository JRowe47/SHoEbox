"""Compile instrumentation around the unchanged E8 kernel; bounded research policies."""
import ctypes as C, hashlib, os, subprocess, tempfile
from pathlib import Path
import numpy as np
from experiments.joint_cold_gate import adapter as e8
from experiments.balanced_record_context import backend as a8
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
_LIB={}

# Training-only credit mask, exactly reports/exp0305a8/A9_REGISTRATION.md.
# The flag is a stack argument; no inference or serialized model state changes.
HEADER_CHANGES={
    'void local(const Labels& labels){':'void local(const Labels& labels,bool allow_record_key_update=true){',
    'if(yes)projection_steps+=QRANK+VRANK;':'if(yes)projection_steps+=(allow_record_key_update?QRANK:0)+VRANK;',
    'a5_reconstruct(head.key.data()+phase*QRANK,x,key,lr*4);':'if(allow_record_key_update)a5_reconstruct(head.key.data()+phase*QRANK,x,key,lr*4);',
}
def header(policy):
    if policy in ('e8','a8'):return a8.header(policy)
    if policy!='a9':raise ValueError(policy)
    text=a8.header('a8')
    assert text.count('0x3841353050584553ULL')==2
    text=text.replace('0x3841353050584553ULL','0x3941353050584553ULL')
    for before,after in HEADER_CHANGES.items():
        assert text.count(before)==1
        text=text.replace(before,after)
    return text

def source(policy='e8'):
    text=a8.source('a8' if policy=='a9' else policy)
    if policy=='a9':
        start=text.index('int qr_train(');end=text.index('\nint ',start+10)
        # Retain the default ABI and add an explicit training-only byte flag ABI.
        train=text[start:end].replace('int qr_train(', 'int a9_train(')
        train=train.replace('const int32_t* labels,size_t n', 'const int32_t* labels,const uint8_t* credit,size_t n')
        train=train.replace('a.local(lab);','a.local(lab,!credit||credit[t]);')
        text+='\nextern "C" {\n'+train+'\n}\n'
    return text

def library(policy='e8'):
    if policy in _LIB:return _LIB[policy]
    # Reuse E8's complete checked ABI declarations, replacing only library bytes.
    original=e8._LIB
    text=source(policy);head=header(policy);flags=['-std=c++17','-O3','-march=native','-ffp-contract=off','-shared','-fPIC']
    h=hashlib.sha256((text+head+str(flags)).encode())
    for p in sorted((ROOT/'multiscale').glob('*.hpp')):h.update(p.read_bytes())
    h.update(subprocess.check_output(['g++','--version']))
    cache=Path(tempfile.gettempdir())/'shoebox-exp0305a9'/h.hexdigest()[:20];cache.mkdir(parents=True,exist_ok=True)
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
    if policy in ('a8','a9'):
        lib.a2_probe.argtypes=[C.c_void_p,C.POINTER(C.c_double)];lib.a2_probe.restype=C.c_int
    if policy in ('a8','a9'):
        lib.a5_objective_probe.argtypes=[C.POINTER(C.c_float),C.c_int,C.POINTER(C.c_double)];lib.a5_objective_probe.restype=C.c_int
        lib.a5_code_probe.argtypes=[C.POINTER(C.c_float)];lib.a5_code_probe.restype=None
    if policy in ('a8','a9'):
        lib.a6_presence_probe.argtypes=[C.c_float,C.c_int,C.POINTER(C.c_float)];lib.a6_presence_probe.restype=None
    if policy=='a9':
        lib.a9_train.argtypes=list(lib.qr_train.argtypes[:5])+[C.c_char_p]+list(lib.qr_train.argtypes[5:])
        lib.a9_train.restype=C.c_int
    e8._LIB=original or base;_LIB[policy]=lib;return lib

class Model(e8.Adapter):
    def __init__(self,policy='e8',parameters=500000,seed=1,kind='paired_local',_ptr=None):
        self.policy=policy;self.lib=library(policy);self.ptr=_ptr or self.lib.qr_new(parameters,seed,1,e8.KINDS[kind],1)
        if not self.ptr:raise RuntimeError(self.lib.ms_error().decode())
        if _ptr is None:self.call('qr_configure',.16,1.,.1)
    def run(self,x,y,mask=None,labels=None,*,allow_record_key_update=True):
        if self.policy!='a9':
            if allow_record_key_update is not True:raise ValueError('credit mask belongs to A9 only')
            return super().run(x,y,mask,labels)
        if len(x)!=len(y) or (mask is not None and len(mask)!=len(x)):raise ValueError('equal lengths required')
        if labels is not None:
            labels=np.ascontiguousarray(labels,dtype=np.int32)
            if labels.shape!=(len(x),6):raise ValueError('six prefix targets per byte required')
        if isinstance(allow_record_key_update,(bool,np.bool_)):
            credit=bytes([int(allow_record_key_update)])*len(x)
        else:
            values=np.asarray(allow_record_key_update)
            if values.shape!=(len(x),) or not np.isin(values,[0,1]).all():raise ValueError('one binary credit flag per byte required')
            credit=values.astype(np.uint8).tobytes()
        loss=C.c_double();targets=C.c_size_t()
        self.call('a9_train',x,y,mask if mask is not None else bytes([1])*len(x),labels.ctypes.data_as(C.POINTER(C.c_int32)) if labels is not None else None,credit,len(x),C.byref(loss),C.byref(targets))
        return loss.value,targets.value
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
