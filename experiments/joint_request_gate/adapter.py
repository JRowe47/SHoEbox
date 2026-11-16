"""EXP01-E7: calibrated bit-code byte distribution and learned applicability."""
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
    return '#include "readout.hpp"\n'+text+'\n'+(HERE/'api.cpp').read_text()


def library():
    global _LIB
    if _LIB is not None:return _LIB
    text=source();compiler=os.environ.get('CXX','g++')
    flags=['-std=c++17','-O3','-march=native','-ffp-contract=off','-shared','-fPIC']
    h=hashlib.sha256(text.encode()+str(flags).encode()+subprocess.check_output([compiler,'--version']))
    for p in [*sorted(HERE.glob('*.hpp')),*sorted((ROOT/'multiscale').glob('*.hpp'))]:h.update(p.read_bytes())
    cache=Path(tempfile.gettempdir())/'shoebox-joint-request-gate';cache.mkdir(exist_ok=True)
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
    _LIB=lib;return lib


class Adapter:
    def __init__(self,parameters=500000,seed=1,core='minimal',kind='paired_local',joint=True,rate=.16,lr_scale=1.,auxiliary=.1,*,_ptr=None):
        if core not in ('minimal','full'):raise ValueError('core must be named')
        self.lib=library();self.ptr=_ptr or self.lib.qr_new(parameters,seed,core=='minimal',KINDS[kind],joint)
        if not self.ptr:raise RuntimeError(self.lib.ms_error().decode())
        if _ptr is None:
            try:self.call('qr_configure',rate,lr_scale,auxiliary)
            except Exception:self.close();raise
    def _alive(self):
        if not self.ptr:raise RuntimeError('closed adapter')
        return self.ptr
    def call(self,name,*args):
        if getattr(self.lib,name)(self._alive(),*args):raise RuntimeError(self.lib.ms_error().decode())
    def close(self):
        if self.ptr:self.lib.qr_free(self.ptr);self.ptr=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def reset(self):self.call('qr_reset')
    def finish(self):self.call('qr_finish')
    def observe(self,byte):
        if type(byte) is not int or not 0<=byte<256:raise ValueError('byte required')
        self.call('qr_observe',byte)
    def log_probs(self):
        out=(C.c_float*256)();self.call('qr_probs',out);return np.array(out)
    def run(self,x,y,mask=None,labels=None):
        if len(x)!=len(y) or (mask is not None and len(mask)!=len(x)):raise ValueError('equal lengths required')
        if labels is not None:
            labels=np.ascontiguousarray(labels,dtype=np.int32)
            if labels.shape!=(len(x),6):raise ValueError('six prefix targets per byte required')
        loss=C.c_double();targets=C.c_size_t()
        self.call('qr_train',x,y,mask if mask is not None else bytes([1])*len(x),labels.ctypes.data_as(C.POINTER(C.c_int32)) if labels is not None else None,len(x),C.byref(loss),C.byref(targets))
        return loss.value,targets.value
    def train_episode(self,ep,remaining):
        if remaining<0:raise ValueError('nonnegative budget required')
        self.reset();used=min(len(ep.data)-1,remaining)
        loss,n=self.run(ep.data[:used],ep.data[1:used+1],ep.mask[1:used+1],prefix_labels(ep)[:used])
        return used,n,loss,None
    def clone(self,frozen=True,intervention=None):
        ptr=self.lib.qr_clone(self._alive(),frozen,LESIONS[intervention])
        if not ptr:raise RuntimeError(self.lib.ms_error().decode())
        return Adapter(_ptr=ptr)
    def adopt(self,other):self.call('qr_adopt',other._alive())
    def score(self,ep,vocab=2,rollout=0):
        from experiments.mad.tasks import VALUE
        if len(ep.data)!=len(ep.mask):raise ValueError('equal lengths required')
        self.reset();out=(C.c_double*9)()
        self.call('qr_score',ep.data,ep.mask,len(ep.data),VALUE,vocab,rollout,out);return list(out)
    def diagnostic(self):
        out=(C.c_float*6)();self.call('qr_diagnostic',out);return np.array(out)
    def benchmark(self,data,count):
        if not data or count<0:raise ValueError('nonempty data and nonnegative count required')
        out=C.c_double();self.call('qr_benchmark',data,len(data),count,C.byref(out));return out.value
    def inventory(self):
        out=(C.c_double*24)();self.call('qr_info',out)
        return dict(core_parameters=int(out[0]),readout_parameters=int(out[1]),parameters=int(out[0]+out[1]),
                    bytes=int(out[2]),head_targets=int(out[3]),training=bool(out[4]),joint=bool(out[5]),
                    kind=list(KINDS)[int(out[6])],core='minimal' if out[7] else 'full',
                    readout_cache_bytes=int(out[8]),experts_per_level=list(map(int,out[9:13])),
                    allocated_head_bytes=int(out[13]),allocated_adapter_bytes=int(out[14]),
                    added_persistent_sequence_state_bytes=0,optimizer_state_bytes=int(out[17]),rls_updates=int(out[18]),normalization_bytes=int(out[15]),normalization_samples=0,local_training_windows=int(out[16]),existing_stream_and_retained_bytes=int(out[19]),source_count=int(out[20]),max_sources=64,gate_parameters=65 if int(out[6])>=3 else 0,gate_targets=int(out[21]),local_projection_steps=int(out[22]),local_presence_steps=int(out[23]))
    def gate(self):
        out=C.c_float();self.call('qr_gate',C.byref(out));return out.value
    def evaluate_bytes(self,data):
        with self.clone() as f:
            f.reset();out=C.c_double();f.call('qr_stream_score',data,len(data),C.byref(out));return out.value
    def features(self):
        out=(C.c_float*64)();self.call('qr_features',out);return np.array(out)
    def credit(self,dx):self.call('qr_credit',(C.c_float*64)(*dx))
    def core_dumps(self):
        n=C.c_size_t();self.call('qr_core_save',None,0,C.byref(n));buf=C.create_string_buffer(n.value)
        self.call('qr_core_save',buf,n.value,C.byref(n));return buf.raw
    def dumps(self):
        n=C.c_size_t();self.call('qr_save',None,0,C.byref(n));buf=C.create_string_buffer(n.value)
        self.call('qr_save',buf,n.value,C.byref(n));payload=buf.raw
        return hashlib.sha256(payload).digest()+payload
    @classmethod
    def loads(cls,raw):
        if len(raw)<32 or hashlib.sha256(raw[32:]).digest()!=raw[:32]:raise ValueError('query checkpoint checksum mismatch')
        lib=library();ptr=lib.qr_load(raw[32:],len(raw)-32)
        if not ptr:raise RuntimeError(lib.ms_error().decode())
        return cls(_ptr=ptr)


def prefix_labels(ep):
    """Local syntax/byte reconstruction labels, training only, no lookup.

    Each target is determined exclusively by a newly completed observed window.
    Labels contain neither a selected record index nor any unobserved answer.
    """
    from experiments.mad.tasks import MAP,QUERY,END,SEP,ANSWER
    result=np.full((len(ep.data)-1,6),-1,dtype=np.int32)
    for t in range(len(result)):
        result[t,5]=int(ep.data[t]==ANSWER) # observed request only; no answer mask
        if t>=1 and ep.data[t-1]==QUERY:result[t,0]=ep.data[t]
        if t>=7 and (t+1)%4==0:
            window=ep.data[t-7:t+1]
            for phase in range(4):
                if window[phase]==MAP and window[phase+2]==SEP and window[phase+4]==END:
                    result[t,phase+1]=256*window[phase+1]+window[phase+3]
    return result
