"""Read-only evidence-presence audit. Oracle locations are DIAGNOSTIC ONLY.

Reconstruct fixed constituent signatures independently in Python, then ask
whether the exact existing retained record is still present at each answer.
No fitted decoder, training update, inference hook, or native answer uses this.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from .adapter import Adapter, HERE, ROOT
from .tasks import dataset, shifted
from experiments.mad_next.tasks import MAP, QUERY
from experiments.query_readout.run import write

CPP='''#include "readout.hpp"
extern "C" void access_inspect(void* p,float* out,int* counts){
    const auto& a=*static_cast<const shoe::QueryAdapter*>(p);
    for(int l=0;l<4;++l){const auto& r=a.model->levels[l].retained;counts[l]=r.count;
        for(int k=0;k<r.count;++k)std::copy(r.units[k].stable.begin(),r.units[k].stable.end(),out+(l*16+k)*64);
    }
}
'''

def inspector():
    h=hashlib.sha256(CPP.encode())
    for p in [*HERE.glob('*.hpp'),*sorted((ROOT/'multiscale').glob('*.hpp'))]:h.update(p.read_bytes())
    folder=Path(tempfile.gettempdir())/'shoebox-access-inspect';folder.mkdir(exist_ok=True)
    path=folder/(h.hexdigest()+'.so')
    if not path.exists():
        cpp=path.with_suffix('.cpp');cpp.write_text(CPP)
        subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I'+str(HERE),'-I'+str(ROOT/'multiscale'),str(cpp),'-o',str(path)],check=True)
        cpp.unlink()
    lib=C.CDLL(str(path));lib.access_inspect.argtypes=[C.c_void_p,C.POINTER(C.c_float),C.POINTER(C.c_int)]
    lib.access_inspect.restype=None
    return lib

def byte_stable(byte):
    result=[]
    for j in range(64):
        x=((byte+1)*0x9e3779b9 ^ (j+1)*0x85ebca6b)&0xffffffff
        x^=x>>16;x=x*0x7feb352d&0xffffffff;x^=x>>15;x=x*0x846ca68b&0xffffffff;x^=x>>16
        result.append(.125 if x&1 else -.125)
    return np.array(result,dtype=np.float32)

def normalize(x):return x/np.sqrt(np.dot(x,x)+np.float32(1e-8))

def stable_group(children):
    out=np.zeros(64,dtype=np.float32)
    for p in range(4):
        for j in range(64):out[j]+=(-.5 if (j+p)%3==0 else .5)*children[p][(j*(2*p+1)+13*p)%64]
    return normalize(out)

def pair(previous,current):return normalize(current+previous[(np.arange(64)*3+13)%64])

def record_signature(data,at):
    start=at//4*4
    assert start<=at and at+5<=start+8
    chunks=[stable_group([byte_stable(b) for b in data[s:s+4]]) for s in (start,start+4)]
    return pair(*chunks)

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    # No model fitting or selection: this is an exact signature-presence check.
    plan=dict(scope='diagnostic-only oracle record spans; excluded from native inference',seed=7,cases=64,offsets=64,delays=[0,64,128,256],fitted_parameters=0,training_bytes=0,prefix_modes=['teacher','native'])
    write(a.out.with_name('source_presence_plan.json'),plan)
    lib=inspector();cases=dataset(64,'test');rows=[]
    with Adapter.loads((a.runs/'7'/'pool.qr').read_bytes()) as m,m.clone() as f:
        for mode in plan['prefix_modes']:
            for delay in plan['delays']:
                totals=np.zeros((2,3));n=0
                for offset in range(64):
                    for base in cases:
                        ep=shifted(base,offset,delay);records=[i for i,b in enumerate(ep.data) if b==MAP]
                        record_codes=[record_signature(ep.data,at) for at in records]
                        query_positions=[i for i,b in enumerate(ep.data) if b==QUERY]
                        query_codes=[pair(byte_stable(QUERY),byte_stable(ep.data[i+1])) for i in query_positions]
                        f.reset();j=0
                        for t,byte in enumerate(ep.data[:-1]):
                            if mode=='native' and ep.mask[t]:byte=int(f.log_probs().argmax())
                            f.observe(byte)
                            if ep.mask[t+1]:
                                out=np.zeros((4,16,64),dtype=np.float32);counts=np.zeros(4,dtype=np.int32)
                                lib.access_inspect(f.ptr,out.ctypes.data_as(C.POINTER(C.c_float)),counts.ctypes.data_as(C.POINTER(C.c_int)))
                                for k,code in enumerate([query_codes[j],*record_codes]):
                                    level=0 if k==0 else 1
                                    totals[j,k]+=any(np.max(np.abs(x-code))<2e-6 for x in out[level,:counts[level]])
                                j+=1
                        n+=1
                rows.append(dict(prefix_mode=mode,delay=delay,present_by_answer=(totals/n).tolist(),presentations=n))
    write(a.out,dict(plan=plan,columns=['current_query_pair','record0_containing_8byte_pair','record1_containing_8byte_pair'],rows=rows))

if __name__=='__main__':main()
