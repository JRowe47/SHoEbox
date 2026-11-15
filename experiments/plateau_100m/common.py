"""Private continuation probes; preserve the source checkpoint and production code."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from shoe_multiscale import Model
from multiscale import model as runtime
from experiments.scale_100m.train import DocumentChunks, frozen_score, atomic, file_hash

def build(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'multiscale').glob('*.hpp'):shutil.copy2(p,directory/p.name)
    text=(ROOT/'multiscale/native.cpp').read_text()
    text=text.replace('namespace shoe {','namespace shoe {\nint diagnostic_balance_update=1;\nfloat diagnostic_sdr_weight=.1f;\nfloat diagnostic_aux_factor[4]{1,1,1,1};',1)
    assert text.count('native_scale=aux_weight/std::max(1,targets);')==1
    text=text.replace('native_scale=aux_weight/std::max(1,targets);','native_scale=diagnostic_aux_factor[scale]*aux_weight/std::max(1,targets);')
    assert text.count('x*=.1f/std::max(1,sdr_targets);')==1
    text=text.replace('x*=.1f/std::max(1,sdr_targets);','x*=diagnostic_sdr_weight/std::max(1,sdr_targets);')
    routing=(directory/'routing_geometry_impl.hpp').read_text()
    assert routing.count('if(routing_arm>=2){')==1
    routing=routing.replace('if(routing_arm>=2){','if(routing_arm>=2&&diagnostic_balance_update){')
    (directory/'routing_geometry_impl.hpp').write_text(routing)
    text+='\n'+(Path(__file__).with_name('inspection.cpp')).read_text()
    (directory/'native.cpp').write_text(text)
    subprocess.run(['g++','-std=c++17','-O3','-march=native','-ffp-contract=off','-fPIC','-shared',str(directory/'native.cpp'),'-o',str(directory/'diagnostic.so')],check=True)

def install(directory):
    base=runtime._library();lib=C.CDLL(str(Path(directory).resolve()/'diagnostic.so'))
    for name in vars(base):
        if name.startswith('ms_') or name=='plateau_probe':
            f=getattr(base,name);g=getattr(lib,name);g.argtypes=f.argtypes;g.restype=f.restype
    lib.diag_policy.argtypes=[C.c_int,C.c_float];lib.diag_policy.restype=C.c_int
    lib.diag_control.argtypes=[C.c_void_p,C.c_int,C.c_int];lib.diag_control.restype=C.c_int
    lib.diag_trace.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_size_t,C.c_void_p,C.c_void_p,C.c_void_p];lib.diag_trace.restype=C.c_int
    lib.diag_bank.argtypes=[C.c_void_p,C.c_int,C.c_void_p];lib.diag_bank.restype=C.c_int
    if hasattr(lib,'diag_set_bias'):
        lib.diag_set_bias.argtypes=[C.c_void_p,C.c_float];lib.diag_set_bias.restype=C.c_int
        lib.diag_aux_levels.argtypes=[C.c_float]*4;lib.diag_aux_levels.restype=C.c_int
        lib.diag_validate_gradient.argtypes=[C.c_void_p,C.c_void_p];lib.diag_validate_gradient.restype=C.c_int
    runtime._LIBS['addressed']=lib
    return base,lib

def score(model,data,docs,intervention='full',details=False):
    total=0.;count=0;records=[];trace=np.zeros((4,24));routes=[[] for _ in range(4)]
    with model.clone() as retired:
        retired.finish()
        with retired.inference_copy(cold=True,clear_memory=True) as frozen:
            if intervention.startswith('mask'):frozen.intervene(mask=int(intervention[-1]))
            elif intervention.startswith('swap'):frozen.swap_experts(int(intervention[-1]))
            elif intervention=='no_recall':frozen.intervene_recall('no_read')
            elif intervention=='no_address':frozen.intervene_memory('no_read')
            elif intervention=='no_bias':frozen._check(frozen._lib.diag_control(frozen._alive(),1,0))
            elif intervention=='fast_only':frozen._check(frozen._lib.diag_control(frozen._alive(),2,0))
            elif intervention.startswith('zero_expert'):frozen._check(frozen._lib.diag_control(frozen._alive(),3,int(intervention[-1])))
            for d in docs:
                frozen.reset_context(clear_memory=True);a,b=d['start'],d['stop'];n=b-a-1
                if details:
                    local=np.zeros((4,24));rr=np.full((4,n),-1,np.int32);losses=np.zeros(n)
                    frozen._check(frozen._lib.diag_trace(frozen._alive(),data[a:b-1].ctypes.data,data[a+1:b].ctypes.data,n,local.ctypes.data,rr.ctypes.data,losses.ctypes.data))
                    loss=float(losses.mean());trace+=local
                    for l in range(4):routes[l].append(rr[l,rr[l]>=0])
                else:loss=frozen.run(data[a:b-1],data[a+1:b])
                total+=loss*n;count+=n;records.append(dict(row_idx=d['row_idx'],targets=n,nll=loss))
    output=dict(nll=total/count,targets=count,documents=records)
    if details:
        output['trace']=trace.tolist();output['routing']=[]
        for l,items in enumerate(routes):
            r=np.concatenate(items);counts=np.bincount(r,minlength=model.stats()['experts'][l]);p=counts/counts.sum();nz=p[p>0]
            output['routing'].append(dict(events=int(counts.sum()),distinct=int((counts>0).sum()),effective=float(np.exp(-(nz*np.log(nz)).sum())),maximum_share=float(p.max()),counts=counts.tolist(),routes=r.tolist()))
    return output

def take_documents(documents,budget):
    result=[];remaining=budget
    for doc in documents:
        d=dict(doc);n=min(d['stop']-d['start']-1,remaining);d['stop']=d['start']+n+1;d['targets']=n
        result.append(d);remaining-=n
        if not remaining:break
    assert remaining==0
    return result
