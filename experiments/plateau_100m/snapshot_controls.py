import argparse
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from common import Model,atomic
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--cpu',type=int,default=4);a=p.parse_args()
os.sched_setaffinity(0,{a.cpu});work=a.base/'diagnostic_40m';source=Path(__file__).with_name('accessors.cpp')
subprocess.run(['g++','-std=c++17','-O3','-fPIC','-shared',str(source),'-o',str(work/'accessors.so')],check=True)
lib=C.CDLL(str(work/'accessors.so'));lib.inspect_means.argtypes=[C.c_void_p,C.c_void_p];lib.replace_means.argtypes=[C.c_void_p,C.c_void_p];lib.scale_bias.argtypes=[C.c_void_p,C.c_int,C.c_float]
means20=np.zeros((4,64),np.float32)
with Model.load(a.base/'run_100m/latest.shoe') as old:assert lib.inspect_means(old._alive(),means20.ctypes.data)==0
data=np.fromfile(a.base/'data/development.bin',np.uint8);docs=json.loads((a.base/'data/manifest.json').read_text())['files']['development']['documents']
cases=[('no_bias_fast',0,0.),('no_bias_L1',1,0.),('no_bias_L2',2,0.),('no_bias_L3',3,0.),('quarter_bias_all',-1,.25),('mean_20M',None,None)]
results={}
with Model.load(a.base/'run_20m_40m/latest.shoe') as parent:
 parent.finish()
 for name,level,factor in cases:
    start=time.perf_counter();total=0;count=0;records=[]
    with parent.inference_copy(cold=True,clear_memory=True) as model:
        if level is None:assert lib.replace_means(model._alive(),means20.ctypes.data)==0
        else:assert lib.scale_bias(model._alive(),level,factor)==0
        for d in docs:
            model.reset_context(clear_memory=True);i,j=d['start'],d['stop'];n=j-i-1;loss=model.run(data[i:j-1],data[i+1:j]);total+=n*loss;count+=n
            records.append(dict(row_idx=d['row_idx'],targets=n,nll=loss))
    results[name]=dict(nll=total/count,targets=count,documents=records,seconds=time.perf_counter()-start)
    atomic(work/'results/snapshot_controls.json',results);print(name,results[name]['nll'],flush=True)
