import argparse
import ctypes as C
import json
import os
from pathlib import Path
import numpy as np
from common import Model,build,install,atomic

p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--cpu',type=int,default=3);a=p.parse_args()
os.sched_setaffinity(0,{a.cpu});work=a.base/'diagnostic_40m';build(work/'runtime_gradient');_,lib=install(work/'runtime_gradient')
lib.diag_gradient.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_size_t,C.c_void_p];lib.diag_gradient.restype=C.c_int
manifest=json.loads((work/'protocol.json').read_text());data=np.fromfile(work/'data/train.bin',np.uint8)
# Disjoint documents supply a bounded diagnostic workload. No copy is retained.
docs=manifest['train_documents'][:24]
results={}
for label,relative in [('20M','run_100m'),('40M','run_20m_40m')]:
    aggregate=np.zeros((4,20));total=0
    with Model.load(a.base/relative/'latest.shoe') as m:
        for d in docs:
            m.reset_context(clear_memory=True);start=d['start'];n=min(4096,d['targets'])
            out=np.zeros((4,20));m._check(lib.diag_gradient(m._alive(),data[start:start+n].ctypes.data,data[start+1:start+n+1].ctypes.data,n,out.ctypes.data))
            aggregate[:,:12]+=out[:,:12];aggregate[:,12:14]=np.maximum(aggregate[:,12:14],out[:,12:14]);total+=n
    assert np.all(aggregate[:,12]<2e-6)
    results[label]=dict(targets=total,documents=len(docs),channels=['language','contrastive','SDR_prediction','reconstruction'],
      columns=['blocks','language_energy','contrastive_energy','SDR_energy','reconstruction_energy','language_dot_contrastive','language_dot_SDR','language_dot_reconstruction','contrastive_dot_SDR','contrastive_dot_reconstruction','SDR_dot_reconstruction','combined_energy','maximum_linearity_error','maximum_gradient'],values=aggregate[:,:14].tolist())
    print(label,json.dumps(results[label]),flush=True)
atomic(work/'results/gradient_components.json',results)
