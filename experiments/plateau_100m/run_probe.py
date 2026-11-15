import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from common import Model,install,score,atomic,file_hash

p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--intervention',default='full');p.add_argument('--cpu',type=int,default=0)
p.add_argument('--checkpoint',default='40');p.add_argument('--details',action='store_true')
p.add_argument('--dataset',choices=['development','screen','audit'],default='development');a=p.parse_args()
os.sched_setaffinity(0,{a.cpu});install(a.runtime)
data=np.fromfile(a.base/'data/development.bin',np.uint8);docs=json.loads((a.base/'data/manifest.json').read_text())['files']['development']['documents']
if a.dataset!='development':
    data=np.fromfile(a.base/'diagnostic_40m/data/train.bin',np.uint8)
    docs=json.loads((a.base/'diagnostic_40m/protocol.json').read_text())['fresh_'+a.dataset+'_documents']
checkpoint=a.base/('run_100m/latest.shoe' if a.checkpoint=='20' else 'run_20m_40m/latest.shoe')
start=time.perf_counter()
with Model.load(checkpoint) as model:
    result=score(model,data,docs,a.intervention,a.details)
    result.update(checkpoint_targets=int(a.checkpoint)*1_000_000,intervention=a.intervention,dataset=a.dataset,seconds=time.perf_counter()-start)
    if a.details:
        banks=[]
        for level,n in enumerate(model.stats()['experts']):
            arr=np.zeros((n,5));model._check(model._lib.diag_bank(model._alive(),level,arr.ctypes.data));banks.append(arr.tolist())
        result['banks']=banks
atomic(a.out,result)
print(json.dumps(dict(checkpoint=a.checkpoint,intervention=a.intervention,nll=result['nll'],seconds=result['seconds'])),flush=True)
