"""Development-only localization: first query versus mixed query positions.

These batch ridge probes are diagnostics, never native predictions or training
arms. They explain why the prior first-answer result does not qualify a shared
readout across all query positions. Fitting/standardization use train only.
"""
import argparse
import hashlib
import os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from .adapter import Adapter,prefix_labels
from .tasks import Spec,dataset
from experiments.mad_next.tasks import fingerprint
from experiments.query_readout.run import write

def collect(model,cases):
    xs=[];ys=[]
    with model.clone() as f:
        for ep in cases:
            f.reset();labels=prefix_labels(ep)
            for t,byte in enumerate(ep.data[:-1]):
                f.observe(byte)
                if ep.mask[t+1]:xs.append(f.features()[:64]);ys.append(labels[t])
    return np.array(xs,dtype=float),np.array(ys)

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    spec=Spec('binding',vocab=2);tr=dataset(spec,2048,'train');dv=dataset(spec,256,'dev')
    with Adapter.loads(a.checkpoint.read_bytes()) as m:x,y=collect(m,tr);z,t=collect(m,dv)
    rows={}
    for name,sl in [('both_queries',slice(None)),('first_query',slice(None,None,2))]:
        xx=x[sl];yy=y[sl];zz=z[sl];tt=t[sl];mean=xx.mean(0);scale=np.maximum(xx.std(0),1e-6)
        xx=np.c_[(xx-mean)/scale,np.ones(len(xx))];zz=np.c_[(zz-mean)/scale,np.ones(len(zz))]
        predictions=[]
        for column in range(5):
            w=np.linalg.solve(xx.T@xx+np.diag([1]*64+[0]),xx.T@np.eye(256)[yy[:,column]])
            predictions.append((zz@w).argmax(1))
        predictions=np.array(predictions).T
        rows[name]=dict(accuracy=(predictions==tt).mean(0).tolist(),train_samples=len(xx),dev_samples=len(zz),parameters_per_head=65*256)
    write(a.out,dict(columns=['query','key0','key1','value0','value1'],rows=rows,scope='development-only offline ridge, not scored native capability',
        ridge=1,normalization='training only, unpenalized intercept',checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),train_sha256=fingerprint(tr),dev_sha256=fingerprint(dv)))

if __name__=='__main__':main()
