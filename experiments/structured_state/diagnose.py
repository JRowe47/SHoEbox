"""Frozen diagnostic: does an independent linear probe recover register state?"""
import argparse
import json
import os
from pathlib import Path
import random
import time
import numpy as np
from .adapter import Adapter
from .tasks import episode,suite

def extract(m,cases):
    features=[[],[]];labels=[];queries=[];online=[[],[]]
    with m.clone(frozen=True) as frozen:
        for ep in cases:
            frozen.reset();text=ep.text;cut=(len(ep.records)-1)*64
            frozen.run(text[:cut],text[1:cut+1]);x,p=frozen.read();features[0].append(x);online[0].append(int(p[ep.query].argmax()==ep.answer))
            frozen.run(text[cut:cut+64],text[cut+1:cut+65]);x,p=frozen.read();features[1].append(x);online[1].append(int(p[ep.query].argmax()==ep.answer))
            labels.append(ep.states[-1]);queries.append(ep.query)
    return np.asarray(features,dtype=np.float64),np.asarray(labels),np.asarray(queries),np.mean(online,axis=1)

def ridge(train_x,train_y,test_x,test_y,queries,families):
    mean=train_x.mean(0);std=np.maximum(train_x.std(0),1e-6)
    x=(train_x-mean)/std;v=(test_x-mean)/std
    x=np.column_stack([x,np.ones(len(x))]);v=np.column_stack([v,np.ones(len(v))])
    target=np.eye(8)[train_y].reshape(len(x),32)
    penalty=np.eye(x.shape[1]);penalty[-1,-1]=0
    w=np.linalg.solve(x.T@x+penalty,x.T@target)
    pred=(v@w).reshape(len(v),4,8).argmax(-1)
    truth=test_y[np.arange(len(v)),queries]
    correct=pred[np.arange(len(v)),queries]==truth
    return dict(accuracy=float(correct.mean()),
                by_family={family:float(correct[families==family].mean()) for family in ('tracking','composition')},
                all_register_accuracy=float(np.mean(np.all(pred==test_y,axis=1))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--sizes',type=int,nargs='+',default=[1000000,5000000]);p.add_argument('--seeds',type=int,nargs='+',default=[7]);p.add_argument('--wait-for-checkpoints',action='store_true');args=p.parse_args()
    os.sched_setaffinity(0,{sorted(os.sched_getaffinity(0))[1]});rng=random.Random(60019)
    train=[episode(rng,('tracking','composition')[i%2],2+2*(i//2%2),'train') for i in range(1024)]
    audit=suite(19810423,512,'audit');out=dict(train_examples=1024,audit_examples=512,alpha=1.,train_seed=60019,rows=[])
    for size in args.sizes:
      for seed in args.seeds:
       for arm in ('final_only','head_only','state_aux'):
        path=args.checkpoints/f'{size}_{seed}_{arm}.task'
        if args.wait_for_checkpoints:
            deadline=time.monotonic()+240
            while not path.exists() and time.monotonic()<deadline:time.sleep(2)
            if not path.exists():raise RuntimeError(f'checkpoint unavailable: {path.name}')
        if not path.exists():continue
        with Adapter.loads(path.read_bytes()) as m:
            x,y,q,_=extract(m,train);v,vy,vq,online=extract(m,audit)
        row=dict(size=size,seed=seed,arm=arm,online_before_query=float(online[0]),online_after_query=float(online[1]),probes={})
        for stage in (0,1):
            for scale,coords in [('L2',slice(0,64)),('L3',slice(64,128)),('both',slice(None))]:
                key=('before_query' if stage==0 else 'after_query')+'_'+scale
                row['probes'][key]=ridge(x[stage,:,coords],y,v[stage,:,coords],vy,vq,np.array([ep.family for ep in audit]))
        row['feature_std']={k:float(x[s].std(axis=0).mean()) for s,k in [(0,'before_query'),(1,'after_query')]}
        row['mean_energy_fraction']={k:float(np.square(x[s].mean(0)).sum()/np.square(x[s]).sum(1).mean()) for s,k in [(0,'before_query'),(1,'after_query')]}
        out['rows'].append(row);args.out.write_text(json.dumps(out,indent=2)+'\n');print(size,seed,arm,row,flush=True)
if __name__=='__main__':main()
