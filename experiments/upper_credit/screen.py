"""Language-only upper-scale credit: matched corrected-C screening harness."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.plateau_100m import common
from experiments.routing_5m.run import configure
from experiments.scale_100m.train import DocumentChunks,frozen_score,atomic,file_hash
from shoe_multiscale import Model

ARMS={'C':(1,1), 'g4_4':(4,4), 'g4_16':(4,16), 'g16_16':(16,16), 'g16_64':(16,64)}
RECALL={'r1':(1,1),'r4':(4,4),'r4_16':(4,16),'gr1':(1,1)}
ARMS.update({key:(4,16) if key=='gr1' else (1,1) for key in RECALL})

def build(directory):
    common.build(directory)

def install(directory,arm):
    return common.install(directory)

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--arm',choices=ARMS,default='C')
    p.add_argument('--seed',type=int,default=7);p.add_argument('--cpu',type=int,default=0)
    p.add_argument('--targets',type=int,default=10_000_000);p.add_argument('--build',action='store_true')
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    a.work.mkdir(parents=True,exist_ok=True)
    if a.build:build(a.work/'runtime');return
    os.sched_setaffinity(0,{a.cpu});install(a.work/'runtime',a.arm)
    out=a.work/f'{a.arm}_s{a.seed}';out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((a.data/'manifest.json').read_text())
    data=np.fromfile(a.data/'train.bin',np.uint8);dev=np.fromfile(a.data/'development.bin',np.uint8)
    docs=meta['files']['train']['documents'];devdocs=meta['files']['development']['documents']
    stream=DocumentChunks(docs,a.seed);assert a.targets<=stream.targets
    segment_hash=hashlib.sha256(stream.segments.tobytes()).hexdigest()
    if a.resume:
        r=json.loads((out/'results.json').read_text());m=Model.load(out/'latest.shoe');done=r['targets']
        assert r['segment_sha256']==segment_hash and r['arm']==a.arm
    else:
        m=Model(parameters=5_000_000,seed=a.seed,routing_arm='C',balance_policy='warmup',
                language_credit=(1,1,*ARMS[a.arm]),recall_credit=(0,0,*RECALL.get(a.arm,(0,0))))
        done=0;r=dict(arm=a.arm,seed=a.seed,parameters=m.parameter_count,experts=m.stats()['experts'],
            language_gains=list(ARMS[a.arm]),segment_sha256=segment_hash,
            train_sha256=meta['files']['train']['sha256'],development_sha256=meta['files']['development']['sha256'],
            curve=[],targets=0,train_seconds=0.,evaluation_seconds=0.,checkpoint_seconds=0.)
    weighted=0.;window=0
    def record():
        nonlocal weighted,window
        t=time.perf_counter();nll,details=frozen_score(m,dev,devdocs);r['evaluation_seconds']+=time.perf_counter()-t
        r['curve'].append(dict(targets=done,nll=nll,train_seconds=r['train_seconds'],train_nll=weighted/window if window else None))
        r.update(targets=done,development=details)
        t=time.perf_counter();m.save(out/'latest.shoe');r['checkpoint_seconds']+=time.perf_counter()-t
        atomic(out/'results.json',r);print(json.dumps(dict(arm=a.arm,seed=a.seed,targets=done,nll=nll,train_seconds=r['train_seconds'])),flush=True)
        weighted=0.;window=0
    if not a.resume:record()
    while done<a.targets:
        boundary=min(a.targets,(done//1_000_000+1)*1_000_000)
        x,y,reset=stream.take(done,boundary-done);t=time.perf_counter();configure(m,done,'C')
        if reset:m.reset_context(clear_memory=True)
        loss=m.run(data[x:y],data[x+1:y+1]);r['train_seconds']+=time.perf_counter()-t
        done+=y-x;weighted+=loss*(y-x);window+=y-x
        if done==boundary:record()
    r['checkpoint_sha256']=file_hash(out/'latest.shoe');r['status']='complete';atomic(out/'results.json',r);m.close()

if __name__=='__main__':main()
