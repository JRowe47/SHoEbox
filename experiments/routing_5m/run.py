"""Paired 5M-parameter routing ablations on document-contained FineWeb bytes."""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.plateau_100m import common
from experiments.scale_100m.train import DocumentChunks, frozen_score, atomic, file_hash
from multiscale import model as runtime
from shoe_multiscale import Model

ARMS = {
    'D': ('D', 1, 1., 3),
    'C': ('C', 1, 1., 3),
    'B': ('B', 1, 1., 3),
    'A': ('A', 1, 1., 3),
    'D_no_balance': ('D', 0, 1., 3),
    'D_warmup_only': ('D', 1, 0., 3),
    'C_warmup_only': ('C', 1, 0., 3),
    'D_L1_only': ('D', 1, 1., 1),
    'D_high_rate': ('D', 1, 1., 3),
}

def build(path):
    common.build(path)
    path = Path(path)
    p = path/'native.cpp'
    source = p.read_text().replace('int diagnostic_balance_update=1;',
        'int diagnostic_balance_update=1;\nfloat diagnostic_starvation=1.f;\nint diagnostic_predictive_max=3;')
    assert source.count('routing_arm==3&&scale>0') == 5
    source = source.replace('routing_arm==3&&scale>0',
                            'routing_arm==3&&scale>0&&scale<=diagnostic_predictive_max')
    source += '\nextern "C" int screen_policy(float starve,int maximum){shoe::diagnostic_starvation=starve;shoe::diagnostic_predictive_max=maximum;return 0;}\n'
    p.write_text(source)
    p = path/'routing_geometry_impl.hpp'
    source = p.read_text().replace('routing_arm==3&&scale>0',
        'routing_arm==3&&scale>0&&scale<=diagnostic_predictive_max')
    assert source.count('.002f*(1-warm)*starvation') == 1
    source = source.replace('.002f*(1-warm)*starvation',
                            'diagnostic_starvation*.002f*(1-warm)*starvation')
    p.write_text(source)
    subprocess.run(['g++','-std=c++17','-O3','-march=native','-ffp-contract=off',
        '-fPIC','-shared',str(path/'native.cpp'),'-o',str(path/'diagnostic.so')],check=True)

def install(path, arm):
    base, lib = common.install(path)
    lib.screen_policy.argtypes = [C.c_float,C.c_int]
    lib.screen_policy.restype = C.c_int
    _, balance, starvation, maximum = ARMS[arm]
    lib.diag_policy(balance, .1)
    lib.screen_policy(starvation, maximum)
    return base, lib

def configure(model, done, arm):
    p = done/5_000_000
    scale = max(.01,p/.01) if p<.01 else .1+.45*(1+math.cos(math.pi*min(1,(p-.01)/.99)))
    if done>=5_000_000 and arm!='D_high_rate':
        scale *= .25
    model.configure(fast_rate=.008*scale, slow_rate=.08*scale, query_rate=.008*scale,
        head_rate=.16*scale, route_rate=.002, auxiliary_weight=.1,
        reconstruction_weight=.05, constituent_credit=1.)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,default='D')
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--cpu',type=int,default=0)
    p.add_argument('--targets',type=int,default=10_000_000)
    p.add_argument('--build',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.work.mkdir(parents=True,exist_ok=True)
    if a.build:
        build(a.work/'runtime');return
    os.sched_setaffinity(0,{a.cpu});install(a.work/'runtime',a.arm)
    out=a.work/f'{a.arm}_s{a.seed}';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.data/'manifest.json').read_text())
    docs=manifest['files']['train']['documents'];devdocs=manifest['files']['development']['documents']
    data=np.memmap(a.data/'train.bin',mode='r',dtype=np.uint8)
    dev=np.memmap(a.data/'development.bin',mode='r',dtype=np.uint8)
    stream=DocumentChunks(docs,a.seed)
    assert a.targets<=stream.targets
    segment_hash=hashlib.sha256(stream.segments.tobytes()).hexdigest()
    if a.resume:
        r=json.loads((out/'results.json').read_text());m=Model.load(out/'latest.shoe')
        assert r['segment_sha256']==segment_hash and r['arm']==a.arm
        done=r['targets']
    else:
        m=Model(parameters=5_000_000,seed=a.seed,routing_arm=ARMS[a.arm][0],
                semantic_laces=False,router='scan')
        done=0
        r=dict(arm=a.arm,seed=a.seed,parameters=m.parameter_count,experts=m.stats()['experts'],
            segment_sha256=segment_hash,train_sha256=manifest['files']['train']['sha256'],
            development_sha256=manifest['files']['development']['sha256'],
            curve=[],train_seconds=0.,evaluation_seconds=0.,checkpoint_seconds=0.,wall_seconds=0.,
            policy=ARMS[a.arm],targets=0)
    started=time.perf_counter();previous_wall=r['wall_seconds'];weighted=0.;window=0

    def record():
        nonlocal weighted,window
        t=time.perf_counter();nll,details=frozen_score(m,dev,devdocs);r['evaluation_seconds']+=time.perf_counter()-t
        r['curve'].append(dict(targets=done,nll=nll,train_seconds=r['train_seconds'],
            train_nll=weighted/window if window else None,routing=m.routing_stats(),rates=m.stats()['learning_rates']))
        r.update(targets=done,development=details,wall_seconds=previous_wall+time.perf_counter()-started)
        t=time.perf_counter();m.save(out/'latest.shoe');r['checkpoint_seconds']+=time.perf_counter()-t
        if done==5_000_000:m.save(out/'5m.shoe')
        atomic(out/'results.json',r)
        print(json.dumps(dict(arm=a.arm,seed=a.seed,targets=done,nll=nll,train_seconds=r['train_seconds'])),flush=True)
        weighted=0.;window=0

    if not a.resume:record()
    while done<a.targets:
        boundary=min(a.targets,(done//1_000_000+1)*1_000_000)
        x,y,reset=stream.take(done,boundary-done)
        t=time.perf_counter();configure(m,done,a.arm)
        if reset:m.reset_context(clear_memory=True)
        loss=m.run(data[x:y],data[x+1:y+1]);r['train_seconds']+=time.perf_counter()-t
        done+=y-x;weighted+=loss*(y-x);window+=y-x
        if done==boundary:record()
    r['checkpoint_sha256']=file_hash(out/'latest.shoe');r['status']='complete';atomic(out/'results.json',r)
    m.close()

if __name__=='__main__':main()
