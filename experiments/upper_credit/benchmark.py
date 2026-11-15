"""Serial paired timing in separate processes for distinct experimental ABIs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from shoe_multiscale import Model
from experiments.scale_100m.train import DocumentChunks,atomic

def worker(a):
    os.sched_setaffinity(0,{a.cpu})
    if a.arm.startswith('e'):
        from experiments.upper_credit.specialize import install
    elif a.arm in ('cmean','cnorm'):
        from experiments.upper_credit.centered import install
    else:
        from experiments.upper_credit.screen import install
    install(a.work/'runtime',a.arm)
    meta=json.loads((a.data/'manifest.json').read_text());data=np.fromfile(a.data/'train.bin',np.uint8)
    stream=DocumentChunks(meta['files']['train']['documents'],7)
    with Model.load(a.work/f'{a.arm}_s{a.seed}'/'latest.shoe') as m:
        m.reset_context(clear_memory=True);done=0;start=time.perf_counter()
        while done<a.targets:
            x,y,reset=stream.take(done,a.targets-done)
            if reset:m.reset_context(clear_memory=True)
            m.run(data[x:y],data[x+1:y+1]);done+=y-x
        m.finish();seconds=time.perf_counter()-start
    row=dict(arm=a.arm,seed=a.seed,repeat=a.repeat,targets=done,seconds=seconds,bytes_per_second=done/seconds)
    atomic(a.out,row);print(json.dumps(row),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path);p.add_argument('--baseline',type=Path)
    p.add_argument('--candidate',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--arm',required=True)
    p.add_argument('--cpu',type=int,default=0);p.add_argument('--targets',type=int,default=200000)
    p.add_argument('--seed',type=int,default=7);p.add_argument('--repeat',type=int,default=0);p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if a.worker:worker(a);return
    rows=[];a.out.parent.mkdir(parents=True,exist_ok=True)
    for seed in (1,7,19):
        for repeat in range(3):
            order=('C',a.arm) if repeat%2==0 else (a.arm,'C')
            for arm in order:
                out=a.out.parent/f'timing_{arm}_{seed}_{repeat}.json';work=a.baseline if arm=='C' else a.candidate
                subprocess.run([sys.executable,__file__,'--worker','--work',str(work),'--data',str(a.data),'--out',str(out),
                    '--arm',arm,'--seed',str(seed),'--repeat',str(repeat),'--cpu',str(a.cpu),'--targets',str(a.targets)],check=True)
                rows.append(json.loads(out.read_text()))
    atomic(a.out,dict(cpu=a.cpu,targets=a.targets,rows=rows,includes='training, document context resets and final pending-credit retirement; excludes process startup, compilation, load and checkpoint writes',execution='one process at a time; alternating paired order'))
if __name__=='__main__':main()
