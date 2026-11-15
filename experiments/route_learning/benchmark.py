"""Isolated serial paired timing; compilation and checkpoint I/O excluded."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from shoe_multiscale import Model
from experiments.scale_100m.train import DocumentChunks,atomic,file_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--targets',type=int,default=200000);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--cpu',type=int,default=2);a=p.parse_args();os.sched_setaffinity(0,{a.cpu})
    meta=json.loads((a.data/'manifest.json').read_text())['files']['train']
    data=np.memmap(a.data/meta['file'],dtype=np.uint8,mode='r');rows=[]
    for repeat in range(a.repeats):
        stream=DocumentChunks(meta['documents'],1007+repeat);assert a.targets<=stream.targets
        for arm in (('C','route') if repeat%2==0 else ('route','C')):
            with Model.load(a.checkpoint) as m:
                m.reset_context(clear_memory=True)
                if arm=='route':m.configure_route_lr()
                done=0;t=time.perf_counter()
                while done<a.targets:
                    x,y,reset=stream.take(done,a.targets-done)
                    if reset:m.reset_context(clear_memory=True)
                    m.run(data[x:y],data[x+1:y+1]);done+=y-x
                m.finish();elapsed=time.perf_counter()-t
            row=dict(arm=arm,repeat=repeat,targets=done,seconds=elapsed,bytes_per_second=done/elapsed)
            rows.append(row);print(json.dumps(row),flush=True)
    means={arm:sum(r['targets'] for r in rows if r['arm']==arm)/sum(r['seconds'] for r in rows if r['arm']==arm) for arm in ('C','route')}
    atomic(a.out,dict(rows=rows,bytes_per_second=means,route_speed_change=means['route']/means['C']-1,
        cpu=a.cpu,checkpoint_sha256=file_hash(a.checkpoint),
        protocol='serial alternating paired order; same source checkpoint; independent reshuffles per pair; includes resets and final retirement'))

if __name__=='__main__':main()
