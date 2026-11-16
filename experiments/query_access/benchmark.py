"""Isolated CPU cost: full native distribution at every observed byte."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from .adapter import Adapter
from .tasks import dataset,balanced_offsets
from .run import ARMS
from experiments.query_readout.run import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cpu=sorted(os.sched_getaffinity(0))[0];os.sched_setaffinity(0,{cpu})
    data=b''.join(e.data for e in balanced_offsets(dataset(256,'dev')));rows={}
    for arm in ARMS:
        with Adapter.loads((a.runs/'7'/f'{arm}.qr').read_bytes()) as m:
            times=[];checks=[]
            for _ in range(3):
                with m.clone() as f:
                    f.reset();start=time.process_time();check=f.benchmark(data,65536);times.append(time.process_time()-start);checks.append(check)
            rows[arm]=dict(cpu_seconds=times,bytes_per_cpu_second=65536/float(np.median(times)),checksums=checks,inventory=m.inventory())
    write(a.out,dict(cpu=cpu,bytes_per_repetition=65536,repetitions=3,seed=7,stream_sha256=hashlib.sha256(data).hexdigest(),prediction='all 256 bytes at every byte, one pass, C++ loop',rows=rows))

if __name__=='__main__':main()
