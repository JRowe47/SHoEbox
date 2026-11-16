"""Isolated full-256-byte inference throughput; no Python per-byte loop."""
import argparse
import hashlib
import os
from pathlib import Path
import time
import numpy as np
from .adapter import Adapter,library
from .run import ARMS,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',required=True);a=p.parse_args()
    library();cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    data=b'Observed byte streams carry names, values, and questions through bounded recurrent states.\n'
    rows=[]
    for core in ('minimal','full'):
        for arm in ARMS:
            with Adapter.loads((a.runs/f'{core}_7'/f'{arm}.qr').read_bytes()) as m,m.clone() as f:
                f.benchmark(data,1024);times=[];checks=[]
                for _ in range(3):
                    f.reset();start=time.process_time();checks.append(f.benchmark(data,32768));times.append(time.process_time()-start)
                assert len(set(checks))==1
            rows.append(dict(core=core,arm=arm,seed=7,count=32768,cpu_seconds=times,
                             bytes_per_cpu_second=float(32768/np.median(times)),checksum=checks[0]))
            print(core,arm,rows[-1]['bytes_per_cpu_second'],flush=True)
    write(a.out,dict(cpu=cpu,data_sha256=hashlib.sha256(data).hexdigest(),rows=rows,
        scope='observe plus full 256-byte distribution every byte; one pinned process; 3 repetitions; seed 7; no training or checkpoint I/O'))


if __name__=='__main__':main()
