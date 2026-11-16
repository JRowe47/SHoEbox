"""Isolated full-256-output, every-byte inference benchmark, fixed order rotation."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from .adapter import Adapter
from .tasks import dataset,balanced_offsets
from .stage2 import ARMS
from experiments.query_access.adapter import Adapter as Pool
from experiments.query_readout.run import write

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--d1',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 cpu=sorted(os.sched_getaffinity(0))[0];os.sched_setaffinity(0,{cpu});data=b''.join(e.data for e in balanced_offsets(dataset(256,'dev')));models={};rows={}
 try:
  for arm in ARMS:
   cls=Pool if arm=='pool' else Adapter;folder=a.runs if arm=='paired_latest' else a.d1
   models[arm]=cls.loads((folder/'7'/f'{arm}.qr').read_bytes());rows[arm]=dict(cpu_seconds=[],checksums=[],inventory=models[arm].inventory())
  for rep in range(3):
   order=ARMS[rep:]+ARMS[:rep]
   for arm in order:
    with models[arm].clone() as f:
     f.reset();f.benchmark(data,1024);f.reset();start=time.process_time();check=f.benchmark(data,65536);elapsed=time.process_time()-start
    rows[arm]['cpu_seconds'].append(elapsed);rows[arm]['checksums'].append(check)
  for row in rows.values():row['bytes_per_cpu_second']=65536/float(np.median(row['cpu_seconds']))
  ratio=rows['paired_latest']['bytes_per_cpu_second']/rows['native']['bytes_per_cpu_second']
  write(a.out,dict(cpu=cpu,bytes_per_repetition=65536,repetitions=3,seed=7,stream_sha256=hashlib.sha256(data).hexdigest(),prediction='all 256 bytes at every byte, one C++ pass',rows=rows,candidate_relative_throughput=ratio,passes=ratio>=.8))
 finally:
  for m in models.values():m.close()
if __name__=='__main__':main()
