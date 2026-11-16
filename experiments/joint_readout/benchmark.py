"""Isolated every-byte, full-256-output throughput after all other jobs stop."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from .adapter import Adapter
from .tasks import dataset,balanced_offsets
from experiments.query_readout.run import write

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
 data=b''.join(e.data for e in balanced_offsets(dataset(256,'dev')));arms=['native_all','frozen_d3','joint_d3','gated_all','gated_no_local'];arms=[n for n in arms if (a.runs/'7'/f'{n}.qr').exists()];models={};rows={}
 try:
  for arm in arms:models[arm]=Adapter.loads((a.runs/'7'/f'{arm}.qr').read_bytes());rows[arm]=dict(cpu_seconds=[],checksums=[],inventory=models[arm].inventory())
  for rep in range(3):
   for arm in arms[rep:]+arms[:rep]:
    with models[arm].clone() as f:
     f.reset();f.benchmark(data,1024);f.reset();c=time.process_time();check=f.benchmark(data,65536);elapsed=time.process_time()-c
    rows[arm]['cpu_seconds'].append(elapsed);rows[arm]['checksums'].append(check)
  for r in rows.values():r['bytes_per_cpu_second']=65536/float(np.median(r['cpu_seconds']))
  relative=rows['gated_all']['bytes_per_cpu_second']/rows['native_all']['bytes_per_cpu_second']
  write(a.out,dict(cpu=cpu,bytes_per_repetition=65536,repetitions=3,seed=7,stream_sha256=hashlib.sha256(data).hexdigest(),rows=rows,relative=relative,passes=relative>=.8,prediction='all 256 bytes at each byte; one native pass'))
 finally:
  for m in models.values():m.close()
if __name__=='__main__':main()
