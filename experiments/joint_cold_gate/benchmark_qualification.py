"""Isolated capacity-matched full-256-byte inference; no fit or selection."""
import argparse,hashlib,os,time
from pathlib import Path
import numpy as np
from .adapter import Adapter
from .tasks import dataset,balanced_offsets
from experiments.joint_readout.adapter import Adapter as Old
from experiments.query_readout.run import write

def measure(runs,retained=False):
 cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu});models={};rows={}
 data=b''.join(e.data for e in balanced_offsets(dataset(256,'dev')))
 try:
  for name,cls in [('native_all',Old),('frozen_d3',Old),('local_gate',Adapter)]:
   raw=(runs/'7'/(name+('.retained' if retained else '')+'.qr')).read_bytes();models[name]=cls.loads(raw);rows[name]=dict(cpu_seconds=[],inventory=models[name].inventory(),checkpoint_sha256=hashlib.sha256(raw).hexdigest())
  names=list(models)
  for repeat in range(3):
   for name in names[repeat:]+names[:repeat]:
    with models[name].clone() as model:
     model.reset();model.benchmark(data,1024);model.reset();start=time.process_time();model.benchmark(data,65536);rows[name]['cpu_seconds'].append(time.process_time()-start)
  for row in rows.values():row['bytes_per_cpu_second']=65536/float(np.median(row['cpu_seconds']))
  for row in rows.values():row['relative']=row['bytes_per_cpu_second']/rows['native_all']['bytes_per_cpu_second']
  return dict(retained=retained,cpu=cpu,bytes_per_repetition=65536,repetitions=3,stream_sha256=hashlib.sha256(data).hexdigest(),rows=rows,relative=rows['local_gate']['relative'],scope='Isolated native C++ full 256-byte output at every input byte; compilation/cloning/warmup excluded.')
 finally:
  for model in models.values():model.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--retained',action='store_true');args=p.parse_args()
 if args.out.exists():raise FileExistsError(args.out)
 result=measure(args.runs,args.retained);write(args.out,result);print(result['relative'])
if __name__=='__main__':main()
