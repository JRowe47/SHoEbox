"""Isolated E6/E8 pilot inference against the preserved same-budget native."""
import argparse,hashlib,os,time
from pathlib import Path
import numpy as np
from .adapter import Adapter
from .tasks import dataset,balanced_offsets
from experiments.joint_local_gate.adapter import Adapter as E6
from experiments.joint_readout.adapter import Adapter as E1
from experiments.query_readout.run import write
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
 inputs={'native':(E1,'exp01e-formal/7/native_all.qr'),'E6':(E6,'exp01e-local-gate-pilot/retained.qr'),'E8':(Adapter,'exp01e-request-gate-pilot/retained.qr')};models={};rows={};data=b''.join(e.data for e in balanced_offsets(dataset(256,'dev')))
 try:
  for name,(cls,stem) in inputs.items():
   raw=(a.workspace/stem).read_bytes();models[name]=cls.loads(raw);rows[name]=dict(cpu_seconds=[],inventory=models[name].inventory(),checkpoint_sha256=hashlib.sha256(raw).hexdigest(),checkpoint=stem)
  names=list(inputs)
  for rep in range(3):
   for name in names[rep:]+names[:rep]:
    with models[name].clone() as m:
     m.reset();m.benchmark(data,1024);m.reset();c=time.process_time();m.benchmark(data,65536);rows[name]['cpu_seconds'].append(time.process_time()-c)
  for r in rows.values():r['bytes_per_cpu_second']=65536/float(np.median(r['cpu_seconds']))
  for r in rows.values():r['relative']=r['bytes_per_cpu_second']/rows['native']['bytes_per_cpu_second']
  write(a.out,dict(cpu=cpu,bytes_per_repetition=65536,repetitions=3,stream_sha256=hashlib.sha256(data).hexdigest(),rows=rows,relative=rows['E8']['relative'],scope='Isolated complete 256-byte distribution each input byte; pilot checkpoints, same nominal 500K. Native checkpoint comes from E1 matched-budget native arm.'))
  print({k:r['relative'] for k,r in rows.items()})
 finally:
  for m in models.values():m.close()
if __name__=='__main__':main()
