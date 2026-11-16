"""Isolated timing of retained pilot policies; no promotion from pilot timing."""
import argparse,hashlib,importlib,os,time
from pathlib import Path
import numpy as np
from .adapter import Adapter
from experiments.joint_switch.tasks import dataset,balanced_offsets
from experiments.query_readout.run import write

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu});data=b''.join(e.data for e in balanced_offsets(dataset(256,'dev')))
 specs={'native':('joint_readout','exp01e-formal/7/native_all.qr'),'E2':('joint_switch','exp01e-switch-pilot/retained.qr'),'E3':('joint_mixture','exp01e-mixture-pilot/retained.qr'),'E4':('joint_calibrated','exp01e-calibrated-pilot/retained.qr'),'E5':('joint_evidence_gate','exp01e-evidence-gate-pilot/retained.qr')};models={};rows={}
 try:
  for k,(module,file) in specs.items():
   cls=importlib.import_module('experiments.'+module+'.adapter').Adapter;models[k]=cls.loads((a.root/file).read_bytes());rows[k]=dict(cpu_seconds=[],inventory=models[k].inventory())
  names=list(specs)
  for rep in range(3):
   for k in names[rep:]+names[:rep]:
    with models[k].clone() as f:
     f.reset();f.benchmark(data,1024);f.reset();c=time.process_time();f.benchmark(data,65536);elapsed=time.process_time()-c
    rows[k]['cpu_seconds'].append(elapsed)
  for r in rows.values():r['bytes_per_cpu_second']=65536/float(np.median(r['cpu_seconds']))
  for r in rows.values():r['relative']=r['bytes_per_cpu_second']/rows['native']['bytes_per_cpu_second']
  write(a.out,dict(cpu=cpu,bytes_per_repetition=65536,repetitions=3,stream_sha256=hashlib.sha256(data).hexdigest(),rows=rows,scope='pilot inference only; fixed complete 256-byte distribution every input byte'))
 finally:
  for m in models.values():m.close()
if __name__=='__main__':main()
