import json,os,time
from pathlib import Path
import numpy as np
import model
from pipeline import SnapshotPipeline,process_snapshot
ROOT=Path(__file__).resolve().parent;model.KERNEL='cached';os.sched_setaffinity(0,{0})
size='1m' if model.core.E==12 else '5m';source=ROOT.parent/'utility_probe/results'/size/'seed1/center_split/last.npz'
data=np.frombuffer((ROOT.parent/'scale_probe/data/fresh.bin').read_bytes(),np.uint8)[2000000:]
validation=np.frombuffer((ROOT.parent/'scale_probe/data/validation.bin').read_bytes(),np.uint8)
result={};total=400000;interval=100000
for policy in ['synchronous','overlapped']:
 m=model.Model.load(source);directory=ROOT/'results/overlap'/size/policy;directory.mkdir(parents=True,exist_ok=True)
 pipeline=SnapshotPipeline(validation,directory,2) if policy=='overlapped' else None
 rows=[];snapshot_seconds=0.;training_seconds=0.;start=time.perf_counter()
 try:
  for pos in range(0,total,interval):
   t=time.perf_counter();loss=m.run(data[pos:pos+interval],data[pos+1:pos+interval+1],mode='periodic',lr=.008,lp=.008/3,route=.002);training_seconds+=time.perf_counter()-t
   if pipeline:pipeline.submit(m,pos+interval)
   else:
    t=time.perf_counter();snapshot=m.snapshot_copy();snapshot_seconds+=time.perf_counter()-t
    rows.append(process_snapshot(snapshot,pos+interval,validation,directory/f'at_{pos+interval}.npz',None))
   print(size,policy,pos+interval,loss,flush=True)
 finally:
  if pipeline:rows=pipeline.close();snapshot_seconds=pipeline.snapshot_seconds
 seconds=time.perf_counter()-start
 assert os.sched_getaffinity(0)=={0}
 result[policy]=dict(seconds=seconds,training_seconds=training_seconds,training_bytes_per_wall_second=total/seconds,snapshot_seconds=snapshot_seconds,wait_seconds=pipeline.wait_seconds if pipeline else 0.,records=rows,final_fingerprint=m.fingerprint())
 m.close()
assert result['synchronous']['final_fingerprint']==result['overlapped']['final_fingerprint']
assert [r['validation_nll'] for r in result['synchronous']['records']]==[r['validation_nll'] for r in result['overlapped']['records']]
assert [r['snapshot_fingerprint'] for r in result['synchronous']['records']]==[r['snapshot_fingerprint'] for r in result['overlapped']['records']]
out=dict(size=size,targets=total,interval=interval,validation_targets=len(validation)-1,exact=True,results=result,speedup=result['synchronous']['seconds']/result['overlapped']['seconds'])
(ROOT/'results'/f'overlap_{size}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
