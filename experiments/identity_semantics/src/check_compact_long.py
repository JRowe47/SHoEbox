"""8K byte worker/chunk/restart equality measured from current memory only."""
import json,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from runtime import fresh_model,Model,core
from memory_inspection import capture_memory,compare_memory,read_memory_index,memory_fingerprint
raw=b'The river runs through the valley. The king crossed the bridge. '
x=np.frombuffer((raw*140)[:8193],np.uint8);models=[];rows=[]
try:
 for workers in (1,2):
  m=fresh_model(seed=7,profile='pipeline',pipeline_workers=workers);models.append(m);parts=[]
  with tempfile.TemporaryDirectory() as tmp:
   cuts=[0,8192] if workers==1 else [0,1,97,2049,4096,8192]
   for lo,hi in zip(cuts,cuts[1:]):
    loss,values=m.run(x[lo:hi],x[lo+1:hi+1],mode='periodic',lr=.04,lp=.04/3,logits=True);parts.append(values)
    if workers==2 and hi==2049:
     path=Path(tmp)/'pending.npz';m.save(path);m.close();m=Model.load(path);models.append(m);m.configure_pipeline_workers(2)
  values=np.concatenate(parts);snapshot=capture_memory(m);inventory=read_memory_index(m)
  assert sum(v['event_ordinal'] for v in inventory.values())>sum(v['constituents'] for v in inventory.values())
  assert all(v['invalid_retained_spans']==v['missing_content_records']==0 for v in inventory.values())
  if workers==1:expected=values;reference=snapshot
  else:np.testing.assert_array_equal(expected,values);assert compare_memory(reference,snapshot)['equal']
  rows.append(dict(workers=workers,memory_fingerprint=memory_fingerprint(snapshot),memory=m.inspect_pipeline(),live=inventory))
 output=ROOT/'results'/f'compact_long_e{core.E}.json';output.parent.mkdir(exist_ok=True)
 output.write_text(json.dumps(dict(targets=8192,bitwise_workers=True,bitwise_chunks=True,bitwise_restart=True,runs=rows),indent=2)+'\n')
 print('8,192-byte worker/chunk/restart equality passed')
finally:
 for m in models:m.close()
