"""Frozen survivor-vs-filler attribution, registered before this read-only run."""
import json,sys,time
from pathlib import Path
import numpy as np
from experiments.record_presence_cost.backend import Model
from experiments.record_presence_cost.tasks import cases
from experiments.record_presence_cost.run import sha,write
from experiments.joint_cold_gate.adapter import prefix_labels
w=Path(sys.argv[1]);runs=w/'a6';out=w/'survivors.json';assert not out.exists()
hashes={str(p.relative_to(runs)):sha(p) for p in runs.glob('*/a[56].qr')}
write(w/'survivors.lock.json',dict(source_sha256=sha(Path(__file__)),checkpoint_hashes=hashes,audit_opened=False,fitting_bytes=0))
rows=[];total=pad=0;start=time.process_time()
for seed in (1,7,19):
 for arm in ('a5','a6'):
  records=[]
  with Model.loads((runs/str(seed)/(arm+'.qr')).read_bytes(),arm) as m,m.clone() as f:
   for index,c in enumerate(cases('dev',32,condition='diverse256')):
    f.reset();known={};labels=prefix_labels(c.ep);answer=0
    for t,b in enumerate(c.ep.data[:-1]):
     f.observe(b);total+=1;pad+=int(b==9)
     for ph,packed in enumerate(labels[t,1:5]):
      if packed>=0:known[(int(packed)//256,int(packed)%256,ph)]=f.state()[2].copy()
     if not c.ep.mask[t+1]:continue
     q,target=c.query_keys[answer],c.expected[answer];n,_,_,units=f.state();selected,decoded=f.decoded();d=f.diagnostic();present=recognized=shortlisted=False;unannotated=0;owners={}
     for slot in range(n):
      for ph in range(4):
       own=[(key,val) for (key,val,phase),sig in known.items() if ph==phase and np.array_equal(sig,units[slot,1:])];owners[slot,ph]=own
       if decoded[slot,ph,0]>0 and not own:unannotated+=1
       if (q,target) in own:
        present=True;recognized|=bool(decoded[slot,ph,0]>0);shortlisted|=any((slot,ph)==tuple(s) for s in selected)
     mass={'target':0.,'other_original':0.,'unannotated':0.}
     for j,pair in enumerate(selected):
      own=owners.get(tuple(pair),[]);kind='target' if (q,target) in own else 'other_original' if own else 'unannotated';mass[kind]+=float(d[0]) if j==0 else 1-float(d[0])
     records.append(dict(presentation=index,answer=answer,present=present,recognized=recognized,shortlisted=shortlisted,correct=int(f.log_probs().argmax())==target,unannotated_recognized=unannotated,mass=mass));answer+=1
  alive=[r for r in records if r['present']];bad=[r for r in alive if not r['correct']]
  summary=dict(answers=len(records),present=np.mean([r['present'] for r in records]),correct=np.mean([r['correct'] for r in records]),survivor_accuracy=np.mean([r['correct'] for r in alive]),survivor_shortlisted=np.mean([r['shortlisted'] for r in alive]),survivor_recognized=np.mean([r['recognized'] for r in alive]),survivor_false_negative=np.mean([not r['recognized'] for r in alive]),unannotated_recognized=np.mean([r['unannotated_recognized'] for r in records]),unannotated_mass=np.mean([r['mass']['unannotated'] for r in records]),incorrect_survivor_unannotated_mass=np.mean([r['mass']['unannotated'] for r in bad]))
  summary={k:float(v) if isinstance(v,np.floating) else v for k,v in summary.items()};rows.append(dict(seed=seed,arm=arm,summary=summary,records=records));print(seed,arm,summary,flush=True)
  write(out,dict(rows=rows,inference_bytes=total,padding_bytes=pad,cpu_seconds=time.process_time()-start,fitting_bytes=0,audit_opened=False))
assert hashes=={str(p.relative_to(runs)):sha(p) for p in runs.glob('*/a[56].qr')}
