"""Registered data-only coverage and existing-outcome attribution; no model calls."""
import hashlib,json,sys,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from experiments.record_presence_cost.account_correctness import Counter as Ledger
from experiments.record_presence_cost.stream import advance,new_cursor
w=Path(sys.argv[1]);start=time.process_time();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
class Coverage(Ledger):
 def __init__(self):
  super().__init__();self.window=bytearray();self.position=0;self.positive=Counter();self.contexts=defaultdict(set);self.padded=Counter();self.diverse=Counter()
 def reset(self):super().reset();self.window=bytearray();self.position=0
 def run(self,x,y,mask=None,labels=None):
  for i,byte in enumerate(x):
   self.window.append(byte);self.window=self.window[-8:];self.position+=1
   if labels is None or self.position<8 or self.position%4:continue
   win=bytes(self.window)
   if all(160<=b<256 for b in win):self.diverse[win]+=1
   for ph,packed in enumerate(labels[i,1:5]):
    if packed<0:continue
    key=(int(packed)//256,ph);self.positive[key]+=1;self.contexts[key].add(win);self.padded[key]+=int(9 in win)
  return super().run(x,y,mask,labels)
m=Coverage();cursor,_=advance(m,new_cursor(),2000000)
fit=json.loads((w/'a6/results.json').read_text());assert json.loads(json.dumps(cursor))==fit['rows'][0]['arms']['a6']['cursor']
rows=[dict(key=k,phase=ph,positive_events=m.positive[k,ph],distinct_eight_byte_windows=len(m.contexts[k,ph]),events_containing_padding=m.padded[k,ph]) for k,ph in sorted(m.positive)]
subsets={}
for arm in ('e8','a5','a6'):
 subsets[arm]={}
 for condition in ('pad0','diverse256','compete16','overwrite16'):
  values=[r['arms'][arm]['development'][condition]['autoreg_correct'] for r in fit['rows']]
  subsets[arm][condition]={name:float(np.mean([v[lo:hi] for v in values])) for name,lo,hi in [('first_32_worlds',0,128),('remaining_96_worlds',128,512),('all_128_worlds',0,512)]}
result=dict(source_sha256=sha(Path(__file__)),registration_sha256=sha(Path(__file__).with_name('DIAGNOSTIC_ATTRIBUTION_ADDENDUM.md')),input_sha256=sha(w/'a6/results.json'),
 primary_subsets=subsets,continuation_per_arm=dict(input_bytes_traversed=2000000,record_key_events=sum(m.positive.values()),record_key_events_containing_padding=sum(m.padded.values()),distinct_positive_key_phase_windows=sum(len(x) for x in m.contexts.values()),fully_diverse_negative_window_events=sum(m.diverse.values()),distinct_fully_diverse_negative_windows=len(m.diverse),per_key_phase=rows),
 fitting_bytes=0,inference_bytes=0,cpu_seconds=time.process_time()-start,scope='Existing outcomes and exact data-only traversal. No weights, alternative policy, augmented example, pilot or inference.')
(w/'coverage.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='continuation_per_arm'},indent=2));print({k:v for k,v in result['continuation_per_arm'].items() if k!='per_key_phase'})
