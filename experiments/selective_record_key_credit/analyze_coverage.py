"""Registered data-only coverage and saved-outcome attribution; zero model calls."""
import hashlib,json,sys,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from .credit import Counter as Ledger
from .stream import advance,new_cursor
from .run import sha,write,ROOT
class Coverage(Ledger):
 def __init__(self):
  super().__init__();self.window=bytearray();self.position=0;self.positive=Counter();self.contexts=defaultdict(set);self.padded=Counter();self.diverse=Counter();self.episode=-1;self.blocks={}
 def reset(self):super().reset();self.window=bytearray();self.position=0;self.episode+=1
 def run(self,x,y,mask=None,labels=None,*,allow_record_key_update=True):
  block="diverse" if self.policy in ("a8","a9") and (self.episode//64)%2 else "pad"
  b=self.blocks.setdefault(block,dict(input_bytes=0,positive=Counter(),contexts=defaultdict(set),padded=Counter(),diverse=Counter()))
  b["input_bytes"]+=len(x)
  for i,byte in enumerate(x):
   self.window.append(byte);self.window=self.window[-8:];self.position+=1
   if labels is None or self.position<8 or self.position%4:continue
   win=bytes(self.window)
   if all(160<=b<256 for b in win):self.diverse[win]+=1;b["diverse"][win]+=1
   for ph,packed in enumerate(labels[i,1:5]):
    if packed<0:continue
    key=(int(packed)//256,ph);self.positive[key]+=1;self.contexts[key].add(win);self.padded[key]+=int(9 in win);b["positive"][key]+=1;b["contexts"][key].add(win);b["padded"][key]+=int(9 in win)
  return super().run(x,y,mask,labels,allow_record_key_update=allow_record_key_update)
def main():
 w=Path(sys.argv[1]);start=time.process_time();fit=json.loads((w/'a9/results.json').read_text());coverage={};subsets={}
 for arm in ('e8','a8','a9'):
  m=Coverage();m.policy=arm;cursor,_=advance(m,new_cursor(policy=arm),2000000)
  assert json.loads(json.dumps(cursor))==fit['rows'][0]['arms'][arm]['cursor']
  rows=[dict(key=k,phase=ph,positive_events=m.positive[k,ph],distinct_eight_byte_windows=len(m.contexts[k,ph]),events_containing_padding=m.padded[k,ph]) for k,ph in sorted(m.positive)]
  coverage[arm]=dict(input_bytes_traversed=2000000,record_key_events=sum(m.positive.values()),record_key_events_containing_padding=sum(m.padded.values()),distinct_positive_key_phase_windows=sum(len(x) for x in m.contexts.values()),fully_diverse_negative_window_events=sum(m.diverse.values()),distinct_fully_diverse_negative_windows=len(m.diverse),per_key_phase=rows)
  coverage[arm]["credit_counts"]=dict(m.total)
  coverage[arm]["by_block"]={name:dict(input_bytes=v["input_bytes"],record_key_events=sum(v["positive"].values()),record_key_events_containing_padding=sum(v["padded"].values()),distinct_positive_key_phase_windows=sum(len(x) for x in v["contexts"].values()),fully_diverse_negative_window_events=sum(v["diverse"].values()),distinct_fully_diverse_negative_windows=len(v["diverse"]),per_key_phase=[dict(key=k,phase=ph,positive_events=v["positive"][k,ph],distinct_eight_byte_windows=len(v["contexts"][k,ph]),events_containing_padding=v["padded"][k,ph]) for k,ph in sorted(v["positive"])]) for name,v in m.blocks.items()}
  subsets[arm]={}
  for condition in ('pad0','diverse256','compete16','overwrite16'):
   values=[r['arms'][arm]['development'][condition]['autoreg_correct'] for r in fit['rows']]
   subsets[arm][condition]={name:float(np.mean([v[lo:hi] for v in values])) for name,lo,hi in [('first_32_worlds',0,128),('remaining_96_worlds',128,512),('all_128_worlds',0,512)]}
 result=dict(source_sha256=sha(Path(__file__)),registration_sha256=sha(ROOT/'reports/exp0305a9/DIAGNOSIS_PLAN.md'),input_sha256=sha(w/'a9/results.json'),primary_subsets=subsets,continuation_per_arm=coverage,fitting_bytes=0,inference_bytes=0,cpu_seconds=time.process_time()-start,scope='Saved outcomes and exact data-only traversal. No weights, alternate policy, fit or inference.')
 write(w/'coverage.json',result);print(json.dumps({a:{k:v for k,v in r.items() if k!='per_key_phase'} for a,r in coverage.items()},indent=2))
if __name__=='__main__':main()
