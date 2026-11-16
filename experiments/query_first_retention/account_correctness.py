"""Replay exact discarded Python-test inputs WITHOUT computing/fitting weights."""
import argparse,json
from pathlib import Path
import numpy as np
from .tasks import cases
from .stream import new_cursor,advance
from experiments.joint_cold_gate.adapter import prefix_labels
class Counter:
 def __init__(self):
  self.n=0;self.total=dict(input_bytes=0,padding_bytes=0,targets=0,local_projection_steps=0,local_presence_steps=0,gate_targets=0,local_training_windows=0)
 def reset(self):self.n=0
 def finish(self):pass
 def inventory(self):return {'bytes':self.n}
 def run(self,x,y,mask=None,labels=None):
  t=self.total;t['input_bytes']+=len(x);t['padding_bytes']+=x.count(9);targets=len(x) if mask is None else sum(mask);t['targets']+=targets
  for i in range(len(x)):
   self.n+=1
   if labels is None:continue
   lab=labels[i]
   if self.n>=2:
    t['gate_targets']+=1;t['local_presence_steps']+=1;t['local_training_windows']+=1
    if lab[0]>=0:t['local_projection_steps']+=16
   if self.n>=8 and self.n%4==0:
    t['local_presence_steps']+=4;t['local_training_windows']+=1;t['local_projection_steps']+=24*int(sum(lab[1:5]>=0))
  return 0,targets

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
 for name,policies in [('prefit_python',1),('prefit_python_final',1),('prefit_python_corrected',3)]:
  m=Counter();ep=cases('train',1)[0].ep
  for _ in range(4):m.reset();m.run(ep.data[:-1],ep.data[1:],labels=prefix_labels(ep))
  for _ in range(policies):
   m.reset();c=new_cursor();c['finish_interval']=997;c['next_finish']=997;s,_=advance(m,c,1001)
   n=m.n;advance(m,s,1000);m.n=n;t,_=advance(m,s,371);advance(m,t,629)
  for c in cases('train',4,condition='compete16'):
   m.reset();m.run(c.ep.data[:-1],c.ep.data[1:],labels=prefix_labels(c.ep))
  rows.append(dict(process=name,counts=m.total))
 totals={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']}
 a.out.write_text(json.dumps(dict(rows=rows,totals=totals,method='Exact data/cursor-only replay of all three actual Python pre-fit invocations; zero fitted computation. Historical unit/native microtrajectories are separately CPU-metered.'),indent=2)+'\n')
if __name__=='__main__':main()
