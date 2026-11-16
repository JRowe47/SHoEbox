"""Read-only development diagnostics; unordered source identity is never assumed."""
import argparse,json
from pathlib import Path
import numpy as np
from .adapter import Adapter
from .tasks import dataset,balanced_offsets
from experiments.factorized_readout.adapter import prefix_labels

def diagnose(m,cases):
 rows=np.zeros((2,6));count=np.zeros(2);fail=[]
 with m.clone() as f:
  for ei,ep in enumerate(cases):
   f.reset();labels=prefix_labels(ep);j=0
   for t,byte in enumerate(ep.data[:-1]):
    if ep.mask[t]:byte=int(f.log_probs().argmax())
    f.observe(byte)
    if ep.mask[t+1]:
     d=f.diagnostic();lab=labels[t];chosen=0 if d[0]>=.5 else 1
     query_ok=d[1]==lab[0];selected_ok=d[2+chosen]==lab[0]
     keys_ok=set(d[2:4])==set(lab[1:3]);pair_ok=all(any(d[2+k]==lab[1+i] and d[4+k]==lab[3+i] for i in range(2)) for k in range(2))
     native=int(f.log_probs().argmax())==ep.data[t+1];value_ok=d[4+chosen]==ep.data[t+1]
     rows[j]+=[query_ok,keys_ok,pair_ok,selected_ok,value_ok,native];count[j]+=1
     if not native:fail.append(dict(case=ei,answer=j,predicted=d.tolist(),labels=lab.tolist()))
     j+=1
 return dict(columns=['query','both_keys','paired_key_values','selected_key','selected_value','native'],by_answer=(rows/count[:,None]).tolist(),failures=fail)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 with Adapter.loads(a.checkpoint.read_bytes()) as m:r=diagnose(m,balanced_offsets(dataset(512,'dev')))
 a.out.write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k!='failures'})
