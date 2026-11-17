"""Data-only replay of local event counts; never computes or updates model weights."""
import argparse,json
from pathlib import Path
import numpy as np
from .tasks import cases
from .stream import new_cursor,advance
from experiments.joint_cold_gate.adapter import prefix_labels
class Counter:
 def __init__(self):
  self.n=0;self.total=dict(input_bytes=0,padding_bytes=0,targets=0,local_projection_steps=0,local_presence_steps=0,gate_targets=0,local_training_windows=0,positive_query_keys=0,positive_record_keys=0)
 def reset(self):self.n=0
 def finish(self):pass
 def inventory(self):return {'bytes':self.n}
 def run(self,x,y,mask=None,labels=None):
  t=self.total;t['input_bytes']+=len(x);t['padding_bytes']+=x.count(9);targets=len(x) if mask is None else int(sum(mask));t['targets']+=targets
  for i in range(len(x)):
   self.n+=1
   if labels is None:continue
   lab=labels[i]
   if self.n>=2:
    t['gate_targets']+=1;t['local_presence_steps']+=1;t['local_training_windows']+=1
    if lab[0]>=0:t['local_projection_steps']+=16;t['positive_query_keys']+=1
   if self.n>=8 and self.n%4==0:
    positive=int(sum(lab[1:5]>=0));t['local_presence_steps']+=4;t['local_training_windows']+=1;t['local_projection_steps']+=24*positive;t['positive_record_keys']+=positive
  return 0,targets

def augmented(t,policy):
 t=dict(t);events=t['positive_query_keys']+t['positive_record_keys'] if policy=='a5' else 0
 t.update(hybrid_attraction_components=events,hybrid_discrimination_components=events,hybrid_gradient_vectors=events,hybrid_gradient_rows=16*events,hybrid_code_comparisons=256*events,hybrid_coordinate_distance_comparisons=4096*events)
 return t

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workspace',type=Path);a=p.parse_args();rows=[]
 # Exact successful 11-test invocation. Controls: E8/A4 against old source.
 for policy in ('e8','a4','a5'):
  m=Counter();ep=cases('train',1)[0].ep
  if policy!='a5':
   for _ in range(2):m.reset();m.run(ep.data[:-1],ep.data[1:],labels=prefix_labels(ep))
  m.reset();c=new_cursor();c['finish_interval']=997;c['next_finish']=997;s,_=advance(m,c,1001)
  n=m.n;advance(m,s,1000);m.n=n;t,_=advance(m,s,371);advance(m,t,629)
  if policy=='a5':
   for c in cases('train',4,condition='compete16'):
    m.reset();m.run(c.ep.data[:-1],c.ep.data[1:],labels=prefix_labels(c.ep))
   m.reset();d=b'Generic byte objectives stay separate.';m.run(d,d)
   for c in cases('train',2,condition='overwrite16'):
    m.reset();m.run(c.ep.data[:-1],c.ep.data[1:],labels=prefix_labels(c.ep))
  if policy in ('a4','a5'):
   d=bytes(range(256))*2;m.run(d[:-1],d[1:])
  rows.append(dict(policy=policy,counts=augmented(m.total,policy)))
 totals={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']}
 result=dict(rows=rows,totals=totals,method='Exact data-only replay of successful 11-test Python invocation. Inference and independent objective checks separately counted; no fitted computation in replay.',
  objective_microchecks=dict(python_native_calls=256,python_independent_calls=256,code_comparisons=(256+256)*256,native_calls_per_build=1029,native_independent_reference_calls_per_build=768,native_code_comparisons_per_build=(1029+768)*256,native_finite_difference_coordinates_per_build=256,native_actual_affine_gradient_rows_per_build=4176,native_expected_affine_gradient_rows_per_build=4096,builds=2),
  frozen_inference_bytes=1024+sum(1 for _ in cases('train',1,condition='compete16')[0].ep.data[:-1])+3,
  historical_microtrajectories='Historical/native tests are fully process-metered; historical scalar update counts remain uninstrumented. A5 inherited numeric/request/admission checks use zero positive key events.')
 if a.workspace:
  research=[];fits=json.loads((a.workspace/'a5/results.json').read_text())
  for warmup in (True,False):
   m=Counter();cursor,_=advance(m,new_cursor(warmup,not warmup),250000 if warmup else 2000000)
   for seedrow in fits['rows']:
    for arm,fit in seedrow['arms'].items():
     fit=fit['warmup'] if warmup else fit
     assert json.loads(json.dumps(cursor))==fit['cursor']
     for k in ('local_projection_steps','local_presence_steps','gate_targets','local_training_windows'):
      assert m.total[k]==fit['inventory'][k]-fit['initial_inventory'][k]
     research.append(dict(seed=seedrow['seed'],arm=arm,phase='warmup' if warmup else 'continuation',counts=augmented(m.total,arm)))
  result['research']=research;result['research_totals']={k:sum(r['counts'][k] for r in research) for k in research[0]['counts']}
  resume=[]
  for path in sorted((a.workspace/'a5').glob('*/*.qr')):
   cursor=json.loads(path.with_suffix('.cursor.json').read_text());m=Counter();m.n=cursor['model_bytes'];advance(m,cursor,1000)
   resume.append(dict(file=str(path.relative_to(a.workspace/'a5')),counts={k:2*v for k,v in augmented(m.total,path.name.split('.')[0]).items()}))
  result['resume']=resume;result['resume_totals']={k:sum(r['counts'][k] for r in resume) for k in resume[0]['counts']}
 a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(prefit=totals,research=result.get('research_totals'),resume=result.get('resume_totals')),indent=2))
if __name__=='__main__':main()
