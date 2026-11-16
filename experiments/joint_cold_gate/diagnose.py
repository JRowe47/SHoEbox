"""Frozen development-only applicability and intervention diagnostics."""
import argparse,hashlib,json,time,ctypes as C
from pathlib import Path
import numpy as np
from .adapter import Adapter
from .tasks import dataset,balanced_offsets
from experiments.joint_request_gate.adapter import Adapter as E7
from experiments.joint_readout.regressions import Corpus
from experiments.query_access.run import evaluate
from experiments.query_readout.run import write

def measure(cls,path,corpus):
 raw=path.read_bytes();start=time.process_time()
 with cls.loads(raw) as model:
  before=model.dumps();r={'checkpoint_sha256':hashlib.sha256(raw).hexdigest(),'language_dev_nll':corpus.evaluate(model,'dev')}
  # Do not call evaluate_bytes here: its nested clone resets interventions.
  with model.clone(intervention='gate_off') as f:
   total=loss=0
   for doc in corpus.docs('dev'):
    f.reset();value=C.c_double();f.call('qr_stream_score',doc,len(doc),C.byref(value));n=len(doc)-1;loss+=value.value*n;total+=n
   r['gate_off_language_dev_nll']=loss/total
  with model.clone() as f:
   values=[]
   for doc in corpus.docs('dev'):
    f.reset()
    for byte in doc[:-1]:f.observe(byte);values.append(f.gate())
  values=np.asarray(values)
  r['generic_gate']={'count':len(values),'mean':float(values.mean()),'min':float(values.min()),'max':float(values.max()),'quantiles':dict(zip(('p50','p90','p95','p99'),map(float,np.quantile(values,[.5,.9,.95,.99])))),'above_0.01':float((values>.01).mean()),'above_0.5':float((values>.5).mean())}
  cases=balanced_offsets(dataset(512,'dev'))
  r['gate_interventions']={k:evaluate(model,cases[:256],k,False)['native_rollout'] for k in ('gate_off','gate_on')}
  with model.clone() as f:
   gates=[]
   for ep in cases:
    f.reset()
    for at,byte in enumerate(ep.data[:-1]):
     f.observe(byte)
     if ep.mask[at+1]:gates.append(f.gate())
  r['answer_gate']={'count':len(gates),'mean':float(np.mean(gates)),'min':float(min(gates)),'max':float(max(gates))}
  assert before==model.dumps();r['source_frozen_exact']=True
 r['diagnostic_cpu_seconds']=time.process_time()-start
 return r

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--e7',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 rows={};corpus=Corpus(a.data)
 for name,cls,folder in [('E7',E7,a.e7),('E8',Adapter,a.runs)]:
  for phase in ('binding','retained'):
   rows[name+'_'+phase]=measure(cls,folder/(phase+'.qr'),corpus)
   write(a.out,{'scope':'Frozen development only; zero fitted updates; audit unopened','rows':rows})
 print({k:(v['language_dev_nll'],v['generic_gate']['mean']) for k,v in rows.items()})
if __name__=='__main__':main()
