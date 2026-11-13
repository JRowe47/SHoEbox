"""Matched 1M-parameter smoke study and same-core timings; no checkpoint export."""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
from runtime import fresh_model,core

def evaluate(m,data,intervention=None):
 c=m.evaluation_copy(diagnostic=intervention is not None)
 try:
  if intervention:c.configure_contribution(*intervention)
  return float(c.run(data[:-1],data[1:]))
 finally:c.close()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--tokens',type=int,default=50000);p.add_argument('--output',type=Path,default=ROOT/'results/completion_study.json');p.add_argument('--kernels',nargs='+',choices=['messages','completion'],default=['messages','completion']);a=p.parse_args()
 cpu=sorted(os.sched_getaffinity(0))[-1];os.sched_setaffinity(0,{cpu})
 train=np.fromfile(a.data/'train_segment.txt',np.uint8);held=np.fromfile(a.data/'heldout.bin',np.uint8)
 if a.tokens>=len(train):raise ValueError('Training stream is too short')
 result=dict(tokens=a.tokens,experts=core.E,core=cpu,train_sha256=hashlib.sha256(train).hexdigest(),heldout_sha256=hashlib.sha256(held).hexdigest(),runs=[])
 if a.output.exists() and len(a.kernels)==1:
  previous=json.loads(a.output.read_text());result['runs']=[r for r in previous['runs'] if r['kernel'] not in a.kernels]
 for seed,kernels in ((1,('messages','completion')),(2,('completion','messages'))):
  for kernel in kernels:
   if kernel not in a.kernels:continue
   m=fresh_model(seed,kernel=kernel)
   try:
    # Compile and warm instruction/data paths on an independent model.
    warm=m.snapshot_copy();warm.run(train[:512],train[1:513],mode='periodic',lr=.08,lp=.08/3);warm.close()
    row=dict(seed=seed,kernel=kernel,parameters=m.parameter_count,training=[],seconds=0.)
    for pos in range(0,a.tokens,5000):
     n=min(5000,a.tokens-pos);lr=.08*(.1+.45*(1+math.cos(math.pi*(pos+n/2)/1000000)))
     start=time.perf_counter();loss=m.run(train[pos:pos+n],train[pos+1:pos+n+1],mode='periodic',lr=lr,lp=lr/3,route=.002);seconds=time.perf_counter()-start
     row['seconds']+=seconds;row['training'].append(dict(tokens=pos+n,nll=loss,seconds=seconds))
     print(seed,kernel,pos+n,round(loss,5),flush=True)
    row['bytes_per_second']=a.tokens/row['seconds'];row['validation_nll']=evaluate(m,held[:8193]);row['test_nll']=evaluate(m,held[32768:40961])
    row['composition']=m.composition_metrics();row['completion']=m.completion_metrics();row['teaching']=m.relay_teaching_status();row['lace']=m.lace_metrics()
    row['contribution_delta_nll']={kind+':'+region:evaluate(m,held[32768:40961],(kind,region))-row['test_nll'] for kind,region in [('expert_transform','L3'),('expert_transform','F4'),('expert_transform','G5'),('semantic_payload','S3')]}
    result['runs'].append(row);a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
   finally:m.close()
if __name__=='__main__':main()
