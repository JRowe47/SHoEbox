"""Matched 1M-parameter study; tune on validation, then test two fresh seeds."""
import argparse,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
from runtime import fresh_model,Model,core
def write(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def evaluate(m,data):
 c=m.frozen_copy();loss=0.;n=len(data)-1
 for i in range(0,n,8192):
  z=min(8192,n-i);loss+=z*c.run(data[i:i+z],data[i+1:i+z+1])
 c.close();return loss/n
def worker(a):
 os.sched_setaffinity(0,{a.core});folder=ROOT/'results'/a.out;folder.mkdir(exist_ok=True)
 train=np.fromfile(ROOT/'data/train_segment.txt',np.uint8);held=np.fromfile(ROOT/'data/heldout.bin',np.uint8)
 validation=held[:32769];test=held[32768:]
 m=fresh_model(a.seed,kernel=a.kernel);assert m.parameter_count in (1023744,1031936)
 r=dict(kernel=a.kernel,seed=a.seed,peak_lr=a.peak,parameters=m.parameter_count,tokens=0,training=[],validation=[],train_seconds=0.,status='running',core=a.core,composition_bytes=m.composition_state().size,train_sha256=hashlib.sha256(train).hexdigest(),heldout_sha256=hashlib.sha256(held).hexdigest(),validation_targets=len(validation)-1,test_targets=len(test)-1,selection_seed=3)
 r['validation'].append(dict(tokens=0,nll=evaluate(m,validation)))
 for pos in range(0,a.tokens,5000):
  n=min(5000,a.tokens-pos);lr=a.peak*(.1+.45*(1+math.cos(math.pi*(pos+n/2)/1000000)))
  start=time.perf_counter();loss=m.run(train[pos:pos+n],train[pos+1:pos+n+1],mode='periodic',lr=lr,lp=lr/3,route=.002);sec=time.perf_counter()-start
  r['train_seconds']+=sec;r['tokens']=pos+n;r['training'].append(dict(tokens=pos+n,nll=loss,lr=lr,seconds=sec))
  if (pos+n)%100000==0 or pos+n==a.tokens:
   val=evaluate(m,validation);r['validation'].append(dict(tokens=pos+n,nll=val));write(folder/'training.json',r);print(a.kernel,a.seed,a.peak,pos+n,round(val,6),flush=True)
 if a.tokens==1000000:
  r['test_nll']=evaluate(m,test);r['full_heldout_nll']=evaluate(m,held);r['stats']=m.stats();r['composition_metrics']=m.composition_metrics();r['lace_metrics']=m.lace_metrics();m.save(folder/'last.npz')
 r['semantic_updates']=m.semantic_updates();r['semantic_norm']=float(np.linalg.norm(m.semantic_parameters()));r['relay_teaching']=m.relay_teaching_status();r['status']='complete';write(folder/'training.json',r);m.close()
def launch(jobs,max_workers):
 pending=list(jobs);active=[]
 while pending or active:
  while pending and len(active)<max_workers:
   job=pending.pop(0);log=open(ROOT/'results'/(job['out']+'.log'),'w');cmd=[sys.executable,__file__,'worker']
   for k,v in job.items():cmd+=['--'+k,str(v)]
   proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);active.append((proc,log,job))
  for p,l,j in active[:]:
   code=p.poll()
   if code is not None:
    l.close();active.remove((p,l,j));print(j['out'],code,flush=True)
    if code:raise RuntimeError('Study worker failed: '+j['out'])
  if active:time.sleep(.2)
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['worker','pilots','full']);p.add_argument('--kernel');p.add_argument('--seed',type=int);p.add_argument('--peak',type=float);p.add_argument('--tokens',type=int);p.add_argument('--out');p.add_argument('--core',type=int);a=p.parse_args()
 if a.stage=='worker':worker(a);return
 cpus=sorted(os.sched_getaffinity(0))
 if a.stage=='pilots':
  jobs=[]
  for rate in [.02,.04,.08]:
   for kernel in ['binding','semantics','semantics_fixed']:jobs.append(dict(kernel=kernel,seed=3,peak=rate,tokens=100000,out=f'pilot_{kernel}_{rate}',core=cpus[len(jobs)%3]))
  launch(jobs,3);selection={}
  for kernel in ['binding','semantics','semantics_fixed']:
   rows=[json.loads((ROOT/'results'/j['out']/'training.json').read_text()) for j in jobs if j['kernel']==kernel];best=min(rows,key=lambda r:r['validation'][-1]['nll']);selection[kernel]=dict(peak_lr=best['peak_lr'],validation_nll=best['validation'][-1]['nll'],pilot_seed=3,pilot_targets=100000)
  write(ROOT/'results/rate_selection.json',selection);print(selection,flush=True)
 else:
  selection=json.loads((ROOT/'results/rate_selection.json').read_text());jobs=[]
  for seed in [1,2]:
   for kernel in ['binding','semantics','semantics_fixed']:jobs.append(dict(kernel=kernel,seed=seed,peak=selection[kernel]['peak_lr'],tokens=1000000,out=f'full_{kernel}_seed{seed}',core=cpus[len(jobs)]))
  launch(jobs,3)
if __name__=='__main__':main()
