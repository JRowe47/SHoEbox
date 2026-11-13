"""Independent interventions and alternating same-core timing for the study."""
import argparse,json,os,statistics,sys,time
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
from runtime import Model,core
from run_semantics_study import write
TEMP=ROOT/'results/message_tmp'
REGIONS=('L3','E4','F4','V4','G5')
KINDS=('neural_output','expert_transform','semantic_residual','relay_context','whole_source','semantic_payload','feedback_channel')
def data_eval(m,x):
 n=len(x)-1;total=0
 for pos in range(0,n,8192):
  length=min(8192,n-pos);total+=length*m.run(x[pos:pos+length],x[pos+1:pos+length+1])
 return total/n

def interventions(seed_filter=None,kernel_filter=None):
 data=np.fromfile(ROOT/'data/heldout.bin',np.uint8)[32768:];rows=[];parity=[]
 for seed in ((seed_filter,) if seed_filter else (1,2)):
  for kernel in ((kernel_filter,) if kernel_filter else ('relays','messages')):
   path=TEMP/f'full_{kernel}_seed{seed}/last.npz';m=Model.load(path)
   base=m.evaluation_copy(diagnostic=True);frozen=m.evaluation_copy();base_params=base.flat.copy();base_sem=base.semantic_parameters()
   baseline=data_eval(base,data);prod=data_eval(frozen,data);assert baseline==prod
   np.testing.assert_array_equal(base.flat,base_params);np.testing.assert_array_equal(base.semantic_parameters(),base_sem)
   signature=[v['event_hash'] for v in base.contribution_metrics().values()];parity.append(dict(kernel=kernel,seed=seed,nll=baseline,exact=True))
   base.close();frozen.close()
   for kind in KINDS:
    if kernel=='relays' and kind in ('semantic_payload','feedback_channel'):continue
    for region in REGIONS:
     c=m.evaluation_copy(diagnostic=True);c.configure_contribution(kind,region);nll=data_eval(c,data)
     metrics=c.contribution_metrics();same=[v['event_hash'] for v in metrics.values()]==signature
     if kind!='whole_source':assert same,(kernel,seed,kind,region)
     rows.append(dict(kernel=kernel,seed=seed,kind=kind,region=region,nll=nll,delta=nll-baseline,events_preserved=same,applications=metrics[region]['intervention_applications']))
     c.close()
   m.close();write((TEMP/f'interventions_{kernel_filter}_seed{seed_filter}.json') if seed_filter else (ROOT/'results/message_contributions.json'),dict(test_targets=len(data)-1,parity=parity,rows=rows));print(kernel,seed,'contributions complete',flush=True)

def timing():
 core_id=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{core_id});x=np.fromfile(ROOT/'data/train_segment.txt',np.uint8)[:9217]
 sources={k:Model.load(TEMP/f'full_{k}_seed1/last.npz') for k in ('relays','messages')};rows=[]
 for trial in range(6):
  for kernel in ('relays','messages') if trial%2==0 else ('messages','relays'):
   m=sources[kernel].snapshot_copy();m.run(x[:1024],x[1:1025],mode='periodic',lr=.008,lp=.008/3,route=.002)
   begin=time.perf_counter();loss=m.run(x[1024:-1],x[1025:],mode='periodic',lr=.008,lp=.008/3,route=.002);sec=time.perf_counter()-begin
   rows.append(dict(kernel=kernel,trial=trial,seconds=sec,bytes_per_second=8192/sec,nll=loss));m.close();print('timing',kernel,trial,round(8192/sec),flush=True)
 medians={k:statistics.median(r['bytes_per_second'] for r in rows if r['kernel']==k) for k in sources}
 for m in sources.values():m.close()
 write(ROOT/'results/message_throughput.json',dict(core=core_id,targets_per_trial=8192,trials=rows,median_bytes_per_second=medians,ratio=medians['messages']/medians['relays'],excludes='compilation, load/copy, 1024-target warmup and checkpoint I/O'))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['interventions','timing','worker']);p.add_argument('--seed',type=int);p.add_argument('--kernel');p.add_argument('--core',type=int);a=p.parse_args()
 if a.stage=='worker':
  os.sched_setaffinity(0,{a.core});interventions(a.seed,a.kernel)
 elif a.stage=='timing':timing()
 else:
  import subprocess
  cpus=sorted(os.sched_getaffinity(0));jobs=[]
  for seed in (1,2):
   for kernel in ('relays','messages'):
    log=open(TEMP/f'interventions_{kernel}_seed{seed}.log','w')
    proc=subprocess.Popen([sys.executable,__file__,'worker','--seed',str(seed),'--kernel',kernel,'--core',str(cpus[len(jobs)%len(cpus)])],stdout=log,stderr=subprocess.STDOUT)
    jobs.append((proc,log,kernel,seed))
  for proc,log,kernel,seed in jobs:
   code=proc.wait();log.close()
   if code:raise RuntimeError(f'{kernel} seed {seed} failed')
  records=[json.loads((TEMP/f'interventions_{k}_seed{s}.json').read_text()) for _,_,k,s in jobs]
  write(ROOT/'results/message_contributions.json',dict(test_targets=78771,parity=[p for r in records for p in r['parity']],rows=[p for r in records for p in r['rows']]))
