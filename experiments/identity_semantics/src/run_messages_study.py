"""Validation-selected message controls; matched two-seed million-byte study."""
import argparse,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
from runtime import fresh_model,Model,core
from run_semantics_study import evaluate,write
TEMP=ROOT/'results/message_tmp'
CANDIDATES=[
 dict(name='small',payload_gain=.25,feedback_activation=.1,feedback_routing=.1,semantic_lr=1.,peak=.08),
 dict(name='medium',payload_gain=.5,feedback_activation=.25,feedback_routing=.25,semantic_lr=1.,peak=.08),
 dict(name='routing',payload_gain=.5,feedback_activation=.1,feedback_routing=.5,semantic_lr=1.,peak=.08),
 dict(name='payload',payload_gain=1.,feedback_activation=.1,feedback_routing=.1,semantic_lr=1.,peak=.08),
 dict(name='medium_slow',payload_gain=.5,feedback_activation=.25,feedback_routing=.25,semantic_lr=1.,peak=.04),
 dict(name='medium_semantic',payload_gain=.5,feedback_activation=.25,feedback_routing=.25,semantic_lr=2.,peak=.08),
]
def worker(job):
 os.sched_setaffinity(0,{job['core']});folder=TEMP/job['name'];folder.mkdir(parents=True,exist_ok=True)
 data=np.fromfile(ROOT/'data/train_segment.txt',np.uint8);held=np.fromfile(ROOT/'data/heldout.bin',np.uint8)
 m=fresh_model(job['seed'],kernel=job['kernel'])
 if job['kernel']=='messages':m.configure_messages(job['policy'])
 out=dict(job=job,compiled_kernel=Path(m.lib._name).name,parameters=m.parameter_count,train_sha256=hashlib.sha256(data).hexdigest(),heldout_sha256=hashlib.sha256(held).hexdigest(),training=[],validation=[],train_seconds=0.,status='running')
 for pos in range(0,job['tokens'],5000):
  n=min(5000,job['tokens']-pos);lr=job['peak']*(.1+.45*(1+math.cos(math.pi*(pos+n/2)/1000000)))
  begin=time.perf_counter();loss=m.run(data[pos:pos+n],data[pos+1:pos+n+1],mode='periodic',lr=lr,lp=lr/3,route=.002);sec=time.perf_counter()-begin
  out['train_seconds']+=sec;out['training'].append(dict(tokens=pos+n,nll=loss,lr=lr,seconds=sec))
  if (pos+n)%100000==0 or pos+n==job['tokens']:
   val=evaluate(m,held[:32769]);out['validation'].append(dict(tokens=pos+n,nll=val));write(folder/'training.json',out);print(job['name'],pos+n,round(val,6),flush=True)
 if job['tokens']==1000000:
  out['test_nll']=evaluate(m,held[32768:]);out['test_targets']=78771
  out['lace_metrics']=m.lace_metrics();out['message_metrics']=m.message_metrics();out['composition_metrics']=m.composition_metrics();out['teaching']=m.relay_teaching_status();out['stats']=m.stats()
  for v in out['teaching'].values():
   assert v['expired_producers']==v['duplicate_schedules']==0
   assert v['scheduled']==sum(v[k] for k in ('neural_targets','relay_targets','nonlearning_targets','expired_producers'))+int(v['pending_tick']>0)
  assert all(v['missing_parent']==0 for v in out['lace_metrics'].values())
  assert all(v['expired_input_edges']==0 for v in out['composition_metrics'].values())
  assert all(v['missing_feedback_producers']==0 for v in out['message_metrics'].values())
  m.save(folder/'last.npz')
 out['semantic_updates']=m.semantic_updates();out['status']='complete';write(folder/'training.json',out);m.close()
def launch(jobs,workers=4):
 pending=list(jobs);active=[];TEMP.mkdir(parents=True,exist_ok=True)
 while pending or active:
  while pending and len(active)<workers:
   job=pending.pop(0);path=TEMP/(job['name']+'.json');write(path,job);log=open(TEMP/(job['name']+'.log'),'w')
   p=subprocess.Popen([sys.executable,__file__,'worker','--job',str(path)],stdout=log,stderr=subprocess.STDOUT);active.append((p,log,job))
  for p,log,job in active[:]:
   if p.poll() is not None:
    log.close();active.remove((p,log,job));print(job['name'],p.returncode,flush=True)
    if p.returncode:raise RuntimeError('Worker failed: '+job['name'])
  if active:time.sleep(.2)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['worker','pilots','full']);p.add_argument('--job',type=Path);a=p.parse_args()
 if a.stage=='worker':worker(json.loads(a.job.read_text()));return
 cpus=sorted(os.sched_getaffinity(0));jobs=[]
 if a.stage=='pilots':
  for i,c in enumerate(CANDIDATES):
   jobs.append(dict(name='pilot_'+c['name'],kernel='messages',seed=3,tokens=100000,core=cpus[i%4],peak=c['peak'],policy={k:v for k,v in c.items() if k not in ('name','peak')}))
  launch(jobs,min(4,len(cpus)));rows=[json.loads((TEMP/j['name']/'training.json').read_text()) for j in jobs]
  winner=min(rows,key=lambda r:r['validation'][-1]['nll']);write(ROOT/'results/message_selection.json',dict(candidates=rows,winner=winner['job'],test_used=False))
 else:
  chosen=json.loads((ROOT/'results/message_selection.json').read_text())['winner']
  for seed in (1,2):
   for kernel in ('relays','messages'):
    jobs.append(dict(name=f'full_{kernel}_seed{seed}',kernel=kernel,seed=seed,tokens=1000000,core=cpus[len(jobs)%len(cpus)],peak=.08 if kernel=='relays' else chosen['peak'],policy={} if kernel=='relays' else chosen['policy']))
  launch(jobs,min(4,len(cpus)))
  write(ROOT/'results/message_training.json',[json.loads((TEMP/j['name']/'training.json').read_text()) for j in jobs])
if __name__=='__main__':main()
