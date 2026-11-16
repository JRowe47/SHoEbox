"""Registered EXP01-E: fit on development, lock, then independently audit."""
import argparse,hashlib,json,os,time,resource
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .adapter import Adapter,PARAMETERS,PAIRED_PARAMETERS,ROOT,library
from .tasks import dataset,balanced_offsets,SALT
from .stream import new_cursor,advance
from experiments.query_access.run import evaluate
from experiments.constituent_readout.stage3 import audits
from experiments.mad_next.tasks import fingerprint
from experiments.query_readout.run import write
ARMS={'native_masked':('native',True,False,False),'frozen_d3':('paired_latest',False,False,True),
 'joint_d3':('paired_latest',True,False,True),'native_all':('native',True,True,False),
 'gated_all':('paired_gated',True,True,True),'gated_no_local':('paired_gated',True,True,False)}

def source_hashes():
 paths=[ROOT/'reports/exp01e/PROTOCOL.md']
 for group in ('multiscale','experiments/joint_readout','experiments/constituent_readout','experiments/query_access','experiments/query_readout','experiments/mad_next','experiments/mad'):
  paths += [p for p in (ROOT/group).glob('*') if p.suffix in ('.py','.cpp','.hpp')]
 return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}

def fit(m,cases,dev,budget,interval,cursor):
 cpu=wall=0.;curve=[];total_loss=0.;initial=m.inventory();start_bytes=cursor['bytes'];start_targets=cursor['targets']
 while cursor['bytes']-start_bytes<budget:
  count=min(interval,budget-(cursor['bytes']-start_bytes));c=time.process_time();w=time.perf_counter()
  cursor,loss=advance(m,cursor,count,cases);m.finish();cpu+=time.process_time()-c;wall+=time.perf_counter()-w;total_loss+=loss
  curve.append(dict(bytes=cursor['bytes']-start_bytes,cpu_seconds=cpu,train_nll=total_loss/max(1,cursor['targets']-start_targets),dev=evaluate(m,dev,all_modes=False)))
 return dict(bytes=budget,targets=cursor['targets']-start_targets,padding_bytes=cursor['padding_bytes'],answer_targets=cursor['answer_targets'],cpu_seconds=cpu,wall_seconds=wall,bytes_per_cpu_second=budget/cpu,curve=curve,inventory=m.inventory(),initial_inventory=initial,order_sha256=cursor['order_sha256']),cursor

def job(args):
 seed,plan,folder=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)%len(plan['cpus'])]})
 out=Path(folder)/str(seed);out.mkdir(exist_ok=True);tr=dataset(2048,'train');dv=balanced_offsets(dataset(512,'dev'))
 row=dict(seed=seed,train_hash=fingerprint(tr),dev_hash=fingerprint(dv),arms={})
 for arm in plan['arms']:
  kind,joint,all_bytes,local=ARMS[arm];extra=PARAMETERS if kind=='paired_gated' else PAIRED_PARAMETERS if kind!='native' else 0
  with Adapter(parameters=plan['parameters']-extra,seed=seed,kind='native',joint=True) as b:
   warm,cursor=fit(b,tr,dv,250000,250000,new_cursor(701,local=False))
   with Adapter(parameters=plan['parameters'],seed=seed,kind=kind,joint=joint) as m:
    m.adopt(b);m.reset();cursor=new_cursor(all_bytes=all_bytes,local=local)
    result,cursor=fit(m,tr,dv,plan['continuation'],500000,cursor)
    result['warmup']=warm;raw=m.dumps();(out/f'{arm}.qr').write_bytes(raw);write(out/f'{arm}.cursor.json',cursor)
    result['checkpoint_sha256']=hashlib.sha256(raw).hexdigest();result['checkpoint_bytes']=len(raw)
    with Adapter.loads(raw) as restored:assert evaluate(m,dv[:8])==evaluate(restored,dv[:8])
    result['reload_exact']=True
    if plan['phase']=='pilot':result['development_audit']=audits(m,dataset(512,'dev'))
   row['arms'][arm]=result;write(out/'train.json',row);print(seed,arm,result['curve'][-1]['dev']['native_rollout'],flush=True)
 row['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;write(out/'train.json',row);return row

def audit_job(args):
 seed,plan,folder=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)%len(plan['cpus'])]})
 base=dataset(1024,'test');rows={};out=Path(folder)/str(seed)
 for arm in plan['arms']:
  with Adapter.loads((out/f'{arm}.qr').read_bytes()) as m:rows[arm]=audits(m,base,arm not in ('native_masked','native_all'))
  write(out/'audit_partial.json',rows);print('audit',seed,arm,rows[arm]['test']['native_rollout'],flush=True)
 row=dict(seed=seed,test_hash=fingerprint(base),cases=len(base),worlds=len({e.split_key for e in base}),arms=rows);write(out/'audit.json',row);return row

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=('pilot','train','audit'));p.add_argument('--out',required=True,type=Path);p.add_argument('--workers',type=int,default=3);p.add_argument('--parameters',type=int,default=500000);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);library()
 if a.phase!='audit':
  if (a.out/'plan.json').exists():raise FileExistsError('registered run exists')
  arms=['gated_all'] if a.phase=='pilot' else list(ARMS) if a.parameters==500000 else ['native_all','frozen_d3','gated_all']
  plan=dict(experiment='EXP01-E1',phase=a.phase,utc=datetime.now(timezone.utc).isoformat(),seeds=[7] if a.phase=='pilot' else [1,7,19],arms=arms,parameters=a.parameters,warmup=250000,continuation=2000000,fold_salt=SALT.decode(),data_seed=20261120,cpus=sorted(os.sched_getaffinity(0))[:a.workers],source_hashes=source_hashes())
  write(a.out/'plan.json',plan);fn=job
 else:
  plan=json.loads((a.out/'plan.json').read_text());fn=audit_job
  if not (a.out/'selection.json').exists():raise ValueError('lock selection first')
  if (a.out/'audit.json').exists():raise FileExistsError('audit exists')
  for rel,h in plan['source_hashes'].items():
   if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise ValueError('registered source changed: '+rel)
 with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(fn,[(s,plan,a.out) for s in plan['seeds']]))
 write(a.out/('audit.json' if a.phase=='audit' else 'train.json'),dict(plan=plan,rows=rows))
 if a.phase=='train':
  scores=[r['arms']['gated_all']['curve'][-1]['dev']['native_rollout']['native_exact'] for r in rows]
  write(a.out/'selection.json',dict(selected='gated_all',seeds=scores,mean=float(np.mean(scores)),locked_before_audit=True,policy='fixed final checkpoint, no audit tuning'))
if __name__=='__main__':main()
