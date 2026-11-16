"""Three-seed registered training, locked development selection, untouched audit."""
import argparse,hashlib,json,os,resource
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .adapter import Adapter,PARAMETERS,HERE,ROOT,library
from .tasks import dataset,shifted,balanced_offsets,SALT
from .diagnose import diagnose
from experiments.query_access.adapter import Adapter as Pool,PARAMETERS as POOL_PARAMS
from experiments.query_access.run import train,evaluate,frozen_query_pairs
from experiments.mad_next.tasks import fingerprint,Spec,shortcuts
from experiments.query_readout.run import write
ARMS=['native','pool','paired','no_local']

def registration(folder,workers):
 files=[ROOT/'reports/exp01d/PROTOCOL.md',*HERE.glob('*.py'),*HERE.glob('*.hpp'),*HERE.glob('*.cpp'),ROOT/'multiscale/native.cpp',*(ROOT/'multiscale').glob('*.hpp')]
 plan=dict(experiment='EXP01-D/EXP22',registered_utc=datetime.now(timezone.utc).isoformat(),starting_commit='fadf56f6e2cd4b48204ac57848d73e12f5d09874',
   seeds=[1,7,19],arms=ARMS,parameters=500000,warmup_bytes=250000,continuation_bytes=2000000,eval_interval=500000,
   head_rate=.16,local_weight=1,fold_salt=SALT.decode(),data_seed=20261118,training_cases=2048,dev_cases=512,audit_cases=1024,
   cpus=sorted(os.sched_getaffinity(0))[:workers],
   source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
 if (folder/'plan.json').exists():raise FileExistsError('registered run exists')
 write(folder/'plan.json',plan);return plan

def job(args):
 seed,plan,folder=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)%len(plan['cpus'])]})
 stem=Path(folder)/str(seed);stem.mkdir(exist_ok=True);tr=dataset(2048,'train');dv=balanced_offsets(dataset(512,'dev'))
 row=dict(seed=seed,core='minimal-v19',datasets=dict(train=fingerprint(tr),dev=fingerprint(dv)),arms={},warmups={})
 for group in ('native','pool','paired'):
  cls=Pool if group=='pool' else Adapter;capacity=500000-({'native':0,'pool':POOL_PARAMS,'paired':PARAMETERS}[group])
  with cls(parameters=capacity,seed=seed,kind='native',joint=True) as base:
   row['warmups'][group]=train(base,tr,dv,250000,250000,701)
   for arm in (['paired','no_local'] if group=='paired' else [group]):
    if arm=='native':m=base.clone(frozen=False)
    else:
     m=cls(seed=seed,kind='pool' if group=='pool' else 'paired');m.adopt(base)
     if arm=='no_local':m.call('qr_auxiliary',0.)
    with m:
     r=train(m,tr,dv,plan['continuation_bytes'],plan['eval_interval'],702);r['inventory']=m.inventory()
     raw=m.dumps();(stem/f'{arm}.qr').write_bytes(raw);r['checkpoint_sha256']=hashlib.sha256(raw).hexdigest();r['checkpoint_bytes']=len(raw)
     with cls.loads(raw) as restored:assert evaluate(m,dv[:8])==evaluate(restored,dv[:8])
     r['reload_exact']=True;row['arms'][arm]=r;write(stem/'train.json',row)
     print(seed,arm,r['curve'][-1]['dev']['native_rollout'],flush=True)
 row['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;write(stem/'train.json',row);return row

def audit_job(args):
 seed,plan,folder=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)%len(plan['cpus'])]})
 base=dataset(1024,'test');cases=balanced_offsets(base);rows={};stem=Path(folder)/str(seed)
 for arm in ARMS:
  cls=Pool if arm=='pool' else Adapter
  with cls.loads((stem/f'{arm}.qr').read_bytes()) as m:
   row=dict(test=evaluate(m,cases),counterfactual=frozen_query_pairs(m,cases[:256]))
   if arm in ('paired','no_local'):
    row['lesions']={name:evaluate(m,cases[:256],name,False)['native_rollout'] for name in ('no_head','no_query','no_state','uniform','reverse','unpair')}
    row['lesion_unmodified']=evaluate(m,cases[:256],all_modes=False)['native_rollout'];row['diagnostic']=diagnose(m,cases[:256])
    row['diagnostic'].pop('failures')
   with m.clone() as f:
    row['offsets']=[]
    for offset in range(64):
     totals=np.zeros(9)
     for ep in base[:64]:totals+=f.score(shifted(ep,offset),2,2)
     row['offsets'].append(dict(offset=offset,exact=totals[5]/64,first=totals[7]/64,second=totals[8]/64))
    row['delays']=[]
    for delay in (0,64,128,256):
     totals=np.zeros(9)
     for offset in range(64):
      for ep in base[64:80]:totals+=f.score(shifted(ep,offset,delay),2,2)
     row['delays'].append(dict(delay=delay,exact=totals[5]/1024,first=totals[7]/1024,second=totals[8]/1024))
   rows[arm]=row;write(stem/'audit_partial.json',rows);print('audit',seed,arm,row['test']['native_rollout'],flush=True)
 result=dict(seed=seed,arms=rows,test_sha256=fingerprint(cases),test_cases=len(cases),independent_worlds=len({e.split_key for e in cases}),shortcuts=shortcuts(cases,Spec('binding',vocab=2)))
 write(stem/'audit.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','audit']);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);library()
 if a.phase=='train':plan=registration(a.out,a.workers);fn=job
 else:
  plan=json.loads((a.out/'plan.json').read_text());fn=audit_job
  if not (a.out/'selection.json').exists():raise ValueError('development selection must precede audit')
  for rel,expected in plan['source_hashes'].items():
   if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:raise ValueError('registered source changed: '+rel)
 with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(fn,[(s,plan,a.out) for s in plan['seeds']]))
 write(a.out/(a.phase+'.json'),dict(plan=plan,rows=rows))
 if a.phase=='train':
  scores={arm:float(np.mean([r['arms'][arm]['curve'][-1]['dev']['native_rollout']['native_exact'] for r in rows])) for arm in ARMS}
  write(a.out/'selection.json',dict(selected='paired',means=scores,locked_before_audit=True,recipe='D1 unchanged, final 2M checkpoint'))
if __name__=='__main__':main()
