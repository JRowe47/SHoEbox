"""D2 query occurrence policy, independent registration and frozen audit."""
import argparse,hashlib,json,os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .adapter import Adapter,HERE,ROOT,PARAMETERS,library
from .tasks import dataset,balanced_offsets,shifted,SALT
from .diagnose import diagnose
from experiments.query_access.adapter import Adapter as Pool
from experiments.query_access.run import train,evaluate,frozen_query_pairs
from experiments.mad_next.tasks import fingerprint,Spec,shortcuts
from experiments.query_readout.run import write
ARMS=['native','pool','paired','no_local','paired_latest']

def job(args):
 seed,plan,folder=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)%len(plan['cpus'])]})
 stem=Path(folder)/str(seed);stem.mkdir(exist_ok=True);tr=dataset(2048,'train');dv=balanced_offsets(dataset(512,'dev'))
 row=dict(seed=seed,dataset=dict(train=fingerprint(tr),dev=fingerprint(dv)))
 with Adapter(parameters=500000-PARAMETERS,seed=seed,kind='native',joint=True) as b:
  row['warmup']=train(b,tr,dv,250000,250000,701)
  with Adapter(seed=seed,kind='paired_latest') as m:
   m.adopt(b);r=train(m,tr,dv,2000000,500000,702);r['inventory']=m.inventory()
   raw=m.dumps();(stem/'paired_latest.qr').write_bytes(raw);r['checkpoint_sha256']=hashlib.sha256(raw).hexdigest();r['checkpoint_bytes']=len(raw)
   with Adapter.loads(raw) as restored:assert evaluate(m,dv[:8])==evaluate(restored,dv[:8])
   row['fit']=r;row['diagnostic']=diagnose(m,dv);row['diagnostic'].pop('failures');write(stem/'train.json',row)
   print(seed,'D2',r['curve'][-1]['dev']['native_rollout'],flush=True)
 return row

def audit_job(args):
 seed,plan,folder=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)%len(plan['cpus'])]})
 stem=Path(folder)/str(seed);base=dataset(1024,'test');cases=balanced_offsets(base);rows={}
 for arm in ARMS:
  root=Path(folder) if arm=='paired_latest' else Path(plan['d1'])
  cls=Pool if arm=='pool' else Adapter
  with cls.loads((root/str(seed)/f'{arm}.qr').read_bytes()) as m:
   row=dict(test=evaluate(m,cases),counterfactual=frozen_query_pairs(m,cases[:256]))
   if arm in ('paired','paired_latest','no_local'):
    row['lesions']={name:evaluate(m,cases[:256],name,False)['native_rollout'] for name in ('no_head','no_query','no_state','uniform','reverse','unpair')}
    row['lesion_unmodified']=evaluate(m,cases[:256],all_modes=False)['native_rollout'];row['diagnostic']=diagnose(m,cases[:256]);row['diagnostic'].pop('failures')
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
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','audit']);p.add_argument('--out',type=Path,required=True);p.add_argument('--d1',type=Path,default=Path('/tmp/exp01d-formal'));p.add_argument('--workers',type=int,default=3);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);library()
 if a.phase=='train':
  files=[ROOT/'reports/exp01d/PROTOCOL.md',ROOT/'reports/exp01d/AMENDMENT_D2.md',*HERE.glob('*.py'),*HERE.glob('*.hpp'),*HERE.glob('*.cpp'),ROOT/'multiscale/native.cpp',*(ROOT/'multiscale').glob('*.hpp')]
  plan=dict(stage='D2',registered_utc=datetime.now(timezone.utc).isoformat(),seeds=[1,7,19],parameters=500000,warmup=250000,continuation=2000000,head_rate=.16,local_weight=1,fold_salt=SALT.decode(),data_seed=20261118,d1=str(a.d1.resolve()),cpus=sorted(os.sched_getaffinity(0))[:a.workers],source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
  if (a.out/'plan.json').exists():raise FileExistsError('D2 registered run exists')
  write(a.out/'plan.json',plan);fn=job
 else:
  plan=json.loads((a.out/'plan.json').read_text());fn=audit_job
  if not (a.out/'selection.json').exists():raise ValueError('selection must precede audit')
  if (a.out/'audit.json').exists():raise FileExistsError('audit already completed')
  for rel,expected in plan['source_hashes'].items():
   if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:raise ValueError('source changed: '+rel)
 with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(fn,[(s,plan,a.out) for s in plan['seeds']]))
 write(a.out/(a.phase+'.json'),dict(plan=plan,rows=rows))
 if a.phase=='train':
  scores=[r['fit']['curve'][-1]['dev']['native_rollout']['native_exact'] for r in rows]
  write(a.out/'selection.json',dict(selected='paired_latest',mean=float(np.mean(scores)),seeds=scores,locked_before_audit=True,recipe='D2 zero-logit recognition then latest occurrence, final 2M checkpoint'))
if __name__=='__main__':main()
