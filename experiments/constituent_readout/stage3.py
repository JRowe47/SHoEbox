"""D3 registered context-variation comparison and fresh held-out audit."""
import argparse,hashlib,json,os,random,time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .adapter import Adapter,HERE,ROOT,PARAMETERS,library
from .tasks_v3 import dataset,balanced_offsets,shifted,spaced,SALT
from .diagnose import diagnose
from experiments.query_access.adapter import Adapter as Pool,PARAMETERS as POOL_PARAMS
from experiments.query_access.run import evaluate,frozen_query_pairs
from experiments.mad_next.tasks import fingerprint,Spec,shortcuts
from experiments.query_readout.run import write
ARMS=['native','pool','unspaced','spaced']

def train(m,cases,dev,budget,interval,order_seed,use_spacers):
 total=targets=cpu=wall=episode=0;order=[];rng=random.Random(order_seed);augmentation=random.Random(order_seed+1000);digest=hashlib.sha256();curve=[];next_eval=interval;phases=[0]*64;spacing=[0]*8
 while total<budget:
  if not order:order=list(range(len(cases)));rng.shuffle(order)
  idx=order.pop();lengths=[augmentation.randrange(8),augmentation.randrange(8)] if use_spacers else [0,0]
  ep=spaced(cases[idx],lengths) if use_spacers else cases[idx];offset=(episode%64-sum(lengths))%64;ep=shifted(ep,offset);episode+=1
  phases[ep.mask.index(1)%64]+=1
  for n in lengths:spacing[n]+=1
  c=time.process_time();w=time.perf_counter();used,n,_,_=m.train_episode(ep,budget-total);cpu+=time.process_time()-c;wall+=time.perf_counter()-w;total+=used;targets+=n
  digest.update(f'{idx}:{offset}:{lengths}:{used};'.encode())
  if total>=next_eval or total==budget:
   c=time.process_time();w=time.perf_counter();m.finish();cpu+=time.process_time()-c;wall+=time.perf_counter()-w
   curve.append(dict(bytes=total,cpu_seconds=cpu,dev=evaluate(m,dev,all_modes=False)))
   while next_eval<=total:next_eval+=interval
 return dict(bytes=total,targets=targets,cpu_seconds=cpu,wall_seconds=wall,bytes_per_cpu_second=total/cpu,order_sha256=digest.hexdigest(),answer_phase_counts=phases,spacer_counts=spacing,curve=curve)

def audits(m,base,full=True):
 cases=balanced_offsets(base);row=dict(test=evaluate(m,cases),counterfactual=frozen_query_pairs(m,cases[:256]))
 if full:
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
 return row

def job(args):
 seed,plan,folder=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)%len(plan['cpus'])]})
 stem=Path(folder)/str(seed);stem.mkdir(exist_ok=True);tr=dataset(2048,'train');dv=balanced_offsets(dataset(512,'dev'));row=dict(seed=seed,datasets=dict(train=fingerprint(tr),dev=fingerprint(dv)),arms={},warmups={})
 for group in (['paired'] if plan['phase']=='pilot' else ['native','pool','paired']):
  cls=Pool if group=='pool' else Adapter;capacity=500000-({'native':0,'pool':POOL_PARAMS,'paired':PARAMETERS}[group]);aug=group!='paired'
  with cls(parameters=capacity,seed=seed,kind='native',joint=True) as b:
   row['warmups'][group]=train(b,tr,dv,250000,250000,701,aug)
   for arm in (['spaced'] if plan['phase']=='pilot' else ['unspaced','spaced']) if group=='paired' else [group]:
    m=b.clone(frozen=False) if arm=='native' else cls(seed=seed,kind='pool' if group=='pool' else 'paired_latest')
    with m:
     if arm!='native':m.adopt(b)
     r=train(m,tr,dv,2000000,500000,702,arm!='unspaced');r['inventory']=m.inventory();raw=m.dumps();(stem/f'{arm}.qr').write_bytes(raw);r['checkpoint_sha256']=hashlib.sha256(raw).hexdigest();r['checkpoint_bytes']=len(raw)
     with cls.loads(raw) as restored:assert evaluate(m,dv[:8])==evaluate(restored,dv[:8])
     r['reload_exact']=True
     if plan['phase']=='pilot':r['development_audit']=audits(m,dataset(512,'dev'))
     row['arms'][arm]=r;write(stem/'train.json',row);print(seed,arm,r['curve'][-1]['dev']['native_rollout'],flush=True)
 return row

def audit_job(args):
 seed,plan,folder=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)%len(plan['cpus'])]})
 stem=Path(folder)/str(seed);base=dataset(1024,'test');rows={}
 for arm in ARMS:
  cls=Pool if arm=='pool' else Adapter
  with cls.loads((stem/f'{arm}.qr').read_bytes()) as m:rows[arm]=audits(m,base,arm in ('unspaced','spaced'))
  write(stem/'audit_partial.json',rows);print('audit',seed,arm,rows[arm]['test']['native_rollout'],flush=True)
 result=dict(seed=seed,arms=rows,test_sha256=fingerprint(balanced_offsets(base)),test_cases=len(base),independent_worlds=len({e.split_key for e in base}),shortcuts=shortcuts(base,Spec('binding',vocab=2)))
 write(stem/'audit.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilot','train','audit']);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);library()
 if a.phase!='audit':
  files=[ROOT/'reports/exp01d/AMENDMENT_D3.md',*HERE.glob('*.py'),*HERE.glob('*.hpp'),*HERE.glob('*.cpp'),ROOT/'multiscale/native.cpp',*(ROOT/'multiscale').glob('*.hpp')]
  plan=dict(stage='D3',phase=a.phase,registered_utc=datetime.now(timezone.utc).isoformat(),seeds=[7] if a.phase=='pilot' else [1,7,19],parameters=500000,warmup=250000,continuation=2000000,head_rate=.16,local_weight=1,fold_salt=SALT.decode(),data_seed=20261119,cpus=sorted(os.sched_getaffinity(0))[:a.workers],source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
  if (a.out/'plan.json').exists():raise FileExistsError('registered run exists')
  write(a.out/'plan.json',plan);fn=job
 else:
  plan=json.loads((a.out/'plan.json').read_text());fn=audit_job
  if plan['phase']!='train' or not (a.out/'selection.json').exists():raise ValueError('formal selection must precede audit')
  if (a.out/'audit.json').exists():raise FileExistsError('audit completed')
  for rel,expected in plan['source_hashes'].items():
   if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:raise ValueError('source changed: '+rel)
 with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(fn,[(s,plan,a.out) for s in plan['seeds']]))
 write(a.out/('audit.json' if a.phase=='audit' else 'train.json'),dict(plan=plan,rows=rows))
 if a.phase=='train':
  scores=[r['arms']['spaced']['curve'][-1]['dev']['native_rollout']['native_exact'] for r in rows]
  write(a.out/'selection.json',dict(selected='spaced',mean=float(np.mean(scores)),seeds=scores,locked_before_audit=True,recipe='D2 runtime plus 0..7-byte record spacers during training; long delays never trained'))
if __name__=='__main__':main()
