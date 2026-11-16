"""E8 fixed three-seed comparison; final snapshots locked before one audit."""
import argparse,hashlib,json,os,resource
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
from .adapter import Adapter,PARAMETERS,ROOT,library
from .tasks import dataset,balanced_offsets,SALT
from .fit import fit
from .stream import new_cursor
from experiments.joint_readout.adapter import Adapter as Old,PAIRED_PARAMETERS
from experiments.joint_readout.run import fit as oldfit,source_hashes as old_source_hashes
from experiments.joint_readout.regressions import Corpus,language
from experiments.constituent_readout.stage3 import audits
from experiments.mad_next.tasks import fingerprint
from experiments.query_readout.run import write
ARMS={'native_all':(Old,'native',True,True,False,0),'frozen_d3':(Old,'paired_latest',False,False,True,PAIRED_PARAMETERS),'generic_gate':(Adapter,'paired_gated',True,True,True,PARAMETERS),'local_gate':(Adapter,'paired_local',True,True,True,PARAMETERS),'no_local':(Adapter,'paired_local',True,True,False,PARAMETERS)}

def source_hashes():
 sources=old_source_hashes()
 for group in ('joint_cold_gate','joint_request_gate','joint_switch'):
  sources.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'experiments'/group).glob('*') if p.suffix in ('.py','.hpp','.cpp')})
 return sources

def job(args):
 seed,plan,folder,data=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)%len(plan['cpus'])]});out=Path(folder)/str(seed);out.mkdir(exist_ok=True)
 tr=dataset(2048,'train');dv=balanced_offsets(dataset(512,'dev'));row=dict(seed=seed,train_hash=fingerprint(tr),dev_hash=fingerprint(dv),arms={})
 import experiments.joint_readout.regressions as reg
 for arm in plan['arms']:
  cls,kind,joint,all_bytes,local,extra=ARMS[arm];f=fit if cls is Adapter else oldfit
  with cls(parameters=plan['parameters']-extra,seed=seed,kind='native',joint=True) as b:
   warm,_=f(b,tr,dv,250000,250000,new_cursor(701,local=False))
   with cls(parameters=plan['parameters'],seed=seed,kind=kind,joint=joint) as m:
    m.adopt(b);m.reset();r,cursor=f(m,tr,dv,2000000,500000,new_cursor(all_bytes=all_bytes,local=local));r['warmup']=warm
    raw=m.dumps();(out/f'{arm}.qr').write_bytes(raw);write(out/f'{arm}.cursor.json',cursor);r['checkpoint_sha256']=hashlib.sha256(raw).hexdigest()
    r['development_audit']=audits(m,dataset(512,'dev'),arm!='native_all');row['arms'][arm]=r;write(out/'train.json',row)
    if plan['retention']:
     reg.Adapter=cls;reg.binding=dataset
     r['retention']=language(m,Corpus(data),out/f'{arm}.retained.qr',True);r['retention']['development_audit']=audits(m,dataset(512,'dev'),arm!='native_all')
    write(out/'train.json',row);print(seed,arm,r['development_audit']['test']['native_rollout']['native_exact'],flush=True)
 row['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;write(out/'train.json',row);return row

def audit_job(args):
 seed,plan,folder,data=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)%len(plan['cpus'])]});out=Path(folder)/str(seed);base=dataset(1024,'test');row=dict(seed=seed,test_hash=fingerprint(base),arms={})
 for arm in plan['arms']:
  cls=ARMS[arm][0]
  with cls.loads((out/f'{arm}.qr').read_bytes()) as m:r=audits(m,base,arm!='native_all')
  if plan['retention']:
   with cls.loads((out/f'{arm}.retained.qr').read_bytes()) as m:r['retention']=audits(m,base,arm!='native_all')
  row['arms'][arm]=r;write(out/'audit.json',row)
 return row

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','audit']);p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--parameters',type=int,default=500000);p.add_argument('--workers',type=int,default=3);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);library()
 if a.phase=='train':
  if (a.out/'plan.json').exists():raise FileExistsError('registered run exists')
  sources=source_hashes();sources.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*') if p.suffix in ('.py','.hpp','.cpp')})
  plan=dict(utc=datetime.now(timezone.utc).isoformat(),phase='E8-formal',seeds=[1,7,19],parameters=a.parameters,warmup=250000,continuation=2000000,retention=a.parameters==500000,arms=list(ARMS) if a.parameters==500000 else ['native_all','frozen_d3','local_gate'],cpus=sorted(os.sched_getaffinity(0))[:a.workers],source_hashes=sources,fold_salt=SALT.decode(),data_manifest_sha256=hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((ROOT/'reports/exp01e/AMENDMENT_E8.md').read_bytes()).hexdigest())
  write(a.out/'plan.json',plan);fn=job
 else:
  plan=json.loads((a.out/'plan.json').read_text());fn=audit_job
  if not (a.out/'selection.json').exists() or (a.out/'audit.json').exists():raise ValueError('lock final fit before one-time audit')
  for rel,h in plan['source_hashes'].items():
   if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise ValueError('registered source changed: '+rel)
 with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(fn,[(s,plan,a.out,a.data) for s in plan['seeds']]))
 write(a.out/(a.phase+'.json'),dict(plan=plan,rows=rows))
 if a.phase=='train':write(a.out/'selection.json',dict(selected='local_gate',locked_before_audit=True,policy='fixed final snapshots; no audit tuning'))
if __name__=='__main__':main()
