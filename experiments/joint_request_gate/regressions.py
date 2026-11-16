"""E7 fresh grouped production regressions, with each actual adapter named."""
import argparse,hashlib,json,os,random
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
from .adapter import Adapter,ROOT
from .qualify import source_hashes
from experiments.joint_readout.adapter import Adapter as Old
from experiments.joint_readout import regressions as reg
from experiments.mad.tasks import TaskSpec,_candidate,fixed_table,solve
from experiments.query_readout.run import write
SALT=b'SHoEbox-EXP01E7-regression-v1\0'
ORIGINAL_CASES=reg.cases
def fresh_cases(spec,n,fold):
 excluded={e.split_key for e in ORIGINAL_CASES(spec,128,'test')};excluded.update(e.split_key for e in reg.old_dataset(spec,128,'test'));rng=random.Random(20261123);table=fixed_table(spec,47);out=[];seen=set()
 for _ in range(n*10000):
  e=_candidate(spec,rng,table,len(out));h=hashlib.sha256(SALT+spec.task.encode()+e.split_key).digest();b=int.from_bytes(h[:8],'big')%10;f='train' if b<6 else 'dev' if b<8 else 'test'
  if f!=fold or e.split_key in excluded or e.digest in seen:continue
  assert solve(e,spec,table)==e.answer;out.append(e);seen.add(e.digest)
  if len(out)==n:return out
 raise ValueError('insufficient grouped cases')
def job(args):
 seed,plan,folder,data,phase=args;rows=[]
 for arm,cls,kind in [('native',Old,'native'),('d3_joint',Old,'paired_latest'),('local_gate',Adapter,'paired_local')]:
  reg.Adapter=cls;reg.ARMS={arm:kind};reg.cases=fresh_cases
  # Separate folders preserve each arm's write-once checkpoint lineage.
  f=Path(folder)/arm;f.mkdir(exist_ok=True)
  rows.extend((reg.job if phase=='train' else reg.audit_job)((seed,plan,f,data)))
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=('train','audit'));p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if a.phase=='train':
  if (a.out/'plan.json').exists():raise FileExistsError('registered run exists')
  sources=source_hashes();sources.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*') if p.suffix in ('.py','.hpp','.cpp')})
  plan=dict(utc=datetime.now(timezone.utc).isoformat(),phase='E7-regressions',seeds=[1,7,19],cpus=sorted(os.sched_getaffinity(0))[:a.workers],source_hashes=sources,data_manifest_sha256=hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest(),fold_salt=SALT.decode(),budgets=dict(language=1000000,selective_copy=250000,compression=250000),no_local_labels=True)
  write(a.out/'plan.json',plan)
 else:
  plan=json.loads((a.out/'plan.json').read_text())
  if not (a.out/'selection.json').exists() or (a.out/'audit.json').exists():raise ValueError('lock before one audit')
  for rel,h in plan['source_hashes'].items():
   if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise ValueError('registered source changed: '+rel)
 with ProcessPoolExecutor(max_workers=a.workers) as pool:groups=list(pool.map(job,[(s,plan,a.out,a.data,a.phase) for s in plan['seeds']]))
 write(a.out/(a.phase+'.json'),dict(plan=plan,rows=[r for group in groups for r in group]))
 if a.phase=='train':write(a.out/'selection.json',dict(locked=True,policy='all fixed recipes; no audit selection'))
if __name__=='__main__':main()
