"""One-time locked E8 audits, preserving the registered folds and thresholds."""
import argparse,hashlib,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from datetime import datetime,timezone
from .adapter import ROOT
from .qualify import ARMS,audit_job
from .regressions import job as generic_job
from .tasks import dataset,balanced_offsets
from .gates import binding_gate,generic_gate
from experiments.query_access.run import evaluate
from experiments.query_readout.run import write

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify_sources(plan):
 note=json.loads((ROOT/'reports/exp01e8/diagnostic_correction.json').read_text())
 for rel,expected in plan['source_hashes'].items():
  observed=sha(ROOT/rel)
  if observed!=expected:
   if rel==note['file'] and expected==note['old_sha256'] and observed==note['new_sha256']:continue
   raise ValueError('registered source changed: '+rel)

def binding_job(args):
 start=time.process_time();seed,plan,folder,data=args;row=audit_job(args);base=balanced_offsets(dataset(1024,'test'))[:256]
 for arm in plan['arms']:
  cls=ARMS[arm][0]
  for retained in ([False,True] if plan['retention'] else [False]):
   path=Path(folder)/str(seed)/(arm+('.retained' if retained else '')+'.qr')
   with cls.loads(path.read_bytes()) as model:lesions={k:evaluate(model,base,k,False)['native_rollout'] for k in ('gate_off','gate_on')}
   target=row['arms'][arm]['retention'] if retained else row['arms'][arm]
   target['gate_interventions']=lesions
 row['audit_cpu_seconds']=time.process_time()-start;write(Path(folder)/str(seed)/'audit.json',row);return row

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args();w=a.workspace
 # Fail closed before any new audit score is computed.
 dev=json.loads((w/'exp01e8-formal/train.json').read_text())
 for retained in (False,True):
  rows=[r['arms']['local_gate']['retention']['development_audit'] if retained else r['arms']['local_gate']['development_audit'] for r in dev['rows']]
  if not binding_gate(rows)['passed']:raise ValueError('binding development prerequisite failed')
 if not generic_gate(json.loads((w/'exp01e8-regressions/train.json').read_text())['rows'])['passed']:raise ValueError('generic development prerequisite failed')
 for size in ('formal','1m','5m'):
  training=json.loads((w/f'exp01e8-{size}/train.json').read_text())
  if not binding_gate([r['arms']['local_gate']['development_audit'] for r in training['rows']])['passed']:raise ValueError('capacity development prerequisite failed: '+size)
  if json.loads((w/f'exp01e8-{size}/benchmark.json').read_text())['relative']<.8:raise ValueError('inference throughput prerequisite failed: '+size)
 if json.loads((w/'exp01e8-formal/retained_benchmark.json').read_text())['relative']<.8:raise ValueError('retained inference throughput prerequisite failed')
 campaigns=('formal','1m','5m','regressions');locks={}
 for campaign in campaigns:
  folder=w/f'exp01e8-{campaign}';plan=json.loads((folder/'plan.json').read_text());verify_sources(plan)
  if (folder/'audit.json').exists():raise FileExistsError('audit already opened: '+campaign)
  if not (folder/'selection.json').exists():raise ValueError('final selection lock absent')
  files=[f for f in folder.rglob('*') if f.is_file() and (f.suffix in ('.qr','.checkpoint') or f.name.endswith('.cursor.json'))]
  locks[campaign]=dict(plan_sha256=sha(folder/'plan.json'),selection_sha256=sha(folder/'selection.json'),files={str(f.relative_to(folder)):sha(f) for f in sorted(files)})
 lock=w/'e8-audit-lock.json'
 if lock.exists():raise FileExistsError(lock)
 write(lock,dict(utc=datetime.now(timezone.utc).isoformat(),audit_driver_sha256=sha(Path(__file__)),gate_checker_sha256=sha(Path(__file__).with_name('gates.py')),scope='Fixed final checkpoints, no audit selection or revision',campaigns=locks))
 starts={campaign:time.perf_counter() for campaign in campaigns};groups={campaign:[] for campaign in campaigns}
 with ProcessPoolExecutor(max_workers=3*a.workers) as pool:
  futures={}
  for campaign in campaigns[:-1]:
   folder=w/f'exp01e8-{campaign}';plan=json.loads((folder/'plan.json').read_text());data=w/'exp01e-data'
   if sha(data/'manifest.json')!=plan['data_manifest_sha256']:raise ValueError('corpus manifest changed')
   for seed in plan['seeds']:futures[pool.submit(binding_job,(seed,plan,folder,data))]=(campaign,folder,plan)
  for future in as_completed(futures):
   campaign,folder,plan=futures[future];groups[campaign].append(future.result())
   if len(groups[campaign])==len(plan['seeds']):
    write(folder/'audit.json',dict(plan=plan,rows=sorted(groups[campaign],key=lambda r:r['seed']),audit_wall_seconds=time.perf_counter()-starts[campaign]))
    print('audit complete',campaign,flush=True)
 campaign='regressions';folder=w/'exp01e8-regressions';plan=json.loads((folder/'plan.json').read_text());data=w/'exp01e7-data'
 if sha(data/'manifest.json')!=plan['data_manifest_sha256']:raise ValueError('corpus manifest changed')
 start=time.perf_counter()
 with ProcessPoolExecutor(max_workers=a.workers) as pool:groups=list(pool.map(generic_job,[(s,plan,folder,data,'audit') for s in plan['seeds']]))
 write(folder/'audit.json',dict(plan=plan,rows=[r for group in groups for r in group],audit_wall_seconds=time.perf_counter()-start))
 print('audit complete',campaign,flush=True)
if __name__=='__main__':main()
