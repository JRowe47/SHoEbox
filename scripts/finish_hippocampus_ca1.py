"""Finish evaluation once both declared 16-update continuations are complete."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_memory_selectivity import write_json


def main():
 root=ROOT/'artifacts/hippocampus_binding/ca1';out=root/'evaluation';out.mkdir(exist_ok=True)
 status=dict(stage='waiting_for_training',completed=[],failed=[])
 def save():write_json(root/'pipeline.json',status)
 save()
 def ready(name):
  p=root/name/'history.jsonl'
  return p.exists() and p.read_text().splitlines() and json.loads(p.read_text().splitlines()[-1])['step']==16
 while not all(ready(name) for name in ('legacy','bounded')):time.sleep(2)
 status['stage']='evaluation';save()
 jobs=[]
 for name in ('legacy','bounded'):
  for part in ('pairs','retention','language'):
   jobs.append((name+'_'+part,root/name/'last.pt',part,[]))
 jobs.extend([(label,root/'bounded/last.pt','pairs',flags) for label,flags in
              [('bounded_blocked',['--blocked']),('bounded_legacy_policy',['--policy','legacy'])]])
 def work(job):
  name,checkpoint,part,flags=job
  with (out/(name+'.log')).open('w') as log:
   done=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_hippocampus_ca1.py'),
        '--checkpoint',str(checkpoint),'--part',part,'--out',str(out/(name+'.json')),*flags],
        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  return name,done.returncode
 with ThreadPoolExecutor(max_workers=3) as pool:
  for task in as_completed([pool.submit(work,j) for j in jobs]):
   name,code=task.result();status['completed' if not code else 'failed'].append(name);save()
   print(json.dumps({'job':name,'exit_code':code}),flush=True)
 status['stage']='failed' if status['failed'] else 'complete';save()
 if status['failed']:raise SystemExit(1)

if __name__=='__main__':main()
