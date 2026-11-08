"""Drain fixed-endpoint evaluations as matched training arms finish."""
import argparse
from collections import deque
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'artifacts/hippocampus')
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(); root=a.root.resolve(); results=root/'evaluation'; results.mkdir(exist_ok=True)
    queue=deque(); active=[]; scheduled=set(); failed=[]; completed=[]
    def schedule(arm,checkpoint,control='intact',parts=('pairs','single','correction','streams')):
        for part in parts:
            name=f'{arm}_{control}_{part}'
            out=results/(name+'.json'); log=results/(name+'.log')
            command=[sys.executable,str(ROOT/'scripts/evaluate_hippocampus.py'),'--checkpoint',str(checkpoint),
                     '--out',str(out),'--part',part,'--control',control]
            queue.append((name,command,log))
    schedule('parent',root/'parent.pt'); scheduled.add('parent')
    while True:
        for arm in ('A','B','C'):
            if arm in scheduled: continue
            history=root/arm/'history.jsonl'
            if not history.exists(): continue
            try: last=json.loads(history.read_text().splitlines()[-1])
            except (IndexError,json.JSONDecodeError): continue
            if last['step']!=64: continue
            checkpoint=root/arm/'last.pt'
            schedule(arm,checkpoint); scheduled.add(arm)
            if arm=='C':
                for control in ('blocked','shuffled','pooled','no_refinement','no_feedback'):
                    schedule('C',checkpoint,control,('pairs',))
        while queue and len(active)<a.workers:
            name,command,log=queue.popleft(); stream=log.open('w')
            child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            active.append((name,child,stream)); print('start '+name,flush=True)
        remaining=[]
        for name,child,stream in active:
            status=child.poll()
            if status is None: remaining.append((name,child,stream)); continue
            stream.close()
            (failed if status else completed).append(name)
            print(('failed ' if status else 'done ')+name,flush=True)
        active=remaining
        summary=dict(scheduled=sorted(scheduled),active=[x[0] for x in active],queued=[x[0] for x in queue],
                     completed=completed,failed=failed)
        (results/'progress.json').write_text(json.dumps(summary,indent=2)+'\n')
        if len(scheduled)==4 and not active and not queue: break
        time.sleep(2)
    if failed: raise SystemExit('Failed evaluations: '+','.join(failed))

if __name__=='__main__': main()
