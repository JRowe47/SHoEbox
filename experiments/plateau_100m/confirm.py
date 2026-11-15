"""Select two screen winners, preserve screen evidence, then run confirmation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import subprocess
import sys
import time
from common import atomic
from run_arm import ARMS

p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
work=a.base/'diagnostic_40m';source=Path(__file__).parent
while True:
    results={}
    for arm in ARMS:
        path=work/'results'/arm/'results.json'
        if path.exists():
            r=json.loads(path.read_text())
            if r.get('development') and r.get('fresh') and r['targets']==1_000_000:results[arm]=r
    if len(results)==len(ARMS):break
    time.sleep(2)
archive=work/'results/screen';archive.mkdir(exist_ok=True)
for arm,r in results.items():atomic(archive/f'{arm}.json',r)
ranking=sorted((r['development']['nll']+r['fresh']['nll'],arm) for arm,r in results.items() if arm!='baseline')
selected=['baseline']+[arm for _,arm in ranking[:2]]
atomic(work/'selection.json',dict(rule='Lowest sum of full development and fresh-screen NLL at 1M, excluding baseline; retain two arms.',ranking=ranking,confirmed_arms=selected,targets=3_000_000,audit_previously_unscored=True))
print(json.dumps(dict(selected=selected)),flush=True)
common=['--base',str(a.base),'--runtime',str(work/'runtime_followup')]
tasks=[('confirm_'+arm,[sys.executable,str(source/'run_arm.py'),*common,'--arm',arm,'--out',str(work/'results'/arm),'--resume','--targets','3000000']) for arm in selected]
for mode in ['full','no_bias']:
    tasks.append(('probe_40_audit_'+mode,[sys.executable,str(source/'run_probe.py'),*common,'--dataset','audit','--intervention',mode,'--out',str(work/f'results/probe_40_audit_{mode}.json')]))
cpus=queue.Queue()
for cpu in [0,1,2]:cpus.put(cpu)
def run(task):
    name,command=task;cpu=cpus.get()
    try:
        with (work/f'{name}.log').open('w') as log:r=subprocess.run([*command,'--cpu',str(cpu)],stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps(dict(task=name,exit_code=r.returncode)),flush=True);return r.returncode
    finally:cpus.put(cpu)
with ThreadPoolExecutor(max_workers=3) as pool:status=list(pool.map(run,tasks))
if any(status):raise SystemExit(1)
