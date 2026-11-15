"""Run independent experiments in bounded worker processes, not training threads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import subprocess
import sys
from run_arm import ARMS

p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
p.add_argument('--kind',choices=['screen','confirm'],default='screen');p.add_argument('--arms',nargs='*',default=[]);a=p.parse_args()
work=a.base/'diagnostic_40m';source=Path(__file__).parent
common=['--base',str(a.base),'--runtime',str(work/'runtime')]
tasks=[]
if a.kind=='screen':
    tasks.append(('arm_baseline',[sys.executable,str(source/'run_arm.py'),*common,'--arm','baseline','--out',str(work/'results/baseline')]))
    for ckpt in ['20','40']:
        tasks.append((f'probe_{ckpt}_full',[sys.executable,str(source/'run_probe.py'),*common,'--checkpoint',ckpt,'--details','--out',str(work/f'results/probe_{ckpt}_full.json')]))
    for mode in ['mask1','mask2','mask3','swap0','swap1','swap2','swap3','no_recall','no_address','no_bias','fast_only']:
        tasks.append((f'probe_40_{mode}',[sys.executable,str(source/'run_probe.py'),*common,'--intervention',mode,'--out',str(work/f'results/probe_40_{mode}.json')]))
    tasks.append(('probe_40_screen',[sys.executable,str(source/'run_probe.py'),*common,'--dataset','screen','--out',str(work/'results/probe_40_screen.json')]))
    for arm in ARMS:
        if arm!='baseline':tasks.append(('arm_'+arm,[sys.executable,str(source/'run_arm.py'),*common,'--arm',arm,'--out',str(work/'results'/arm)]))
else:
    for arm in a.arms:tasks.append(('confirm_'+arm,[sys.executable,str(source/'run_arm.py'),*common,'--arm',arm,'--out',str(work/'results'/arm),'--resume','--targets','3000000']))
    tasks.append(('probe_40_audit',[sys.executable,str(source/'run_probe.py'),*common,'--dataset','audit','--out',str(work/'results/probe_40_audit.json')]))
cpus=queue.Queue()
for cpu in [0,1,2]:cpus.put(cpu)
def run(task):
    name,command=task;cpu=cpus.get()
    try:
        with (work/f'{name}.log').open('w') as log:
            result=subprocess.run([*command,'--cpu',str(cpu)],stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps(dict(task=name,exit_code=result.returncode)),flush=True)
        return result.returncode
    finally:cpus.put(cpu)
with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,tasks))
if any(results):raise SystemExit(1)
