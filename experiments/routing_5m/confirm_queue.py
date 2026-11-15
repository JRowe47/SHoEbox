"""Reuse each CPU only after its screening run has completed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args()
screen=['D','C','B','A','D_no_balance','D_warmup_only','D_L1_only','D_high_rate','C_warmup_only']
arms=['D','C','D_warmup_only','C_warmup_only']
todo=[(arm,seed) for seed in (7,1,19) for arm in arms]
active={};released=set();results=[]
def completed(arm,seed,n):
    path=a.work/f'{arm}_s{seed}/results.json'
    if not path.exists():return False
    d=json.loads(path.read_text())
    return d.get('status')=='complete' and d['targets']>=n
while todo or active:
    for cpu in range(9):
        if cpu in active:continue
        if cpu not in released:
            if not completed(screen[cpu],7,10_000_000):continue
            released.add(cpu)
        ready=next(((arm,seed) for arm,seed in todo if seed!=7 or completed(arm,seed,10_000_000)),None)
        if ready is None:continue
        arm,seed=ready;todo.remove(ready);out=a.work/f'{arm}_s{seed}';out.mkdir(parents=True,exist_ok=True)
        log=(out/'confirmation.log').open('a')
        cmd=[sys.executable,str(Path(__file__).with_name('run.py')),'--work',str(a.work),'--data',str(a.data),
            '--arm',arm,'--seed',str(seed),'--cpu',str(cpu),'--targets','20000000']
        if seed==7:cmd+=['--resume']
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        active[cpu]=(proc,log,arm,seed)
        print(json.dumps(dict(started=arm,seed=seed,cpu=cpu,pid=proc.pid)),flush=True)
    time.sleep(1)
    for cpu,(proc,log,arm,seed) in list(active.items()):
        code=proc.poll()
        if code is not None:
            log.close();del active[cpu];results.append(code)
            print(json.dumps(dict(finished=arm,seed=seed,code=code)),flush=True)
assert not any(results),results
