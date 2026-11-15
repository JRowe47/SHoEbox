import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
p.add_argument('--arms',nargs='+',default=['D','C','B','A','D_no_balance','D_warmup_only','D_L1_only','D_high_rate'])
p.add_argument('--seeds',type=int,nargs='+',default=[7]);p.add_argument('--targets',type=int,default=10_000_000)
p.add_argument('--resume',action='store_true');a=p.parse_args()
todo=[(arm,seed) for seed in a.seeds for arm in a.arms];active={};results=[]
while todo or active:
    for cpu in range(8):
        if cpu in active or not todo:continue
        arm,seed=todo.pop(0);out=a.work/f'{arm}_s{seed}';out.mkdir(parents=True,exist_ok=True)
        log=(out/'run.log').open('a')
        cmd=[sys.executable,str(Path(__file__).with_name('run.py')),'--work',str(a.work),'--data',str(a.data),
             '--arm',arm,'--seed',str(seed),'--cpu',str(cpu),'--targets',str(a.targets)]
        if a.resume and (out/'latest.shoe').exists():cmd+=['--resume']
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
