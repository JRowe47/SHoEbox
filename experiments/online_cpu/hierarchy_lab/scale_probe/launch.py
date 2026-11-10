import argparse,concurrent.futures,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
names=['baseline','context0','context05','context10','route_center','route_center_full','source_center','route_reset','source_center_reset']
p=argparse.ArgumentParser();p.add_argument('--targets',type=int,default=200000);p.add_argument('--names',default=','.join(names));p.add_argument('--fresh',action='store_true');a=p.parse_args()
tasks=[(e,name) for name in a.names.split(',') for e in [12,64]]
def worker(slot):
 outputs=[]
 for e,name in tasks[slot::8]:
  label=f'{"fresh" if a.fresh else "continued"}_e{e}_{name}_{a.targets}'
  env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','SHOEBOX_EXPERTS':str(e),'PROBE_CPU':str(slot)}
  log=ROOT/'results'/f'{label}.log';log.parent.mkdir(exist_ok=True)
  with log.open('w') as f:
   cmd=['python',str(ROOT/'experiment.py'),name,'--targets',str(a.targets)]+(['--fresh'] if a.fresh else [])
   r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  record={'run':label,'code':r.returncode,'log':str(log)};print(json.dumps(record),flush=True);outputs.append(record)
 return outputs
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 outcomes=[r for rows in pool.map(worker,range(8)) for r in rows]
assert all(r['code']==0 for r in outcomes),outcomes
