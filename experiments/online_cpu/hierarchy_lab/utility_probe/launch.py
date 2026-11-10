import argparse,concurrent.futures,json,os,queue,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--names',required=True);p.add_argument('--sizes',default='1m');p.add_argument('--seeds',default='1');p.add_argument('--targets',type=int,default=300000);p.add_argument('--workers',type=int,default=8);p.add_argument('--cpu-offset',type=int,default=0);a=p.parse_args()
tasks=queue.Queue()
for size in a.sizes.split(','):
 for seed in map(int,a.seeds.split(',')):
  for name in a.names.split(','):tasks.put((size,seed,name))
def worker(cpu):
 while True:
  try:size,seed,name=tasks.get_nowait()
  except queue.Empty:return
  env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','SHOEBOX_EXPERTS':'12' if size=='1m' else '64','PROBE_CPU':str(cpu)}
  log=ROOT/'results'/f'{size}_s{seed}_{name}_{a.targets}.log';log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('w') as f:proc=subprocess.run(['python',str(ROOT/'experiment.py'),name,'--seed',str(seed),'--targets',str(a.targets)],env=env,stdout=f,stderr=subprocess.STDOUT)
  print(json.dumps(dict(size=size,seed=seed,profile=name,targets=a.targets,code=proc.returncode)),flush=True)
  assert proc.returncode==0,str(log)
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(worker,range(a.cpu_offset,a.cpu_offset+a.workers)))
