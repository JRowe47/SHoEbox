"""Execute the recorded validation-only confirmation plan."""
import concurrent.futures,json,os,queue,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
plan=json.loads((ROOT/'confirmation_plan.json').read_text());tasks=queue.Queue()
# Interleave sizes and seeds so partial progress includes both model scales.
for seed in [1,2,3]:
 for size in plan['sizes']:
  for name,seeds in plan['replication'].items():
   if seed in seeds:tasks.put((size,seed,name))
def worker(cpu):
 while True:
  try:size,seed,name=tasks.get_nowait()
  except queue.Empty:return
  env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','SHOEBOX_EXPERTS':'12' if size=='1m' else '64','PROBE_CPU':str(cpu)}
  log=ROOT/'results'/f'confirm_{size}_s{seed}_{name}.log'
  with log.open('w') as f:r=subprocess.run(['python',str(ROOT/'experiment.py'),name,'--seed',str(seed),'--targets',str(plan['targets'])],env=env,stdout=f,stderr=subprocess.STDOUT)
  print(json.dumps(dict(size=size,seed=seed,profile=name,code=r.returncode)),flush=True);assert r.returncode==0,str(log)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(8)))
