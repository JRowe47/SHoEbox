import concurrent.futures,json,os,queue,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
tasks=queue.Queue()
for size in ['1m','5m']:
 for phase,names in [('continued',['baseline','context05','route_center','route_center_full','source_center','route_reset']),('allocation',['uniform','rebalanced']),('fresh1m',['baseline','route_center','raw_full','route_center_full','source_center','separated_ports']),('fresh3m',['baseline','route_center','raw_full','route_center_full','source_center'])]:
  for name in names:
   result=ROOT/'results/evaluation'/f'{phase}_{size}_{name}.json'
   if not result.exists():tasks.put((size,phase,name))
def worker(cpu):
 while True:
  try:size,phase,name=tasks.get_nowait()
  except queue.Empty:return
  e=(32 if size=='1m' else 128) if phase=='allocation' else (12 if size=='1m' else 64)
  env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','SHOEBOX_EXPERTS':str(e),'PROBE_SIZE':size,'PROBE_CPU':str(cpu)}
  r=subprocess.run(['python',str(ROOT/'measure.py'),phase,name],env=env,capture_output=True,text=True)
  (ROOT/'results'/f'measure_{phase}_{size}_{name}.log').write_text(r.stdout+r.stderr)
  print(r.stdout if r.returncode==0 else r.stderr,flush=True);assert r.returncode==0,(size,phase,name)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(8)))
