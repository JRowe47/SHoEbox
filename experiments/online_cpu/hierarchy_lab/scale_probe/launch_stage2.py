import concurrent.futures,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
tasks=[('allocation',s,n) for s in ['1m','5m'] for n in ['uniform','rebalanced']]+[('fresh',s,n) for s in ['1m','5m'] for n in ['baseline','route_center_full','source_center','separated_ports']]
def worker(slot):
 for kind,size,name in tasks[slot::8]:
  e=(32 if size=='1m' else 128) if kind=='allocation' else (12 if size=='1m' else 64)
  script=ROOT.parent/'allocation_probe/experiment.py' if kind=='allocation' else ROOT/'experiment.py'
  total=2000000 if kind=='allocation' else 1000000
  env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','SHOEBOX_EXPERTS':str(e),'PROBE_SIZE':size,'PROBE_CPU':str(slot)}
  log=ROOT/'results'/f'stage2_{kind}_{size}_{name}.log'
  with log.open('w') as f:r=subprocess.run(['python',str(script),name,'--targets',str(total)]+(['--fresh'] if kind=='fresh' else []),env=env,stdout=f,stderr=subprocess.STDOUT)
  print(json.dumps({'kind':kind,'size':size,'profile':name,'code':r.returncode}),flush=True)
  assert r.returncode==0,log
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(8)))
