import json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for size in ['1m','5m']:
 for phase,names in [('continued','baseline,context05,route_center,route_center_full,source_center,route_reset'),('fresh3m','baseline,route_center,raw_full,route_center_full,source_center'),('allocation','uniform,rebalanced'),('fresh1m','baseline,separated_ports')]:
  e=(32 if size=='1m' else 128) if phase=='allocation' else (12 if size=='1m' else 64)
  env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','SHOEBOX_EXPERTS':str(e),'PROBE_SIZE':size,'PROBE_CPU':'8'}
  out=ROOT/'results'/f'benchmark_{phase}_{size}.log'
  with out.open('w') as f:r=subprocess.run(['python',str(ROOT/'measure.py'),phase,names,'--benchmark'],env=env,stdout=f,stderr=subprocess.STDOUT)
  print(json.dumps({'phase':phase,'size':size,'code':r.returncode}),flush=True);assert r.returncode==0,out
