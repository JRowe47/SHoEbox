import concurrent.futures,json,os,queue,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
plan=json.loads((ROOT/'confirmation_plan.json').read_text());assert (ROOT/'selection.json').exists()
tasks=queue.Queue()
for size in plan['sizes']:
 for profile,seeds in plan['replication'].items():
  for seed in seeds:tasks.put((size,seed,profile))
for job in plan.get('extra_jobs',[]):tasks.put((job['size'],job['seed'],job['profile']))
def worker(cpu):
 while True:
  try:size,seed,profile=tasks.get_nowait()
  except queue.Empty:return
  env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','SHOEBOX_EXPERTS':'12' if size=='1m' else '64','PROBE_CPU':str(cpu)}
  proc=subprocess.run(['python',str(ROOT/'measure.py'),profile,'--seed',str(seed)],env=env,capture_output=True,text=True)
  (ROOT/'results'/f'eval_{size}_s{seed}_{profile}.log').write_text(proc.stdout+proc.stderr)
  assert proc.returncode==0,proc.stderr;print(proc.stdout,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(8)))
