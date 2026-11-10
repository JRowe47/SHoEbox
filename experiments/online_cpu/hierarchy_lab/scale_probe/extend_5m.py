import concurrent.futures,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def run(item):
 name,cpu=item;meta=ROOT/'results/fresh/5m'/name/'training.json'
 while not meta.exists():time.sleep(5)
 while json.loads(meta.read_text())['targets']<1000000:time.sleep(5)
 env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','SHOEBOX_EXPERTS':'64','PROBE_CPU':str(cpu)}
 with (ROOT/'results'/f'extension_5m_{name}.log').open('w') as f:r=subprocess.run(['python',str(ROOT/'experiment.py'),name,'--fresh','--targets','3000000'],env=env,stdout=f,stderr=subprocess.STDOUT)
 print(json.dumps({'profile':name,'code':r.returncode}),flush=True);assert r.returncode==0
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,[('baseline',0),('route_center_full',1),('source_center',2)]))
