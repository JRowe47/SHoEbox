import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
tasks=[]
for e in [12,64]:
 tasks.append((e,'check_exact.py',['--kernel','cached','--targets','100000'],{}))
 tasks.append((e,'verify_runtime.py',[],{}))
 tasks.append((e,'check_exact.py',['--kernel','cached','--targets','32768','--label','avx2'],{'SHOEBOX_CXX_FLAGS':'-mno-avx512f'}))
tasks.append((64,'check_masks.py',[],{'SHOEBOX_CXX_FLAGS':'-mno-avx512f'}))
for e in [12,64]:
 for kernel in ['profile','profile_cached']:tasks.append((e,'bench.py',['--kernel',kernel,'--targets','65536','--repeats','2'],{}))
for e in [12,64]:tasks.append((e,'overlap_bench.py',[],{}))
for e in [12,64]:tasks.append((e,'parallel_bench.py',[],{}))
for i,(e,script,args,extra) in enumerate(tasks):
 env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','SHOEBOX_EXPERTS':str(e),**extra}
 name=f'job_{i:02d}_e{e}_{Path(script).stem}';print('START',name,args,flush=True);start=time.time()
 with (ROOT/'results'/f'{name}.log').open('w') as f:r=subprocess.run(['python',str(ROOT/script),*args],env=env,stdout=f,stderr=subprocess.STDOUT)
 print('END',name,'code',r.returncode,'seconds',round(time.time()-start,1),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
