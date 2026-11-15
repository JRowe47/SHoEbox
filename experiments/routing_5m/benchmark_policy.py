import argparse,importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--selected-root',type=Path,required=True);p.add_argument('--original-root',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--cpu',type=int,default=8);a=p.parse_args();os.sched_setaffinity(0,{a.cpu});sys.path.insert(0,str(a.selected_root))
from shoe_multiscale import Model
spec=importlib.util.spec_from_file_location('original_runtime',a.original_root/'multiscale/model.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
meta=json.load(open(a.data/'manifest.json'));data=np.fromfile(a.data/'train.bin',np.uint8)
segments=[(x,min(x+4096,d['stop']-1)) for d in meta['files']['train']['documents'] for x in range(d['start'],d['stop']-1,4096)]
rows=[];arms=['D','C','D_warmup_only','C_warmup_only']
for k,seed in enumerate([1,7,19]):
 for arm in (arms if k%2==0 else list(reversed(arms))):
  cls=Model if 'warmup' in arm else old.Model
  with cls.load(a.work/f'{arm}_s{seed}/latest.shoe') as m:
   if 'warmup' in arm:m.set_balance_policy('warmup')
   m.configure(fast_rate=.0002,slow_rate=.002,query_rate=.0002,head_rate=.004)
   done=0;t=time.perf_counter();c=time.process_time()
   for idx in np.random.default_rng(seed).permutation(len(segments)):
    x,y=segments[idx];y=min(y,x+200000-done);m.reset_context(clear_memory=True);m.run(data[x:y],data[x+1:y+1]);done+=y-x
    if done==200000:break
   sec=time.perf_counter()-t;cpu=time.process_time()-c
  row=dict(arm=arm,seed=seed,targets=done,seconds=sec,cpu_seconds=cpu,bytes_per_second=done/sec,cpu_bytes_per_second=done/cpu)
  rows.append(row);print(json.dumps(row),flush=True)
summary={arm:dict(median_bps=float(np.median([r['bytes_per_second'] for r in rows if r['arm']==arm])),median_cpu_bps=float(np.median([r['cpu_bytes_per_second'] for r in rows if r['arm']==arm]))) for arm in arms}
(a.work/'serial_benchmark.json').write_text(json.dumps(dict(rows=rows,summary=summary,cpu=a.cpu,targets_per_repeat=200000,notes='Original production native code for v9 controls; selected production native code for warmup-only. All training, probes and tests stopped. File load, compilation and checkpointing excluded.'),indent=2));print(json.dumps(summary,indent=2))
