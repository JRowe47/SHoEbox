"""Alternating same-core training timing after all study workers finish."""
import argparse,json,os,statistics,sys,time
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
from runtime import Model
p=argparse.ArgumentParser();p.add_argument('--core',type=int,default=min(os.sched_getaffinity(0)));a=p.parse_args();os.sched_setaffinity(0,{a.core})
models={k:Model.load(ROOT/f'results/full_{k}_seed1/last.npz') for k in ['binding','semantics_fixed','semantics']}
data=np.fromfile(ROOT/'data/train_segment.txt',np.uint8)[:17409];rows=[]
for trial in range(6):
 for k in (['binding','semantics_fixed','semantics'] if trial%2==0 else ['semantics','semantics_fixed','binding']):
  m=models[k].snapshot_copy();m.run(data[:1024],data[1:1025],mode='periodic',lr=.008,lp=.008/3,route=.002)
  t=time.perf_counter();loss=m.run(data[1024:-1],data[1025:],mode='periodic',lr=.008,lp=.008/3,route=.002);seconds=time.perf_counter()-t
  row=dict(kernel=k,trial=trial,targets=16384,seconds=seconds,bytes_per_second=16384/seconds,loss=loss);rows.append(row);print(k,trial,round(row['bytes_per_second']),flush=True);m.close()
med={k:statistics.median(r['bytes_per_second'] for r in rows if r['kernel']==k) for k in models}
r=dict(core=a.core,trials=rows,median_bytes_per_second=med,semantics_over_binding_ratio=med['semantics']/med['binding'],extra_seconds_per_million=1e6/med['semantics']-1e6/med['binding'],includes='forward, routing, dictionary/memory updates, neural learning',excludes='compilation, snapshots, checkpoint I/O, validation')
(ROOT/'results/throughput.json').write_text(json.dumps(r,indent=2)+'\n')
for m in models.values():m.close()
