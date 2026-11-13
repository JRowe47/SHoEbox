"""Alternating same-core training timing after all study workers finish."""
import argparse,json,os,statistics,sys,time
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
from runtime import Model
p=argparse.ArgumentParser();p.add_argument('--core',type=int,default=min(os.sched_getaffinity(0)));a=p.parse_args();os.sched_setaffinity(0,{a.core})
models={k:Model.load(ROOT/f'results/full_{k}_seed1/last.npz') for k in ['composition','binding']}
data=np.fromfile(ROOT/'data/train_segment.txt',np.uint8)[:33793];rows=[]
for trial in range(6):
 for k in (['composition','binding'] if trial%2==0 else ['binding','composition']):
  m=models[k].snapshot_copy();m.run(data[:1024],data[1:1025],mode='periodic',lr=.008,lp=.008/3,route=.002)
  t=time.perf_counter();loss=m.run(data[1024:-1],data[1025:],mode='periodic',lr=.008,lp=.008/3,route=.002);seconds=time.perf_counter()-t
  row=dict(kernel=k,trial=trial,targets=32768,seconds=seconds,bytes_per_second=32768/seconds,loss=loss);rows.append(row);print(k,trial,round(row['bytes_per_second']),flush=True);m.close()
med={k:statistics.median(r['bytes_per_second'] for r in rows if r['kernel']==k) for k in models}
r=dict(core=a.core,trials=rows,median_bytes_per_second=med,binding_over_baseline_ratio=med['binding']/med['composition'],extra_seconds_per_million=1e6/med['binding']-1e6/med['composition'],includes='forward, routing, dictionary/memory updates, neural learning',excludes='compilation, snapshots, checkpoint I/O, validation')
(ROOT/'results/throughput.json').write_text(json.dumps(r,indent=2)+'\n')
for m in models.values():m.close()
