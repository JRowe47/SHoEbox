"""Same-core alternating timing; run after training and other CPU checks exit."""
import argparse,json,os,statistics,sys,time
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
from runtime import Model

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--core',type=int,default=min(os.sched_getaffinity(0)));a=p.parse_args()
 os.sched_setaffinity(0,{a.core})
 models={k:Model.load(ROOT/f'results/relay_full_{k}_seed1/last.npz') for k in ('semantics','relays')}
 data=np.fromfile(ROOT/'data/train_segment.txt',np.uint8)[:17409];rows=[]
 for trial in range(6):
  for k in (('semantics','relays') if trial%2==0 else ('relays','semantics')):
   m=models[k].snapshot_copy()
   m.run(data[:1024],data[1:1025],mode='periodic',lr=.008,lp=.008/3,route=.002)
   start=time.perf_counter()
   loss=m.run(data[1024:-1],data[1025:],mode='periodic',lr=.008,lp=.008/3,route=.002)
   sec=time.perf_counter()-start
   rows.append(dict(kernel=k,trial=trial,targets=16384,seconds=sec,bytes_per_second=16384/sec,loss=loss))
   print(k,trial,round(16384/sec),flush=True);m.close()
 med={k:statistics.median(r['bytes_per_second'] for r in rows if r['kernel']==k) for k in models}
 result=dict(core=a.core,trials=rows,median_bytes_per_second=med,relays_over_semantics_ratio=med['relays']/med['semantics'],
             includes='forward, routing, dictionary/memory updates, neural learning',excludes='compilation, snapshots, checkpoint I/O, validation')
 (ROOT/'results/relay_throughput.json').write_text(json.dumps(result,indent=2)+'\n')
 for m in models.values():m.close()

if __name__=='__main__':main()
