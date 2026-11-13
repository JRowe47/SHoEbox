"""Interleaved, same-core training timings for the released implementation."""
import os,sys,json,time,statistics,hashlib,argparse
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--core',type=int,default=min(os.sched_getaffinity(0)))
args=parser.parse_args()
os.sched_setaffinity(0,{args.core})
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
from runtime import Model,engine
data=np.fromfile(ROOT/'data/train_segment.txt',np.uint8)[:32769]
result=dict(core=args.core,targets_per_trial=32768,trials=[],timing_includes='native forward and all training updates',timing_excludes='loading, snapshots, compilation, evaluation, reporting')
baseline=Model.load(ROOT/'results/native_seed1/last.npz');repaired=Model.load(ROOT/'results/delivery_composition_seed1/last.npz')
result['checkpoints']=['results/native_seed1/last.npz','results/delivery_composition_seed1/last.npz']
result['checkpoint_sha256']=[hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in result['checkpoints']]
result['binary_sha256']={m.runtime_kernel:hashlib.sha256(Path(m.lib._name).read_bytes()).hexdigest() for m in [baseline,repaired]}
for iteration in range(6):
    order=('native','composition') if iteration%2==0 else ('composition','native')
    for name in order:
        m=(baseline if name=='native' else repaired).snapshot_copy()
        m.run(data[:1024],data[1:1025],mode='periodic',lr=.008,lp=.008/3)
        start=time.perf_counter();loss=m.run(data[:-1],data[1:],mode='periodic',lr=.008,lp=.008/3);elapsed=time.perf_counter()-start
        result['trials'].append(dict(iteration=iteration,kernel=name,seconds=elapsed,bytes_s=32768/elapsed,nll=float(loss)))
        print(name,iteration,round(32768/elapsed),flush=True);m.close()
result['median_bytes_s']={name:statistics.median(row['bytes_s'] for row in result['trials'] if row['kernel']==name) for name in ('native','composition')}
result['throughput_ratio']=result['median_bytes_s']['composition']/result['median_bytes_s']['native']
result['extra_seconds_per_million']=1e6/result['median_bytes_s']['composition']-1e6/result['median_bytes_s']['native']
(ROOT/'results/throughput_final.json').write_text(json.dumps(result,indent=2)+'\n');baseline.close();repaired.close()
