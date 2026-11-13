"""Alternating same-core old/new production timing and diagnostic overhead."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def worker(a):
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    os.sched_setaffinity(0,{a.core})
    sys.path.insert(0,str(a.runtime))
    import numpy as np
    import time
    from runtime import Model
    original=Model.load(a.checkpoint)
    if a.condition=='diagnostic':m=original.evaluation_copy(diagnostic=True)
    elif a.mode=='periodic':m=original.snapshot_copy()
    else:m=original.frozen_copy()
    data=np.fromfile(a.data,np.uint8)[:17409]
    m.run(data[:1024],data[1:1025],mode=a.mode,lr=.008,lp=.008/3,route=.002)
    begin=time.perf_counter()
    nll=m.run(data[1024:-1],data[1025:],mode=a.mode,lr=.008,lp=.008/3,route=.002)
    seconds=time.perf_counter()-begin
    print(json.dumps(dict(condition=a.condition,mode=a.mode,targets=16384,seconds=seconds,bytes_per_second=16384/seconds,nll=nll)))
    m.close();original.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-runtime',type=Path)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'results/relay_full_relays_seed1/last.npz')
    p.add_argument('--data',type=Path,default=ROOT/'data/train_segment.txt')
    p.add_argument('--core',type=int,default=min(os.sched_getaffinity(0)))
    p.add_argument('--worker',action='store_true')
    p.add_argument('--runtime',type=Path)
    p.add_argument('--condition',choices=['before','after','diagnostic'])
    p.add_argument('--mode',choices=['eval','periodic'])
    a=p.parse_args()
    if a.worker:return worker(a)
    if a.baseline_runtime is None:p.error('--baseline-runtime must point to the unchanged f9eeb8a runtime')
    conditions=[('before','periodic'),('after','periodic'),('before','eval'),('after','eval'),('diagnostic','eval')]
    rows=[]
    for trial in range(6):
        for condition,mode in conditions if trial%2==0 else reversed(conditions):
            runtime=a.baseline_runtime if condition=='before' else ROOT/'runtime'
            command=[sys.executable,str(Path(__file__).resolve()),'--worker','--runtime',str(runtime.resolve()),
                     '--checkpoint',str(a.checkpoint.resolve()),'--data',str(a.data.resolve()),
                     '--condition',condition,'--mode',mode,'--core',str(a.core)]
            result=subprocess.run(command,check=True,text=True,capture_output=True)
            row=json.loads(result.stdout);row['trial']=trial;rows.append(row)
            print(condition,mode,trial,round(row['bytes_per_second']),flush=True)
    medians={f'{condition}_{mode}':statistics.median(r['bytes_per_second'] for r in rows if r['condition']==condition and r['mode']==mode)
             for condition,mode in conditions}
    for mode in ('periodic','eval'):
        losses=[r['nll'] for r in rows if r['mode']==mode]
        assert max(losses)==min(losses),'Production/disabled-diagnostic numerical drift'
    result=dict(core=a.core,trials=rows,median_bytes_per_second=medians,
                production_training_ratio=medians['after_periodic']/medians['before_periodic'],
                production_evaluation_ratio=medians['after_eval']/medians['before_eval'],
                diagnostic_evaluation_ratio=medians['diagnostic_eval']/medians['after_eval'],
                includes='end-to-end execution; training also includes dictionaries, memory, and neural updates',
                excludes='compilation, load/copy, warmup, checkpoint I/O')
    (ROOT/'results/contribution_throughput.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
