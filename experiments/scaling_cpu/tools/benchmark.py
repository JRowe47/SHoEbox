"""Matched single-core training throughput; does not advance comparison runs."""
from pathlib import Path
import argparse,hashlib,json,os,platform,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from shoebox_runtime import Architecture,Model
from shoebox_runtime.model import atomic_json
from comparison import architecture,load_fixed,runtime_hash


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=ROOT/'data/comparison/calibration.bin')
    p.add_argument('--output',type=Path,default=ROOT/'reports/throughput.json')
    p.add_argument('--targets',type=int,default=250000)
    p.add_argument('--warmup',type=int,default=16384)
    p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
    a=p.parse_args()
    if a.cpu not in os.sched_getaffinity(0):raise ValueError('Unavailable CPU')
    if a.targets<1 or a.warmup<0:raise ValueError('Invalid target budget')
    os.sched_setaffinity(0,{a.cpu})
    data=np.fromfile(a.data,np.uint8)[:a.targets+a.warmup+1]
    if len(data)!=a.targets+a.warmup+1:raise ValueError('Insufficient benchmark data')
    large=json.loads((ROOT/'configs/proposed_100m.json').read_text())
    arms=[(name,architecture(name)) for name in ('control_5m','experts_20m','width_20m')]
    arms.append(('width_100m',Architecture(large['width'],large['experts'])))
    cpu_model=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'unknown')
    result=dict(cpu_model=cpu_model,cpu=a.cpu,workers_per_model=1,concurrent_trainers=1,precision='float32',seed=3,
        warmup_targets=a.warmup,measured_targets=a.targets,chunk_targets=25000,mode='periodic',lr=.008,local_lr=.008/3,route_lr=.002,
        actual_update_observer=True,data_sha256=hashlib.sha256(data.tobytes()).hexdigest(),runtime_sha256=runtime_hash(),
        python=platform.python_version(),numpy=np.__version__,excluded_time=['initialization','native compilation','validation','checkpointing','data I/O'],
        method='Sequential fresh models on one pinned CPU; continuous warmup then non-overlapping measured chunks. Training only; no model-selection claim.',arms=[])
    for name,arch in arms:
        fixed=load_fixed(ROOT/'calibration',arch.width) if arch.width in (64,128) else None
        with Model(arch,seed=3,fixed=fixed) as m:
            m.observe_updates()
            if a.warmup:m.run(data[:a.warmup],data[1:a.warmup+1],mode='periodic')
            chunks=[]
            for begin in range(a.warmup,a.warmup+a.targets,25000):
                end=min(a.warmup+a.targets,begin+25000);start=time.perf_counter()
                loss=m.run(data[begin:end],data[begin+1:end+1],mode='periodic')
                seconds=time.perf_counter()-start
                chunks.append(dict(targets=end-begin,seconds=seconds,nll=loss,bytes_per_second=(end-begin)/seconds))
            seconds=sum(x['seconds'] for x in chunks)
            assert np.isfinite(m.flat).all() and np.isfinite(m.extra_flat).all()
            row=dict(arm=name,architecture=arch.as_dict(),parameters=m.parameter_count,training_seconds=seconds,
                bytes_per_second=a.targets/seconds,median_chunk_bytes_per_second=float(np.median([x['bytes_per_second'] for x in chunks])),
                projected_25m_training_seconds=25000000*seconds/a.targets,finite_parameters=True,chunks=chunks)
            result['arms'].append(row);atomic_json(a.output,result)
            print(json.dumps({k:v for k,v in row.items() if k not in ('architecture','chunks')}),flush=True)


if __name__=='__main__':main()
