"""Registered development-only rate/horizon follow-up to the initial pilot."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
from .adapter import Adapter
from .tasks import Spec,dataset
from experiments.query_readout.run import train,write

def job(args):
    path,rate,cpu=args
    os.sched_setaffinity(0,{cpu});spec=Spec('binding',vocab=2)
    tr=dataset(spec,2048,'train');dv=dataset(spec,256,'dev')
    with Adapter.loads((path/'warmup.qr').read_bytes()) as base,Adapter(seed=7,rate=rate,joint=False) as m:
        m.adopt(base);row=train(m,tr,dv,2000000,500000,602)
        (path/f'rate_{rate}.qr').write_bytes(m.dumps());write(path/f'rate_{rate}.json',row)
        print(rate,[(x['bytes'],x['dev']['native_rollout']['native_exact']) for x in row['curve']],flush=True)
        return {'rate':rate,**row}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rates=[.04,.08,.16,.32];cpus=sorted(os.sched_getaffinity(0))[:4]
    write(a.out/'rates_plan.json',dict(rates=rates,bytes_per_arm=2000000,seed=7,core='minimal',test_access=False))
    with ProcessPoolExecutor(max_workers=4) as ex:rows=list(ex.map(job,[(a.out,r,c) for r,c in zip(rates,cpus)]))
    write(a.out/'rates.json',rows)
