"""Reproduce the fixed development-only RLS pilot after pilot.py warmup."""
import argparse
from pathlib import Path
from .adapter import Adapter
from .tasks import Spec,dataset
from experiments.query_readout.run import train,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    write(a.out/'rls_plan.json',dict(seed=7,warmup_reused=250000,kind='rls',rate=.16,budget=500000,interval=125000,test_access=False))
    tr=dataset(Spec('binding',vocab=2),2048,'train');dv=dataset(Spec('binding',vocab=2),256,'dev')
    with Adapter.loads((a.out/'warmup.qr').read_bytes()) as base,Adapter(seed=7,kind='rls',joint=False,rate=.16) as m:
        m.adopt(base);result=train(m,tr,dv,500000,125000,602)
        write(a.out/'rls.json',result);(a.out/'rls.qr').write_bytes(m.dumps())

if __name__=='__main__':main()
