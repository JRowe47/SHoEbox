"""Development-only pilot; never loads the audit fold."""
import argparse
from pathlib import Path
from .adapter import Adapter,PARAMETERS
from .tasks import dataset,Spec
from experiments.query_readout.run import train,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    spec=Spec('binding',vocab=2,key_pool=48);tr=dataset(spec,2048,'train');dv=dataset(spec,256,'dev')
    plan=dict(seed=7,parameters=500000,warmup=250000,continuation=500000,rates=[.16,.5,1.],arms=['local_frozen','local_joint','answer_frozen'],test_access=False)
    write(a.out/'pilot_plan.json',plan);result={'plan':plan,'arms':{}}
    with Adapter(parameters=500000-PARAMETERS,seed=7,kind='native') as base:
        result['warmup']=train(base,tr,dv,250000,250000,601)
        (a.out/'warmup.qr').write_bytes(base.dumps())
        for arm,rate in [('local_frozen',r) for r in plan['rates']]+[('local_joint',.5),('answer_frozen',.5)]:
            with Adapter(seed=7,kind=arm.split('_')[0],joint=arm.endswith('joint'),rate=rate) as m:
                m.adopt(base);key=f'{arm}_{rate}'
                result['arms'][key]=train(m,tr,dv,500000,125000,602)
                (a.out/f'{key}.qr').write_bytes(m.dumps());write(a.out/'pilot.json',result)
                print(key,[(x['bytes'],x['dev']['native_rollout']['native_exact']) for x in result['arms'][key]['curve']],flush=True)

if __name__=='__main__':main()
