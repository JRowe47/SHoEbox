"""Registered EXP01-D development-only pilot; no audit access."""
import argparse,hashlib,json,time
from pathlib import Path
from datetime import datetime,timezone
from .adapter import Adapter,PARAMETERS,ROOT,HERE
from .tasks import dataset,balanced_offsets,SALT
from experiments.query_access.run import train,evaluate
from experiments.query_readout.run import write

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 files=[ROOT/'reports/exp01d/PROTOCOL.md',*HERE.glob('*.py'),*HERE.glob('*.hpp'),*HERE.glob('*.cpp')]
 plan=dict(stage='D1',registered=datetime.now(timezone.utc).isoformat(),seed=7,warmup=250000,continuation=500000,head_rate=.16,
   auxiliary=1.,fold_salt=SALT.decode(),parameters=500000,readout_parameters=PARAMETERS,
   hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
 if (a.out/'plan.json').exists():raise FileExistsError('pilot exists')
 write(a.out/'plan.json',plan)
 tr=dataset(2048,'train');dv=balanced_offsets(dataset(512,'dev'))
 with Adapter(parameters=500000-PARAMETERS,seed=7,kind='native',joint=True) as b:
  warm=train(b,tr,dv,250000,250000,701);(a.out/'warm.qr').write_bytes(b.dumps());write(a.out/'warmup.json',warm)
  with Adapter(seed=7,kind='paired') as m:
   m.adopt(b);r=train(m,tr,dv,500000,100000,702);(a.out/'paired.qr').write_bytes(m.dumps());r['inventory']=m.inventory();write(a.out/'result.json',r)
   for c in r['curve']:print(c,flush=True)
if __name__=='__main__':main()
