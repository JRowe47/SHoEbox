"""E6 paired development pilots: locally supervised versus generic gate."""
import argparse,hashlib,os
from pathlib import Path
from datetime import datetime,timezone
from .adapter import Adapter,PARAMETERS,ROOT
from .tasks import dataset,balanced_offsets
from experiments.joint_readout.stream import new_cursor
from experiments.joint_readout.run import fit,source_hashes
from experiments.joint_readout.regressions import Corpus,language
from experiments.constituent_readout.stage3 import audits
from experiments.query_readout.run import write

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--arm',choices=('local','generic'),required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if (a.out/'plan.json').exists():raise FileExistsError('registered run exists')
 sources=source_hashes();sources.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*') if p.suffix in ('.py','.hpp','.cpp')})
 write(a.out/'plan.json',dict(utc=datetime.now(timezone.utc).isoformat(),seed=7,arm=a.arm,binding_bytes=2250000,language_bytes=1000000,source_hashes=sources,protocol_sha256=hashlib.sha256((ROOT/'reports/exp01e/AMENDMENT_E6.md').read_bytes()).hexdigest(),audit_opened=False))
 tr=dataset(2048,'train');dv=balanced_offsets(dataset(512,'dev'));row={}
 with Adapter(parameters=500000-PARAMETERS,seed=7,kind='native',joint=True) as b:
  row['warmup'],_=fit(b,tr,dv,250000,250000,new_cursor(701,local=False))
  with Adapter(seed=7,kind='paired_local' if a.arm=='local' else 'paired_gated') as m:
   m.adopt(b);m.reset();row['binding'],cursor=fit(m,tr,dv,2000000,500000,new_cursor(all_bytes=True,local=True))
   (a.out/'binding.qr').write_bytes(m.dumps());write(a.out/'binding.cursor.json',cursor);write(a.out/'train.json',row)
   row['binding']['development_audit']=audits(m,dataset(512,'dev'));write(a.out/'train.json',row)
   import experiments.joint_readout.regressions as reg
   reg.Adapter=Adapter;reg.binding=dataset
   row['retention']=language(m,Corpus(a.data),a.out/'retained.qr',True);write(a.out/'train.json',row)
   row['retention']['development_audit']=audits(m,dataset(512,'dev'));write(a.out/'result.json',row)
   print(a.arm,row['binding']['curve'][-1],row['retention']['curve'][-1],flush=True)
if __name__=='__main__':main()
