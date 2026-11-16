"""Explicit E2-to-E3 weight transfer, then development-only language retention."""
import argparse,hashlib,struct
from pathlib import Path
from datetime import datetime,timezone
from .adapter import Adapter,ROOT
from .tasks import dataset,balanced_offsets
from experiments.joint_readout.run import source_hashes
from experiments.joint_readout.regressions import Corpus,language
from experiments.constituent_readout.stage3 import audits
from experiments.query_readout.run import write

def convert(raw):
 if hashlib.sha256(raw[32:]).digest()!=raw[:32]:raise ValueError('source checksum')
 payload=bytearray(raw[32:])
 if struct.unpack_from('<Q',payload)[0]!=0x32455041514853:raise ValueError('explicit E2 source required')
 struct.pack_into('<Q',payload,0,0x33455041514853)
 return hashlib.sha256(payload).digest()+bytes(payload)
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if (a.out/'plan.json').exists():raise FileExistsError('registered run exists')
 raw=a.source.read_bytes();converted=convert(raw);sources=source_hashes();sources.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*') if p.suffix in ('.py','.hpp','.cpp')})
 plan=dict(utc=datetime.now(timezone.utc).isoformat(),phase='E3-pilot-A',seed=7,inherited_binding_bytes=2250000,new_training_bytes=1000000,source_hashes=sources,source_checkpoint_sha256=hashlib.sha256(raw).hexdigest(),converted_checkpoint_sha256=hashlib.sha256(converted).hexdigest(),amendment_sha256=hashlib.sha256((ROOT/'reports/exp01e/AMENDMENT_E3.md').read_bytes()).hexdigest(),audit_opened=False)
 write(a.out/'plan.json',plan);(a.out/'converted.qr').write_bytes(converted)
 import experiments.joint_readout.regressions as reg
 reg.Adapter=Adapter;reg.binding=dataset
 with Adapter.loads(converted) as m:
  row=dict(before=audits(m,dataset(512,'dev')));write(a.out/'before.json',row)
  row['retention']=language(m,Corpus(a.data),a.out/'retained.qr',True);write(a.out/'train.json',row)
  row['after']=audits(m,dataset(512,'dev'));write(a.out/'result.json',row);print(row['retention']['curve'][-1],flush=True)
if __name__=='__main__':main()
