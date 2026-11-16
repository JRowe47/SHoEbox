"""Separately charged development-only forgetting diagnosis."""
import argparse,hashlib
from pathlib import Path
from datetime import datetime,timezone
from .adapter import Adapter,ROOT
from .regressions import Corpus,language
from .run import source_hashes
from .tasks import dataset
from experiments.constituent_readout.stage3 import audits
from experiments.query_readout.run import write

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if (a.out/'plan.json').exists():raise FileExistsError('registered run exists')
 plan=dict(utc=datetime.now(timezone.utc).isoformat(),seed=7,bytes=1000000,source_hashes=source_hashes(),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),amendment_sha256=hashlib.sha256((ROOT/'reports/exp01e/RETENTION_PILOT.md').read_bytes()).hexdigest(),data_manifest_sha256=hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest(),audit_opened=False)
 write(a.out/'plan.json',plan)
 with Adapter.loads(a.checkpoint.read_bytes()) as m:
  r=language(m,Corpus(a.data),a.out/'retained.qr',True);write(a.out/'train.json',r)
  r['development_audit']=audits(m,dataset(512,'dev'));write(a.out/'result.json',r)
  print(r['curve'][-1],flush=True)
if __name__=='__main__':main()
