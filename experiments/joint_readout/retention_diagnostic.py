"""Frozen weight grafts on development data only; no refitting or promotion."""
import argparse,hashlib,json,struct
from pathlib import Path
from .adapter import Adapter
from .tasks import dataset,balanced_offsets
from experiments.query_access.run import evaluate
from experiments.query_readout.run import write

def parts(raw):
 payload=raw[32:];n=struct.unpack_from('<Q',payload,12)[0];at=20+n;return payload,at

def graft(before,after,head=False,gate=False,core=False):
 b,bi=parts(before);a,ai=parts(after)
 p=bytearray(a)
 if core:
  if ai!=bi:raise ValueError('core size mismatch')
  p[20:ai]=b[20:bi]
 if head:p[ai:ai+39600]=b[bi:bi+39600]
 if gate:p[ai+39634:ai+39894]=b[bi+39634:bi+39894]
 return hashlib.sha256(p).digest()+bytes(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();old=a.before.read_bytes();new=a.after.read_bytes();cases=balanced_offsets(dataset(512,'dev'));rows={}
 for label,opts in [('after',{}),('old_head',dict(head=True)),('old_gate',dict(gate=True)),('old_head_gate',dict(head=True,gate=True)),('old_core',dict(core=True))]:
  with Adapter.loads(graft(old,new,**opts)) as m:
   rows[label]={str(lesion):evaluate(m,cases,lesion,False)['native_rollout'] for lesion in (None,'gate_on')}
  print(label,rows[label],flush=True)
 write(a.out,dict(scope='frozen development-only interventions, no fitting',rows=rows))
if __name__=='__main__':main()
