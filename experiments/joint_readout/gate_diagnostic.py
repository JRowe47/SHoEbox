"""Frozen development-only gate interventions; no fitting or label input."""
import argparse,json
from pathlib import Path
import numpy as np
from .adapter import Adapter
from .tasks import dataset,balanced_offsets
from .regressions import Corpus
from experiments.query_access.run import evaluate
from experiments.query_readout.run import write

def diagnose(path,corpus):
 cases=balanced_offsets(dataset(512,'dev'));values={'answers':[],'other_fixture':[],'language':[]}
 with Adapter.loads(Path(path).read_bytes()) as m:
  scores={name:evaluate(m,cases,lesion=name,all_modes=False)['native_rollout'] for name in (None,'gate_on','gate_off','no_query')}
  with m.clone() as f:
   for ep in cases[:64]:
    f.reset()
    for at,b in enumerate(ep.data[:-1]):f.observe(b);values['answers' if ep.mask[at+1] else 'other_fixture'].append(f.gate())
   for doc in corpus.docs('dev')[:8]:
    f.reset()
    for b in doc[:1024]:f.observe(b);values['language'].append(f.gate())
  return dict(scores={str(k):v for k,v in scores.items()},gate_values={k:dict(mean=float(np.mean(v)),minimum=min(v),maximum=max(v),samples=len(v)) for k,v in values.items()})
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();write(a.out,diagnose(a.checkpoint,Corpus(a.data)))
if __name__=='__main__':main()
