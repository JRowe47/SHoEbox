"""Exercise consolidation, immutable jobs and exact restart on a longer stream."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'));sys.path.insert(0,str(ROOT/'src'))
from runtime import fresh_model,core
from reference_acceptance import collect,check_integrity
x=np.frombuffer((b'The river runs through the valley. The king crossed the bridge. '*140)[:8193],np.uint8)
models=[];rows=[]
for workers in (1,2):
 m=fresh_model(seed=7,profile='pipeline',pipeline_workers=workers);models.append(m)
 loss,logits=m.run(x[:-1],x[1:],mode='periodic',lr=.04,lp=.04/3,logits=True)
 check_integrity(collect(m));metrics=m.event_metrics();assert all(v['missing']==0 for v in metrics.values())
 consolidated=sum(v['consolidated_records'] for v in metrics.values());assert consolidated>0
 if workers==1:expected=logits;fingerprint=m.fingerprint()
 else:np.testing.assert_array_equal(expected,logits);assert fingerprint==m.fingerprint()
 rows.append(dict(workers=workers,nll=loss,consolidated=consolidated,pipeline=m.pipeline_metrics(),fingerprint=m.fingerprint()))
a=models[1];b=a.snapshot_copy();models.append(b)
_,v=a.run(x[:128],x[1:129],mode='periodic',lr=.04,lp=.04/3,logits=True)
_,w=b.run(x[:128],x[1:129],mode='periodic',lr=.04,lp=.04/3,logits=True)
np.testing.assert_array_equal(v,w);assert a.fingerprint()==b.fingerprint()
(ROOT/'results'/f'stage6_long_check_e{core.E}.json').write_text(json.dumps(dict(targets=8192,bitwise_workers=True,bitwise_restart=True,runs=rows),indent=2)+'\n')
for m in models:m.close()
