import json,os
from pathlib import Path
import numpy as np
import model
ROOT=Path(__file__).resolve().parent;os.sched_setaffinity(0,{0});size='1m' if model.core.E==12 else '5m'
source=ROOT.parent/'utility_probe/results'/size/'seed1/center_split/last.npz'
data=np.frombuffer((ROOT.parent/'scale_probe/data/fresh.bin').read_bytes(),np.uint8)
rates=dict(mode='periodic',lr=.008,lp=.008/3,route=.002)
model.KERNEL='baseline';a=model.Model.load(source)
model.KERNEL='cached';b=model.Model.load(source)
checks={}
# Changes to keys through Python are visible at the next native call.
a.params['keys'][0,0,0]+=.01;b.params['keys'][0,0,0]+=.01
for lo,hi in [(0,537),(537,4096)]:
 x=a.run(data[lo:hi],data[lo+1:hi+1],**rates);y=b.run(data[lo:hi],data[lo+1:hi+1],**rates)
 assert x==y and a.fingerprint()==b.fingerprint()
checks['external_key_edits_and_chunk_calls_exact']=True
snapshot=b.snapshot_copy();assert snapshot.fingerprint()==b.fingerprint()
assert not np.shares_memory(snapshot.flat,b.flat) and not np.shares_memory(snapshot.extra_flat,b.extra_flat)
before=snapshot.fingerprint();b.run(data[4096:8192],data[4097:8193],**rates);assert snapshot.fingerprint()==before
checks['snapshot_is_independent_and_complete']=True;snapshot.close()
path=ROOT/'results'/f'verify_{size}.npz';b.save(path);c=model.Model.load(path);assert b.fingerprint()==c.fingerprint()
for m in [b,c]:m.run(data[8192:12288],data[8193:12289],**rates)
assert b.fingerprint()==c.fingerprint();checks['resume_exact']=True;c.close();path.unlink()
f=b.frozen_copy();g=b.frozen_copy();before=b.fingerprint()
_,p=f.run(data[:128],data[1:129],logits=True);_,q=g.run(data[:128],np.zeros(128,np.uint8),logits=True)
assert np.array_equal(p,q) and np.array_equal(f.publications(),g.publications())
assert np.array_equal(f.ymean(),b.ymean()) and before==b.fingerprint()
checks['evaluation_target_independent_and_trainer_preserved']=True;f.close();g.close();a.close();b.close()
# Check initial training before prototypes or routing means have matured.
template=model.Model.load(source);params,_=model.core.initialize(seed=871)
def fresh(kernel):return model.Model(config=template.config,params=params,fixed=template.fixed,lace=template.lace,state_config=template.state_config,probe=template.probe,utility=template.utility,kernel=kernel)
a=fresh('baseline');b=fresh('cached')
x=a.run(data[:4096],data[1:4097],mode='periodic',lr=.08,lp=.08/3,route=.002);y=b.run(data[:4096],data[1:4097],mode='periodic',lr=.08,lp=.08/3,route=.002)
assert x==y and a.fingerprint()==b.fingerprint();checks['cold_start_exact']=True
a.close();b.close();template.close()
(ROOT/'results'/f'runtime_checks_{size}.json').write_text(json.dumps(checks,indent=2)+'\n');print(size,checks,flush=True)
