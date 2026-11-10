import hashlib,json,os,tempfile
import numpy as np
from model import Model,legacy,ROOT,core,evaluate
from experiment import make,TRAIN,SIZE
os.sched_setaffinity(0,{int(os.environ.get('PROBE_CPU','0'))})
rates=dict(mode='periodic',lr=.008,lp=.008/3,route=.002)
def fingerprint(m):return hashlib.sha256(m.fingerprint().encode()+m.probe_state().tobytes()).hexdigest()
checks={}
a=make('baseline');b=legacy.Model.load(ROOT/'sources'/SIZE/'last.npz')
x=a.run(TRAIN[:2048],TRAIN[1:2049],**rates);y=b.run(TRAIN[:2048],TRAIN[1:2049],**rates)
assert x==y and a.fingerprint()==b.fingerprint();checks['exact_legacy_training']=True;a.close();b.close()
for profile in ['baseline','context10','route_center','source_center_reset']:
 a=make(profile);b=make(profile)
 x=a.run(TRAIN[:4096],TRAIN[1:4097],**rates)
 b.run(TRAIN[:1003],TRAIN[1:1004],**rates)
 path=ROOT/'results'/f'temp_{SIZE}_{profile}.npz';path.parent.mkdir(exist_ok=True);b.save(path);b.close();b=Model.load(path)
 b.run(TRAIN[1003:4096],TRAIN[1004:4097],**rates)
 assert fingerprint(a)==fingerprint(b),profile
 before=fingerprint(a);evaluate(a,TRAIN[:4097]);assert before==fingerprint(a)
 c=a.frozen_copy();d=a.frozen_copy();c.run(TRAIN[:256],TRAIN[1:257],logits=True)
 _,u=d.run(TRAIN[:256],np.zeros(256,np.uint8),logits=True)
 c.close();d.close()
 c=a.frozen_copy();d=a.frozen_copy()
 _,p=c.run(TRAIN[:256],TRAIN[1:257],logits=True);_,q=d.run(TRAIN[:256],np.zeros(256,np.uint8),logits=True)
 assert np.array_equal(p,q)
 assert np.array_equal(c.publications(),d.publications())
 checks[profile]={'reload_and_chunk_exact':True,'evaluation_preserves_trainer':True,'target_independence':True}
 a.close();b.close();c.close();d.close();path.unlink()
(ROOT/'results'/f'checks_{SIZE}.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))
