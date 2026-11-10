import hashlib,json,os
import numpy as np
from model import ROOT,Model,core,evaluate
from experiment import make,TRAIN,SIZE
os.sched_setaffinity(0,{int(os.environ.get('PROBE_CPU','8'))})
a=make('separated_ports',True);rate=dict(mode='periodic',lr=.008,lp=.008/3,route=.002)
a.run(TRAIN[:1533],TRAIN[1:1534],**rate)
f=ROOT/'results'/f'ports_check_{SIZE}.npz';a.save(f);b=Model.load(f)
a.run(TRAIN[1533:4096],TRAIN[1534:4097],**rate)
b.run(TRAIN[1533:2300],TRAIN[1534:2301],**rate);b.run(TRAIN[2300:4096],TRAIN[2301:4097],**rate)
# Compare a checkpoint with its own reload. Independently allocated raw C++
# State structs can differ in padding bytes despite identical numeric state.
assert a.fingerprint()==b.fingerprint() and np.array_equal(a.probe_state(),b.probe_state())
c=a.frozen_copy();d=a.frozen_copy();_,x=c.run(TRAIN[:256],TRAIN[1:257],logits=True);_,y=d.run(TRAIN[:256],np.zeros(256,np.uint8),logits=True);assert np.array_equal(x,y)
for m in [a,b,c,d]:m.close()
f.unlink();(ROOT/'results'/f'ports_checks_{SIZE}.json').write_text(json.dumps({'reload_chunk_exact':True,'target_independence':True}));print(SIZE,'separate-port checks passed')
