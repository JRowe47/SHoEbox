import hashlib,json,os
import numpy as np
from model import Model,legacy,ROOT,core
size='1m' if core.E==12 else '5m';os.sched_setaffinity(0,{int(os.environ.get('PROBE_CPU','0'))})
src=ROOT/'sources'/size/'last.npz';m=Model.load(src,reset=True)
past=np.frombuffer((ROOT/'data/fresh.bin').read_bytes(),np.uint8)[:1000001][-65537:]
m.run(past[:-1],past[1:]);counts=np.array(m.stats()['expert_counts']);prob=counts/np.maximum(1,counts.sum(1,keepdims=True))
effective=np.exp(-(prob*np.log(np.maximum(prob,1e-30))).sum(1));m.close()
maximum=32 if core.E==12 else 128
target=effective[2:];caps=np.ones(16,dtype=int)*2;budget=16*core.E
while caps.sum()<budget:
 # Diminishing proportional allocation, bounded by the compiled bank capacity.
 score=target/(caps+.5);score[caps>=maximum]=-1
 caps[np.argmax(score)]+=1
caps=np.r_[[core.E,core.E],caps]
plan={'size':size,'source_experts':core.E,'compiled_capacity':maximum,'capacities':caps.tolist(),'effective_experts':dict(zip(core.NAMES,effective.tolist())),
 'calibration':'Frozen model on the last 65,536 targets of the already-seen first 1M training bytes; no held-out observations.',
 'rankings':[np.argsort(-row,kind='stable').tolist() for row in counts]}
(ROOT/'results'/f'allocation_plan_{size}.json').write_text(json.dumps(plan,indent=2))
# A padded homogeneous bank must reproduce this unpadded legacy control exactly.
m=legacy.Model.load(src,reset=True);train=np.frombuffer((ROOT/'data/train.bin').read_bytes(),np.uint8)
loss=m.run(train[:4096],train[1:4097],mode='periodic',lr=.008,lp=.008/3,route=.002)
ref={'loss':loss,'flat_sha256':hashlib.sha256(m.flat.tobytes()).hexdigest(),'extra_sha256':hashlib.sha256(m.extra_flat.tobytes()).hexdigest()}
(ROOT/'results'/f'allocation_reference_{size}.json').write_text(json.dumps(ref,indent=2));print(json.dumps(plan))
