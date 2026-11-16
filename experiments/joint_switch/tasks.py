"""Fresh EXP01-E grouped worlds; all previously opened audit worlds excluded."""
import hashlib,random
from functools import lru_cache
from experiments.constituent_readout.tasks import dataset as d2
from experiments.constituent_readout.tasks_v3 import dataset as d3,spaced,shifted,balanced_offsets
from experiments.joint_readout.tasks import dataset as e1
from experiments.mad_next.tasks import Spec,sample_world,variants,solve
SALT=b'SHoEbox-EXP01E-switch-v2\0'
@lru_cache(None)
def excluded():return {e.split_key for fn in (d2,d3,e1) for e in fn(1024,'test')}
def fold(key):
 n=int.from_bytes(hashlib.sha256(SALT+key).digest()[:8],'big')%10
 return 'train' if n<6 else 'dev' if n<8 else 'test'
@lru_cache(None)
def dataset(count,split,seed=20261121):
 if split not in ('train','dev','test') or count<1:raise ValueError('invalid data request')
 rng=random.Random(seed);spec=Spec('binding',vocab=2);out=[];seen=set()
 for _ in range(max(10000,count*1000)):
  w=sample_world(spec,rng)
  if w.group_key in excluded() or w.content_key in seen or fold(w.group_key)!=split:continue
  seen.add(w.content_key);family=variants(w,spec,rng);assert all(solve(e,spec)==e.answer for e in family);out.extend(family)
  if len(out)>=count:return out
 raise ValueError('insufficient unique worlds')
