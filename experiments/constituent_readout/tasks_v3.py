"""D3 fresh groups and causal short-spacer augmentation; long delays held out."""
from dataclasses import replace
import hashlib,random
from .tasks import dataset as previous_dataset,shifted,balanced_offsets
from experiments.mad_next.tasks import Spec,sample_world,variants,solve,MAP,PAD
SALT=b'SHoEbox-EXP01D-context-variation-v2\0'

def fold(key):
 b=int.from_bytes(hashlib.sha256(SALT+key).digest()[:8],'big')%10
 return 'train' if b<6 else 'dev' if b<8 else 'test'

def dataset(count,split,seed=20261119):
 if split not in ('train','dev','test') or count<1:raise ValueError('invalid data request')
 excluded={e.split_key for e in previous_dataset(1024,'test')} if split=='test' else set()
 spec=Spec('binding',vocab=2);rng=random.Random(seed);out=[];seen=set()
 for _ in range(max(10000,count*1000)):
  w=sample_world(spec,rng)
  if w.group_key in excluded or w.content_key in seen or fold(w.group_key)!=split:continue
  seen.add(w.content_key);family=variants(w,spec,rng);assert all(solve(e,spec)==e.answer for e in family);out.extend(family)
  if len(out)>=count:return out
 raise ValueError('insufficient unique worlds')

def spaced(ep,lengths):
 if len(lengths)!=2 or any(type(n) is not int or not 0<=n<8 for n in lengths):raise ValueError('two short spacers required')
 data=bytearray();mask=bytearray();ends={i+4 for i,b in enumerate(ep.data) if b==MAP};k=0
 for i,b in enumerate(ep.data):
  data.append(b);mask.append(ep.mask[i])
  if i in ends:data.extend([PAD]*lengths[k]);mask.extend([0]*lengths[k]);k+=1
 if k!=2:raise ValueError('two records required')
 return replace(ep,data=bytes(data),mask=bytes(mask))
