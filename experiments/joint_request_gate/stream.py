"""Byte-exact resumable episode order; no replay of fitted model updates."""
import copy,hashlib,random
from .tasks import dataset,spaced,shifted
from .adapter import prefix_labels

def tuples(x):return tuple(tuples(v) for v in x) if isinstance(x,list) else x

def new_cursor(order_seed=702,all_bytes=False,local=True):
 return dict(schema=1,recipe='EXP01-E1',bytes=0,targets=0,episode=0,order=[],current=None,model_bytes=0,
   order_rng=random.Random(order_seed).getstate(),augmentation_rng=random.Random(order_seed+1000).getstate(),
   all_bytes=all_bytes,local=local,order_sha256='0'*64,padding_bytes=0,answer_targets=0)

def advance(m,cursor,n,cases=None):
 if type(n) is not int or n<0 or cursor.get('schema')!=1:raise ValueError('invalid cursor/budget')
 if m.inventory()['bytes']!=cursor['model_bytes']:raise ValueError('checkpoint/cursor mismatch')
 s=copy.deepcopy(cursor);cases=dataset(2048,'train') if cases is None else cases
 rng=random.Random();rng.setstate(tuples(s['order_rng']));aug=random.Random();aug.setstate(tuples(s['augmentation_rng']))
 remaining=n;loss=0.
 while remaining:
  if s['current'] is None:
   if not s['order']:s['order']=list(range(len(cases)));rng.shuffle(s['order'])
   ix=s['order'].pop();sp=[aug.randrange(8),aug.randrange(8)];off=(s['episode']%64-sum(sp))%64
   s['episode']+=1;s['current']=dict(index=ix,spacers=sp,offset=off,used=0)
   s['order_sha256']=hashlib.sha256(bytes.fromhex(s['order_sha256'])+f'{ix}:{sp}:{off};'.encode()).hexdigest();m.reset()
  cur=s['current'];ep=shifted(spaced(cases[cur['index']],cur['spacers']),cur['offset'])
  start=cur['used'];stop=min(len(ep.data)-1,start+remaining)
  mask=None if s['all_bytes'] else ep.mask[start+1:stop+1]
  labels=prefix_labels(ep)[start:stop] if s['local'] else None
  l,targets=m.run(ep.data[start:stop],ep.data[start+1:stop+1],mask,labels)
  used=stop-start;remaining-=used;loss+=l;s['bytes']+=used;s['targets']+=targets
  s['answer_targets']+=sum(ep.mask[start+1:stop+1]);s['padding_bytes']+=ep.data[start:stop].count(9)
  cur['used']=stop;s['model_bytes']=stop
  if stop==len(ep.data)-1:s['current']=None
 s['order_rng']=rng.getstate();s['augmentation_rng']=aug.getstate()
 return s,loss
