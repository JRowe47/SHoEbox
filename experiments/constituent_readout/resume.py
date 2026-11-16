"""Exact outer-stream cursors for completed D3 frozen-core readout fits.

Reconstructing a cursor replays only deterministic fixture generation, not model
training. The registered order hash must agree before a cursor is accepted.
"""
import copy,hashlib,random
from .tasks_v3 import dataset,spaced,shifted
from .adapter import prefix_labels

def _tuple(x):return tuple(_tuple(v) for v in x) if isinstance(x,list) else x

def _episode(cases,current):
 return shifted(spaced(cases[current['index']],current['spacers']),current['offset'])

def _next(cases,state,rng,augmentation):
 if not state['order']:
  state['order']=list(range(len(cases)));rng.shuffle(state['order'])
 index=state['order'].pop();lengths=[augmentation.randrange(8),augmentation.randrange(8)]
 current=dict(index=index,spacers=lengths,offset=(state['episode_number']%64-sum(lengths))%64,used=0)
 state['episode_number']+=1;return current

def reconstruct_cursor(fit):
 cases=dataset(2048,'train');rng=random.Random(702);augmentation=random.Random(1702)
 state=dict(schema=1,recipe='EXP01-D3-spaced',charged_bytes=0,episode_number=0,order=[],current=None,last_model_bytes=0)
 digest=hashlib.sha256();targets=0
 while state['charged_bytes']<fit['bytes']:
  current=_next(cases,state,rng,augmentation);ep=_episode(cases,current)
  used=min(len(ep.data)-1,fit['bytes']-state['charged_bytes']);targets+=sum(ep.mask[1:used+1]);current['used']=used
  digest.update(f"{current['index']}:{current['offset']}:{current['spacers']}:{used};".encode())
  state['charged_bytes']+=used;state['last_model_bytes']=used
  state['current']=current if used<len(ep.data)-1 else None
 if digest.hexdigest()!=fit['order_sha256'] or targets!=fit['targets']:raise ValueError('registered stream mismatch')
 state['order_rng']=rng.getstate();state['augmentation_rng']=augmentation.getstate();state['registered_order_sha256']=digest.hexdigest()
 return state

def advance(model,cursor,additional_bytes):
 """Continue one D3 fit; returns an updated JSON-serializable cursor.

 Use a training checkpoint. Its reduced core stays frozen; native residual and
 local recognizers continue online. Checkpoint and cursor must travel together.
 """
 if type(additional_bytes) is not int or additional_bytes<0:raise ValueError('nonnegative byte budget required')
 if cursor.get('schema')!=1 or cursor.get('recipe')!='EXP01-D3-spaced':raise ValueError('unsupported cursor')
 inventory=model.inventory()
 if not inventory['training'] or inventory['joint'] or inventory['kind']!='paired_latest':raise ValueError('D3 frozen-core training checkpoint required')
 if inventory['bytes']!=cursor['last_model_bytes']:raise ValueError('checkpoint/cursor position mismatch')
 state=copy.deepcopy(cursor);cases=dataset(2048,'train');rng=random.Random();rng.setstate(_tuple(state['order_rng']));augmentation=random.Random();augmentation.setstate(_tuple(state['augmentation_rng']))
 remaining=additional_bytes
 while remaining:
  if state['current'] is None:state['current']=_next(cases,state,rng,augmentation);model.reset()
  current=state['current'];ep=_episode(cases,current);start=current['used'];stop=min(len(ep.data)-1,start+remaining)
  model.run(ep.data[start:stop],ep.data[start+1:stop+1],ep.mask[start+1:stop+1],prefix_labels(ep)[start:stop])
  used=stop-start;remaining-=used;state['charged_bytes']+=used;current['used']=stop;state['last_model_bytes']=stop
  if stop==len(ep.data)-1:state['current']=None
 model.finish();state['order_rng']=rng.getstate();state['augmentation_rng']=augmentation.getstate()
 return state
