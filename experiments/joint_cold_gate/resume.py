"""Exact extension of registered language and generic-task checkpoint cursors.

No model updates are replayed. Language resumes at the next pinned epoch after
an end cursor. Generic tasks finish the saved partial example before a new one.
These helpers do not add local syntax labels to generic objectives.
"""
import copy,hashlib,io,json,random,zipfile
import numpy as np
from .stream import tuples
from .regressions import fresh_cases
from experiments.mad.tasks import TaskSpec
from experiments.mad.backend import CompressionHead

def language_advance(model,cursor,count,corpus):
 if count<0 or cursor['source_sha256']!=corpus.manifest['files']['train']['sha256']:raise ValueError('cursor/corpus mismatch')
 if model.inventory()['bytes']!=cursor['model_bytes']:raise ValueError('cursor/model mismatch')
 state=copy.deepcopy(cursor);docs=corpus.docs('train');total=state['targets'];left=count
 i=state['document_index'];at=state['used_in_document'];epoch=state.get('epoch',0)
 while left:
  if at==len(docs[i])-1:
   i+=1;at=0
   if i==len(docs):i=0;epoch+=1
  if at==0:model.reset()
  n=min(left,len(docs[i])-1-at,250000-total%250000)
  model.run(docs[i][at:at+n],docs[i][at+1:at+n+1]);at+=n;total+=n;left-=n
  if total%250000==0:model.finish()
 state.update(document_index=i,used_in_document=at,targets=total,epoch=epoch,model_bytes=model.inventory()['bytes'],at_end=i==len(docs)-1 and at==len(docs[i])-1)
 return state

def load_task(path,cls,seed):
 if not zipfile.is_zipfile(path):return cls.loads(path.read_bytes()),None
 with zipfile.ZipFile(path) as z:
  model=cls.loads(z.read('adapter.qr'));head=CompressionHead(4,16,seed)
  with np.load(io.BytesIO(z.read('decoder.npz')),allow_pickle=False) as arrays:
   for k,v in head.arrays.items():v[:]=arrays[k]
 return model,head

def task_advance(model,head,cursor,count,task):
 if count<0 or model.inventory()['bytes']!=cursor['model_bytes']:raise ValueError('cursor/model mismatch')
 spec=TaskSpec('selective_copy',length=2,noise=.5) if task=='selective_copy' else TaskSpec('compression',vocab=16)
 examples=fresh_cases(spec,256,'train');state=copy.deepcopy(cursor);rng=random.Random();rng.setstate(tuples(state['order_rng']))
 # Rebuild only the digest of previous example order; never train again.
 historical=random.Random(1067);order=[];total=0;digest=hashlib.sha256()
 registered=cursor.get('registered_bytes',cursor['bytes'])
 while total<registered:
  if not order:order=list(range(len(examples)));historical.shuffle(order)
  ix=order.pop();ep=examples[ix];cost=len(ep.data)-(head is None);used=min(cost,registered-total)
  total+=used;digest.update(f'{ix}:{used};'.encode())
 if digest.hexdigest()!=cursor['order_sha256']:raise ValueError('historical data order changed')
 # Extension digest is separately chained, so interruptions do not change it.
 chain=state.get('continuation_sha256','0'*64);left=count
 while left:
  if state['used']==state['cost']:
   if not state['order']:state['order']=list(range(len(examples)));rng.shuffle(state['order'])
   ix=state['order'].pop();ep=examples[ix];state.update(last_index=ix,used=0,cost=len(ep.data)-(head is None));model.reset()
  ep=examples[state['last_index']];start=state['used'];stop=min(state['cost'],start+left)
  train_stop=min(stop,len(ep.data)-1)
  if train_stop>start:
   _,targets=model.run(ep.data[start:train_stop],ep.data[start+1:train_stop+1],ep.mask[start+1:train_stop+1]);state['targets']+=targets
  if head is not None and stop==state['cost']:
   model.observe(ep.data[-1]);_,dx=head.train(model.features(),ep.answer,.1);model.credit(dx);state['decoder_updates']+=1
  for at in range(start,stop):chain=hashlib.sha256(bytes.fromhex(chain)+f"{state['last_index']}:{at};".encode()).hexdigest()
  n=stop-start;state['bytes']+=n;left-=n;state['used']=stop
 state.update(model_bytes=model.inventory()['bytes'],order_rng=rng.getstate(),continuation_sha256=chain)
 # Original digest covers the registered prefix, not the separately extended tail.
 state['registered_bytes']=cursor.get('registered_bytes',cursor['bytes'])
 return state
