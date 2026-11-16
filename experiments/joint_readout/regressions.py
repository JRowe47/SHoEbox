"""Matched generic-task fits and separate post-binding language retention.

Fixture syntax labels are NEVER constructed in this module. Reconstruction
reads only the existing final L3 vector and charges its own fitted MLP budget.
"""
import argparse,hashlib,io,json,os,random,time,zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import numpy as np
from .adapter import Adapter
from .run import source_hashes
from .tasks import dataset as binding,balanced_offsets
from experiments.mad.backend import CompressionHead
from experiments.mad.tasks import TaskSpec,VALUE,_candidate,fixed_table,solve,dataset as old_dataset,fingerprint
from experiments.query_readout.run import write
from experiments.query_access.run import evaluate as binding_evaluate
from experiments.constituent_readout.stage3 import audits
ARMS={'native':'native','d3_joint':'paired_latest','gated':'paired_gated'}
SALT=b'SHoEbox-EXP01E-regression-v1\0'

def cases(spec,n,fold):
 excluded={e.split_key for e in old_dataset(spec,128,'test')};rng=random.Random(20261120);table=fixed_table(spec,47);out=[];seen=set()
 for _ in range(n*10000):
  e=_candidate(spec,rng,table,len(out));h=hashlib.sha256(SALT+spec.task.encode()+e.split_key).digest();b=int.from_bytes(h[:8],'big')%10;f='train' if b<6 else 'dev' if b<8 else 'test'
  if f!=fold or e.split_key in excluded or e.digest in seen:continue
  assert solve(e,spec,table)==e.answer;out.append(e);seen.add(e.digest)
  if len(out)==n:return out
 raise ValueError('not enough grouped cases')

class Corpus:
 def __init__(self,path):
  path=Path(path);self.manifest=json.loads((path/'manifest.json').read_text());self.data={}
  for f,info in self.manifest['files'].items():
   b=(path/info['file']).read_bytes();assert hashlib.sha256(b).hexdigest()==info['sha256'];self.data[f]=b
 def docs(self,f):
  b=self.data[f]
  return [b[d['start']:d['stop']] for d in self.manifest['files'][f]['documents']]
 def evaluate(self,m,f):
  docs=self.docs(f);return sum(m.evaluate_bytes(d)*(len(d)-1) for d in docs)/sum(len(d)-1 for d in docs)

def language(m,corpus,checkpoint,retention=False):
 cpu=wall=loss=0.;total=0;curve=[];next_eval=250000
 for i,doc in enumerate(corpus.docs('train')):
  c=time.process_time();w=time.perf_counter();m.reset();cpu+=time.process_time()-c;wall+=time.perf_counter()-w;at=0
  while at<len(doc)-1:
   used=min(len(doc)-1-at,next_eval-total)
   c=time.process_time();w=time.perf_counter();l,n=m.run(doc[at:at+used],doc[at+1:at+used+1]);loss+=l;total+=n;at+=n
   cpu+=time.process_time()-c;wall+=time.perf_counter()-w
   if total==next_eval:
    c=time.process_time();w=time.perf_counter();m.finish();cpu+=time.process_time()-c;wall+=time.perf_counter()-w
    point=dict(bytes=total,cpu_seconds=cpu,train_nll=loss/total,dev_nll=corpus.evaluate(m,'dev'))
    if retention:point['binding_dev']=binding_evaluate(m,balanced_offsets(binding(512,'dev')),all_modes=False)
    curve.append(point);next_eval+=250000
 c=time.process_time();m.finish();cpu+=time.process_time()-c
 raw=m.dumps();checkpoint.write_bytes(raw)
 cursor=dict(schema=1,recipe='EXP01-E-FineWeb-sequential',fold='train',document_index=i,used_in_document=at,targets=total,model_bytes=m.inventory()['bytes'],source_sha256=corpus.manifest['files']['train']['sha256'],at_end=True)
 write(checkpoint.with_suffix('.cursor.json'),cursor)
 with Adapter.loads(raw) as b:assert corpus.evaluate(m,'dev')==corpus.evaluate(b,'dev')
 return dict(bytes=total,targets=total,cpu_seconds=cpu,wall_seconds=wall,bytes_per_cpu_second=total/cpu,curve=curve,inventory=m.inventory(),checkpoint_sha256=hashlib.sha256(raw).hexdigest(),reload_exact=True,local_supervised_updates=0)

def task_eval(m,head,spec,eps,zero=False):
 sums=np.zeros(7);native=restricted=0
 with m.clone() as f:
  for ep in eps:
   if head:
    f.reset()
    for b in ep.data:f.observe(b)
    x=f.features();x[:]=0 if zero else x
    p=head.predict(x);yes=p.argmax(1)==ep.answer;l=-np.log(np.maximum(p[np.arange(len(ep.answer)),ep.answer],1e-30)).sum()
    sums+=np.array([len(ep.answer),l,l,yes.sum(),yes.all(),yes.all(),yes.sum()]);restricted+=yes.all();native+=yes.all()
   else:
    sums+=np.array(f.score(ep,spec.vocab,0)[:7]);restricted+=f.score(ep,spec.vocab,1)[4];native+=f.score(ep,spec.vocab,2)[5]
 return dict(token_accuracy=float(sums[3]/sums[0]),nll=float(sums[2]/sums[0]),native_nll=None if head else float(sums[1]/sums[0]),exact=float(restricted/len(eps)),native_exact=None if head else float(native/len(eps)),cases=len(eps))

def save_task(path,m,head,cursor):
 if head is None:path.write_bytes(m.dumps())
 else:
  with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
   z.writestr('adapter.qr',m.dumps());b=io.BytesIO();np.savez(b,**head.arrays);z.writestr('decoder.npz',b.getvalue())
 write(path.with_suffix('.cursor.json'),cursor)

def load_task(path,spec,seed):
 if spec.task!='compression':return Adapter.loads(path.read_bytes()),None
 with zipfile.ZipFile(path) as z:
  m=Adapter.loads(z.read('adapter.qr'));h=CompressionHead(spec.length,spec.vocab,seed)
  with np.load(io.BytesIO(z.read('decoder.npz')),allow_pickle=False) as a:
   for k,v in h.arrays.items():v[:]=a[k]
 return m,h

def task_fit(seed,kind,spec,lr,train,dev,path):
 head=CompressionHead(spec.length,spec.vocab,seed) if spec.task=='compression' else None
 extra=head.parameters if head else 0
 with Adapter(parameters=500000-extra,seed=seed,kind=kind,joint=True,lr_scale=lr,auxiliary=0.) as m:
  total=targets=cpu=wall=updates=0;order=[];rng=random.Random(1067);digest=hashlib.sha256();curve=[];next_eval=62500
  while total<250000:
   if not order:order=list(range(len(train)));rng.shuffle(order)
   ix=order.pop();ep=train[ix];cost=len(ep.data)-(0 if head else 1);used=min(cost,250000-total);n=min(used,len(ep.data)-1)
   c=time.process_time();w=time.perf_counter();m.reset();loss,nt=m.run(ep.data[:n],ep.data[1:n+1],ep.mask[1:n+1]);targets+=nt
   if head and used==cost:
    m.observe(ep.data[-1]);_,dx=head.train(m.features(),ep.answer,.2*lr);m.credit(dx);updates+=1
   cpu+=time.process_time()-c;wall+=time.perf_counter()-w;total+=used;digest.update(f'{ix}:{used};'.encode())
   if total>=next_eval or total==250000:
    c=time.process_time();m.finish();cpu+=time.process_time()-c;curve.append(dict(bytes=total,cpu_seconds=cpu,dev=task_eval(m,head,spec,dev)));next_eval+=62500
  cursor=dict(schema=1,recipe='EXP01-E-'+spec.task,bytes=total,order=order,order_rng=rng.getstate(),last_index=ix,used=used,cost=cost,order_sha256=digest.hexdigest(),targets=targets,decoder_updates=updates,model_bytes=m.inventory()['bytes'])
  save_task(path,m,head,cursor);inv=m.inventory();inv['reconstruction_parameters']=extra;inv['parameters']+=extra
  b,h=load_task(path,spec,seed)
  with b:assert task_eval(m,head,spec,dev)==task_eval(b,h,spec,dev)
  return dict(bytes=total,targets=targets,cpu_seconds=cpu,wall_seconds=wall,bytes_per_cpu_second=total/cpu,curve=curve,inventory=inv,decoder_updates=updates,order_sha256=digest.hexdigest(),reload_exact=True,local_supervised_updates=0)

def job(args):
 seed,plan,folder,data=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)%len(plan['cpus'])]});out=Path(folder)/str(seed);out.mkdir(exist_ok=True);corpus=Corpus(data);rows=[]
 for task in ('language','selective_copy','compression'):
  for arm,kind in ARMS.items():
   path=out/f'{task}-{arm}.checkpoint'
   if task=='language':
    with Adapter(seed=seed,kind=kind,joint=True) as m:r=language(m,corpus,path)
   else:
    spec=TaskSpec('selective_copy',length=2,noise=.5) if task=='selective_copy' else TaskSpec('compression',vocab=16)
    r=task_fit(seed,kind,spec,2. if task=='selective_copy' else .5,cases(spec,256,'train'),cases(spec,128,'dev'),path)
   r.update(seed=seed,task=task,arm=arm);rows.append(r);write(out/'train.json',rows);print(task,seed,arm,r['curve'][-1],flush=True)
 return rows

def audit_job(args):
 seed,plan,folder,data=args;os.sched_setaffinity(0,{plan['cpus'][plan['seeds'].index(seed)%len(plan['cpus'])]});out=Path(folder)/str(seed);corpus=Corpus(data);rows=json.loads((out/'train.json').read_text())
 for r in rows:
  task=r['task'];path=out/f"{task}-{r['arm']}.checkpoint"
  if task=='language':
   with Adapter.loads(path.read_bytes()) as m:r['test_nll']=corpus.evaluate(m,'test')
  else:
   spec=TaskSpec('selective_copy',length=2,noise=.5) if task=='selective_copy' else TaskSpec('compression',vocab=16)
   m,h=load_task(path,spec,seed)
   with m:
    r['test']=task_eval(m,h,spec,cases(spec,128,'test'))
    if h:r['zero_bottleneck']=task_eval(m,h,spec,cases(spec,128,'test'),True)
 write(out/'audit.json',rows);return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=('train','audit','retention'));p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--binding',type=Path);p.add_argument('--workers',type=int,default=3);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if a.phase in ('train','retention'):
  if (a.out/'plan.json').exists():raise FileExistsError('registered run exists')
  corpus=Corpus(a.data);plan=dict(utc=datetime.now(timezone.utc).isoformat(),phase=a.phase,seeds=[1,7,19],cpus=sorted(os.sched_getaffinity(0))[-a.workers:],source_hashes=source_hashes(),data_manifest_sha256=hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest(),arms=ARMS,budgets=dict(language=1000000,selective_copy=250000,compression=250000),no_local_labels=True)
  write(a.out/'plan.json',plan)
 else:
  plan=json.loads((a.out/'plan.json').read_text())
  if not (a.out/'train.json').exists() or (a.out/'audit.json').exists():raise ValueError('finish fits before one-time audit')
  for rel,h in plan['source_hashes'].items():
   if hashlib.sha256((Path(__file__).resolve().parents[2]/rel).read_bytes()).hexdigest()!=h:raise ValueError('registered source changed: '+rel)
  write(a.out/'selection.json',dict(locked=True,arms='all fixed recipes; no audit selection'))
 if a.phase=='retention':
  rows=[]
  for seed in plan['seeds']:
   out=a.out/str(seed);out.mkdir(exist_ok=True)
   with Adapter.loads((a.binding/str(seed)/'gated_all.qr').read_bytes()) as m:
    r=language(m,corpus,out/'retained.qr',True);r['seed']=seed
    r['development_audit']=audits(m,binding(512,'dev'));rows.append(r);write(a.out/'train.json',dict(plan=plan,rows=rows));print('retention',seed,r['curve'][-1],flush=True)
 else:
  fn=job if a.phase=='train' else audit_job
  with ProcessPoolExecutor(max_workers=a.workers) as pool:groups=list(pool.map(fn,[(s,plan,a.out,a.data) for s in plan['seeds']]))
  write(a.out/(a.phase+'.json'),dict(plan=plan,rows=[r for g in groups for r in g]))
if __name__=='__main__':main()
