"""Compact evidence and cost aggregation for E8; no fitting or audit opening."""
import argparse,hashlib,json,random,shutil
from pathlib import Path
import numpy as np
from .gates import binding_gate,generic_gate
from .regressions import fresh_cases
from experiments.mad.tasks import TaskSpec
from experiments.query_readout.run import write

def read(path):return json.loads(path.read_text())
def padding(row):
 task=row['task']
 if task=='language':return 0
 spec=TaskSpec('selective_copy',length=2,noise=.5) if task=='selective_copy' else TaskSpec('compression',vocab=16)
 eps=fresh_cases(spec,256,'train');rng=random.Random(1067);order=[];used=pad=0;digest=hashlib.sha256()
 while used<row['bytes']:
  if not order:order=list(range(len(eps)));rng.shuffle(order)
  i=order.pop();ep=eps[i];n=min(len(ep.data)-(task!='compression'),row['bytes']-used)
  pad+=ep.data[:n].count(9);used+=n;digest.update(f'{i}:{n};'.encode())
 assert digest.hexdigest()==row['order_sha256']
 return pad

def summarize(rows):
 native=[r['test']['native_rollout'] for r in rows]
 offsets=np.mean([[r['offsets'][i]['exact'] for i in range(64)] for r in rows],axis=0).tolist()
 return dict(primary=[r['native_exact'] for r in native],primary_mean=float(np.mean([r['native_exact'] for r in native])),first=[r['first_answer'] for r in native],second=[r['second_answer'] for r in native],offsets=offsets,offset_mean=float(np.mean(offsets)),worst_offset=min(offsets),both_orders=[r['counterfactual']['both_query_orders_exact'] for r in rows],delays={str(d['delay']):[r['delays'][i]['exact'] for r in rows] for i,d in enumerate(rows[0]['delays'])},gate=binding_gate(rows))

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();w=a.workspace;o=a.out;o.mkdir(parents=True,exist_ok=True);ledger=[]
 def add(campaign,arm,seed,r,initial=None,pad=None):
  inv=r['inventory'];old=initial or r.get('initial_inventory',{});counts={k:inv.get(k,0)-old.get(k,0) for k in ('head_targets','gate_targets','local_training_windows','local_projection_steps','local_presence_steps')}
  ledger.append(dict(campaign=campaign,arm=arm,seed=seed,bytes=r['bytes'],targets=r['targets'],padding_bytes=r.get('padding_bytes',pad or 0),cpu_seconds=r['cpu_seconds'],parameters=inv['parameters'],readout_parameters=inv['readout_parameters'],added_sequence_state_bytes=inv['added_persistent_sequence_state_bytes'],optimizer_state_bytes=inv['optimizer_state_bytes'],adapter_scratch_bytes=inv['readout_cache_bytes'],decoder_updates=r.get('decoder_updates',0),**counts))
 pilot=read(w/'exp01e8-pilot/result.json')
 add('pilot','warmup',7,pilot['warmup']);add('pilot','binding',7,pilot['binding']);add('pilot','retention',7,pilot['retention'],pilot['binding']['inventory'])
 for r in read(w/'exp01e8-regressions/train.json')['rows']:add('regressions',r['task']+'/'+r['arm'],r['seed'],r,pad=padding(r))
 sums={};campaigns={}
 for size in ('formal','1m','5m'):
  path=w/f'exp01e8-{size}/train.json'
  if not path.exists():continue
  d=read(path)
  for row in d['rows']:
   for arm,r in row['arms'].items():
    add(size,arm+'/warmup',row['seed'],r['warmup']);add(size,arm,row['seed'],r)
    if 'retention' in r:add(size,arm+'/retention',row['seed'],r['retention'],r['inventory'])
  sums[size]={'development':{arm:summarize([r['arms'][arm]['development_audit'] for r in d['rows']]) for arm in d['plan']['arms']}}
  if size=='formal':sums[size]['retained_development']={arm:summarize([r['arms'][arm]['retention']['development_audit'] for r in d['rows']]) for arm in d['plan']['arms']}
  if (path.parent/'audit.json').exists():
   audit=read(path.parent/'audit.json');sums[size]['audit']={arm:summarize([r['arms'][arm] for r in audit['rows']]) for arm in d['plan']['arms']}
   if size=='formal':sums[size]['retained_audit']={arm:summarize([r['arms'][arm]['retention'] for r in audit['rows']]) for arm in d['plan']['arms']}
  if (path.parent/'benchmark.json').exists():sums[size]['benchmark']=read(path.parent/'benchmark.json')
 for r in ledger:
  c=campaigns.setdefault(r['campaign'],dict(fits=0,bytes=0,targets=0,padding_bytes=0,cpu_seconds=0,local_projection_steps=0,local_presence_steps=0,gate_targets=0,decoder_updates=0));c['fits']+=1
  for k in c:
   if k!='fits':c[k]+=r[k]
 total={k:sum(c[k] for c in campaigns.values()) for k in next(iter(campaigns.values()))}
 processes={f.name:read(f) for f in sorted(w.glob('e8-*-process.json'))}
 write(o/'accounting.json',dict(total=total,campaigns=campaigns,fits=ledger,process_costs=processes,scope='All research fitting, including warmups, failed/control arms, padding and auxiliary work. Whole-process measurements additionally include evaluation/compilation/I/O and must not be added again to fitting CPU. Initial pilot/generic non-fit CPU was not separately metered. Native internal auxiliary updates are included in CPU, but their scalar write counts are not instrumented.'))
 generic={'development':generic_gate(read(w/'exp01e8-regressions/train.json')['rows'])}
 if (w/'exp01e8-regressions/audit.json').exists():generic['audit']=generic_gate(read(w/'exp01e8-regressions/audit.json')['rows'],True)
 summary=dict(binding=sums,generic=generic,pilot={k:summarize([pilot[k]['development_audit']]) for k in ('binding','retention')},total=total)
 write(o/'summary.json',summary)
 raw=o/'evidence';raw.mkdir(exist_ok=True)
 for folder in sorted(w.glob('exp01e8-*')):
  if folder.is_dir():
   dest=raw/folder.name;dest.mkdir(exist_ok=True)
   for f in folder.glob('*.json'):
    if not f.name.endswith('.cursor.json'):shutil.copy2(f,dest/f.name)
 for f in w.glob('e8-*.json'):shutil.copy2(f,raw/f.name)
 for f in w.glob('e8-*.log'):
  if 'test' in f.name:shutil.copy2(f,raw/(f.stem+'.txt'))
 print(total)
if __name__=='__main__':main()
