"""Aggregate preserved EXP01-E measurements; never fit or select on audit."""
import argparse,json,hashlib,random,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.query_readout.run import write

def mean(xs):return float(np.mean(xs))
def read(p):return json.loads(p.read_text())
def summarize(rows):
 out={}
 for arm in rows[0]['arms']:
  rs=[r['arms'][arm] for r in rows];native=[r['test']['native_rollout'] for r in rs]
  offsets=[mean([r['offsets'][i]['exact'] for r in rs]) for i in range(64)]
  out[arm]=dict(primary=[r['native_exact'] for r in native],first=[r['first_answer'] for r in native],second=[r['second_answer'] for r in native],primary_mean=mean([r['native_exact'] for r in native]),counterfactual=[r['counterfactual']['both_query_orders_exact'] for r in rs],offsets=offsets,offset_mean=mean(offsets),worst_offset=min(offsets),delays={str(d['delay']):[r['delays'][i]['exact'] for r in rs] for i,d in enumerate(rs[0]['delays'])})
  if 'lesions' in rs[0]:out[arm]['lesions']={k:mean([r['lesions'][k]['native_exact'] for r in rs]) for k in rs[0]['lesions']};out[arm]['lesion_unmodified']=mean([r['lesion_unmodified']['native_exact'] for r in rs])
 return out

def generic_padding(task,row):
 from .regressions import cases
 from experiments.mad.tasks import TaskSpec
 spec=TaskSpec('selective_copy',length=2,noise=.5) if task=='selective_copy' else TaskSpec('compression',vocab=16)
 eps=cases(spec,256,'train');rng=random.Random(1067);order=[];total=pad=0;digest=hashlib.sha256()
 while total<row['bytes']:
  if not order:order=list(range(len(eps)));rng.shuffle(order)
  ix=order.pop();ep=eps[ix];cost=len(ep.data)-(task!='compression');used=min(cost,row['bytes']-total)
  pad+=ep.data[:used].count(9);total+=used;digest.update(f'{ix}:{used};'.encode())
 assert digest.hexdigest()==row['order_sha256']
 return pad

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();w=a.workspace;o=a.out;o.mkdir(parents=True,exist_ok=True)
 ledger=[]
 def add(campaign,arm,seed,r,initial=None,language=False,task=None):
  inv=r['inventory'];initial=initial or r.get('initial_inventory',{});counts={}
  for k in ('head_targets','gate_targets','local_training_windows','local_projection_steps','local_presence_steps'):
   counts[k]=inv.get(k,0)-initial.get(k,0)
  if language:
   for k in ('local_training_windows','local_projection_steps','local_presence_steps'):counts[k]=0
   if initial is None:counts['head_targets']=r['bytes'] if inv['kind']!='native' else 0
  pad=r.get('padding_bytes',generic_padding(task,r) if task and task!='language' else 0)
  ledger.append(dict(campaign=campaign,arm=arm,seed=seed,bytes=r['bytes'],targets=r['targets'],padding_bytes=pad,cpu_seconds=r['cpu_seconds'],wall_seconds=r.get('wall_seconds'),parameters=inv['parameters'],readout_parameters=inv['readout_parameters'],added_sequence_state_bytes=inv['added_persistent_sequence_state_bytes'],optimizer_state_bytes=inv['optimizer_state_bytes'],adapter_scratch_bytes=inv['readout_cache_bytes'],decoder_updates=r.get('decoder_updates',0),**counts))
 for campaign in ('pilot','formal','1m','5m'):
  d=read(w/f'exp01e-{campaign}/train.json')
  for r in d['rows']:
   for arm,s in r['arms'].items():add(campaign,arm+'/warmup',r['seed'],s['warmup']);add(campaign,arm,r['seed'],s)
 for r in read(w/'exp01e-regressions/train.json')['rows']:add('regressions',r['task']+'/'+r['arm'],r['seed'],r,task=r['task'])
 for campaign in ('retention-pilot','retention'):
  d=read(w/f'exp01e-{campaign}/train.json');rs=d['rows'] if 'rows' in d else [dict(seed=7,**d)]
  source=read(w/f'exp01e-{"pilot" if campaign=="retention-pilot" else "formal"}/train.json')['rows']
  for r in rs:add(campaign,'gated_all',r['seed'],r,initial=next(s for s in source if s['seed']==r['seed'])['arms']['gated_all']['inventory'],language=True)
 for campaign in ('switch-pilot','mixture-pilot','calibrated-pilot','evidence-gate-pilot','local-gate-pilot','local-gate-control','request-gate-pilot'):
  if not (w/f'exp01e-{campaign}/result.json').exists():continue
  d=read(w/f'exp01e-{campaign}/result.json')
  if 'warmup' in d:
   add(campaign,'warmup',7,d['warmup']);add(campaign,'binding',7,d['binding']);initial=d['binding']['inventory']
  else:
   initial=read(w/'exp01e-switch-pilot/result.json')['binding']['inventory']
  add(campaign,'language-retention',7,d['retention'],initial=initial,language=True)
 if (w/'exp01e7-regressions/train.json').exists():
  for r in read(w/'exp01e7-regressions/train.json')['rows']:
   # Fresh E7 task padding is already charged in bytes. Reconstruct its
   # exact order with the E7 generator before recording the PAD subtotal.
   if r['task']!='language':
    from . import regressions as reg
    from experiments.joint_request_gate.regressions import fresh_cases
    original=reg.cases;reg.cases=fresh_cases
    try:r['padding_bytes']=generic_padding(r['task'],r)
    finally:reg.cases=original
   add('E7-regressions',r['task']+'/'+r['arm'],r['seed'],r,task=None)
 by_campaign={}
 for r in ledger:
  c=by_campaign.setdefault(r['campaign'],dict(fits=0,bytes=0,targets=0,padding_bytes=0,cpu_seconds=0,local_projection_steps=0,local_presence_steps=0,gate_targets=0,decoder_updates=0));c['fits']+=1
  for k in c:
   if k!='fits':c[k]+=r[k]
 total={k:sum(c[k] for c in by_campaign.values()) for k in next(iter(by_campaign.values()))}
 write(o/'accounting.json',dict(scope='Fitting only, including reset/finish/augmentation/local label construction. Frozen evaluations, compilation, checkpoint I/O and correctness-test updates are separate and not timed by this ledger. Native internal auxiliary work is included in CPU/bytes; scalar native update counts are not instrumented.',total=total,campaigns=by_campaign,fits=ledger))
 sums={c:summarize(read(w/f'exp01e-{c}/audit.json')['rows']) for c in ('formal','1m','5m')}
 generic={}
 for task in ('language','selective_copy','compression'):
  generic[task]={}
  for arm in ('native','d3_joint','gated'):
   rs=[r for r in read(w/'exp01e-regressions/audit.json')['rows'] if r['task']==task and r['arm']==arm]
   generic[task][arm]={k:mean([r['test'][k] for r in rs]) for k in ('token_accuracy','exact','nll')} if task!='language' else dict(nll=mean([r['test_nll'] for r in rs]))
 benchmarks={c:read(w/f'exp01e-{c}/benchmark.json') for c in ('formal','1m','5m')}
 write(o/'summary.json',dict(binding=sums,generic=generic,throughput={c:b['relative'] for c,b in benchmarks.items()},total=total))
 # Keep raw measurements and write-once registrations alongside this report.
 raw=o/'evidence';raw.mkdir(exist_ok=True)
 for folder in sorted(w.glob('exp01e*')):
  if not folder.is_dir():continue
  dest=raw/folder.name;dest.mkdir(exist_ok=True)
  for f in folder.glob('*.json'):
   if f.name.endswith('.cursor.json') or f.name.startswith('rows_'):continue
   shutil.copy2(f,dest/f.name)
 shutil.copy2(w/'exp01e-revision-benchmark.json',raw/'revision-benchmark.json')
 for f in w.glob('exp01e-*.json'):shutil.copy2(f,raw/f.name)
 for f in w.glob('e*-tests.log'):shutil.copy2(f,raw/f.name)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
 colors={'native_all':'#6b7280','frozen_d3':'#0284c7','joint_d3':'#7c3aed','gated_all':'#d97706','gated_no_local':'#b91c1c'}
 fig,ax=plt.subplots(1,2,figsize=(13,4.6),layout='constrained');s=sums['formal']
 arms=list(colors);xs=np.arange(len(arms));ys=[s[k]['primary_mean']*100 for k in arms]
 ax[0].bar(xs,ys,color=[colors[k] for k in arms]);ax[0].axhline(95,color='#166534',ls='--',label='95% primary mean gate')
 for i,k in enumerate(arms):ax[0].scatter([i]*3,np.array(s[k]['primary'])*100,color='black',s=14,zorder=4)
 ax[0].set(xticks=xs,xticklabels=['Native\nall bytes','Frozen\nD3','Joint D3\nanswer loss','Joint E1\nall bytes','E1 without\nlocal labels'],ylim=(0,105),ylabel='Native two-answer exactness (%)',title='Fresh E1 audit • three seeds');ax[0].legend(loc='lower right',fontsize=8)
 for k in ('native_all','frozen_d3','gated_all'):ax[1].plot(range(64),np.array(s[k]['offsets'])*100,label=k.replace('_',' '),color=colors[k],lw=1.7)
 ax[1].axhline(95,color='#166534',ls='--',label='95% required offset mean');ax[1].set(xlabel='Boundary offset (bytes)',ylabel='Mean exactness across seeds (%)',ylim=(0,105),title='All 64 offsets • joint mean 94.11%');ax[1].legend(fontsize=8)
 fig.savefig(o/'SHoEbox_EXP01E_Binding.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 rows=read(w/'exp01e-regressions/train.json')['rows']
 for arm,c in [('native','#6b7280'),('d3_joint','#0284c7'),('gated','#d97706')]:
  rr=[r for r in rows if r['task']=='language' and r['arm']==arm];yy=np.mean([[p['dev_nll'] for p in r['curve']] for r in rr],axis=0);xx=np.mean([[p['cpu_seconds'] for p in r['curve']] for r in rr],axis=0)
  ax[0,0].plot(np.array([p['bytes'] for p in rr[0]['curve']])/1e6,yy,'o-',color=c,label=arm);ax[0,1].plot(xx,yy,'o-',color=c,label=arm)
 ax[0,0].set(xlabel='Training targets (millions)',ylabel='Development NLL (nats/byte)',title='Matched generic language • no local labels');ax[0,1].set(xlabel='Mean fitting CPU seconds',ylabel='Development NLL (nats/byte)',title='Language loss versus fitting CPU time');ax[0,0].legend()
 rows=read(w/'exp01e-formal/train.json')['rows']
 for arm in ('native_all','frozen_d3','gated_all'):
  curves=[r['arms'][arm]['curve'] for r in rows];yy=np.mean([[p['dev']['native_rollout']['native_nll'] for p in cc] for cc in curves],axis=0);xx=np.mean([[p['cpu_seconds'] for p in cc] for cc in curves],axis=0)
  ax[1,0].plot(np.array([p['bytes'] for p in curves[0]])/1e6,yy,'o-',color=colors[arm],label=arm);ax[1,1].plot(xx,yy,'o-',color=colors[arm],label=arm)
 ax[1,0].set(xlabel='Continuation bytes incl. padding (millions)',ylabel='Answer NLL (nats/answer byte)',title='Binding curriculum • 250K warmup separate');ax[1,1].set(xlabel='Mean continuation fitting CPU seconds',ylabel='Answer NLL (nats/answer byte)',title='Binding loss versus fitting CPU time');ax[1,0].legend()
 fig.savefig(o/'SHoEbox_EXP01E_Learning.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
 for campaign,label,style in [('retention-pilot','E1 residual','-'),('switch-pilot','E2 logit switch','--'),('mixture-pilot','E3 transferred mixture',':'),('calibrated-pilot','E4 calibrated bits','-.'),('evidence-gate-pilot','E5 evidence gate','--'),('local-gate-pilot','E6 query-presence gate','-'),('local-gate-control','E6 generic gate',':'),('request-gate-pilot','E7 request gate','-')]:
  path=w/f'exp01e-{campaign}/result.json'
  if not path.exists():continue
  r=read(path)
  if campaign=='retention-pilot':r=read(w/f'exp01e-{campaign}/train.json');start=read(w/'exp01e-pilot/train.json')['rows'][0]['arms']['gated_all']['curve'][-1]['dev']['native_rollout']['native_exact']
  elif 'binding' in r:start=r['binding']['development_audit']['test']['native_rollout']['native_exact'];r=r['retention']
  else:start=r['before']['test']['native_rollout']['native_exact'];r=r['retention']
  yy=[start]+[p['binding_dev']['native_rollout']['native_exact'] for p in r['curve']];xx=[0]+[p['bytes']/1e6 for p in r['curve']]
  ax[0].plot(xx,np.array(yy)*100,style,marker='o',label=label,alpha=.85)
 ax[0].scatter([1],[99.4140625],marker='*',s=130,c='black',label='E4 forced gate on (diagnostic)');ax[0].set(xlabel='Generic language targets (millions)',ylabel='Development native exactness (%)',ylim=(-3,105),title='Binding retention during language learning');ax[0].legend(fontsize=7,loc='center right')
 rev=read(w/'exp01e-revision-benchmark.json');labels=['E1 500K','E1 1M','E1 5M']+list(rev['rows'])[1:];vals=[benchmarks[c]['relative'] for c in ('formal','1m','5m')]+[r['relative'] for k,r in rev['rows'].items() if k!='native']
 if (w/'exp01e-local-gate-benchmark.json').exists():
  b=read(w/'exp01e-local-gate-benchmark.json');labels+=['E7 request'];vals+=[b['relative']]
 ax[1].barh(labels,np.array(vals)*100,color=['#15803d' if x>=.8 else '#b91c1c' for x in vals]);ax[1].axvline(80,color='black',ls='--');ax[1].set(xlabel='Retained native inference throughput (%)',xlim=(0,110),title='Isolated CPU • complete 256-byte output')
 for i,v in enumerate(vals):ax[1].text(v*100+1,i,f'{v*100:.2f}%',va='center',fontsize=8)
 fig.savefig(o/'SHoEbox_EXP01E_Retention_and_Speed.png',dpi=180);plt.close(fig)
 print(json.dumps(dict(total=total,campaigns=by_campaign),indent=2))
if __name__=='__main__':main()
