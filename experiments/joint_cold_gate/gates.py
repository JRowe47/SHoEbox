"""Evaluate the unchanged D/E thresholds; never select weights or data."""
import numpy as np

def binding_gate(rows):
 checks=[]
 def add(name,value,minimum,strict=False):
  checks.append(dict(name=name,value=float(value),minimum=minimum,pass_gate=bool(value>minimum if strict else value>=minimum)))
 for key,name in [('native_exact','primary'),('first_answer','first_answer'),('second_answer','second_answer')]:
  vals=[r['test']['native_rollout'][key] for r in rows];add(name+'_mean',np.mean(vals),.95);add(name+'_min_seed',min(vals),.90)
 vals=[r['counterfactual']['both_query_orders_exact'] for r in rows];add('both_orders_mean',np.mean(vals),.90);add('both_orders_min_seed',min(vals),.85)
 offsets=np.mean([[r['offsets'][i]['exact'] for i in range(64)] for r in rows],axis=0)
 add('offset_mean',np.mean(offsets),.95);add('worst_offset_seed_mean',min(offsets),.90)
 for i,delay in enumerate((0,64,128,256)):
  vals=[r['delays'][i]['exact'] for r in rows];add(f'delay_{delay}_mean',np.mean(vals),.90);add(f'delay_{delay}_min_seed',min(vals),.85)
 if all('lesions' in r for r in rows):
  add('query_removal_drop',np.mean([r['lesion_unmodified']['native_exact']-r['lesions']['no_query']['native_exact'] for r in rows]),.20)
  add('reverse_drop',np.mean([r['lesion_unmodified']['native_exact']-r['lesions']['reverse']['native_exact'] for r in rows]),0,True)
 return dict(passed=all(r['pass_gate'] for r in checks),checks=checks)

def generic_gate(rows,audit=False):
 checks=[]
 for task in ('language','selective_copy','compression'):
  scores={}
  for arm in ('native','d3_joint','local_gate'):
   selected=[r for r in rows if r['task']==task and r['arm']==arm]
   if len(selected)!=3:raise ValueError('three seeds required')
   if task=='language':values=[r['test_nll'] if audit else r['curve'][-1]['dev_nll'] for r in selected]
   else:values=[r['test']['token_accuracy'] if audit else r['curve'][-1]['dev']['token_accuracy'] for r in selected]
   scores[arm]=dict(seeds=[r['seed'] for r in selected],values=values,mean=float(np.mean(values)))
  delta=scores['local_gate']['mean']-scores['native']['mean']
  checks.append(dict(task=task,scores=scores,candidate_minus_native=delta,margin=.01,pass_gate=delta<=.01 if task=='language' else delta>=-.01))
 return dict(passed=all(r['pass_gate'] for r in checks),checks=checks)
