"""Verify A6 evidence and apply the unchanged registered gates."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from .run import ROOT,sources,write,sha
from .tasks import manifest

def threshold(v,mean,minimum):
 return dict(mean=float(np.mean(v)),minimum=float(min(v)),mean_threshold=mean,minimum_threshold=minimum,pass_gate=bool(np.mean(v)>=mean and min(v)>=minimum))
def offsets(v):
 a=np.asarray(v);mean=float(a.mean());worst=float(a.mean(axis=0).min())
 return dict(mean=mean,worst=worst,mean_threshold=.95,worst_threshold=.9,pass_gate=mean>=.95 and worst>=.9)
def passed(v):
 if isinstance(v,dict):return all(passed(x) for k,x in v.items() if k not in ('mean','minimum','worst','relative_native','mean_threshold','minimum_threshold','worst_threshold','threshold'))
 if isinstance(v,list):return all(passed(x) for x in v)
 return bool(v)
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();w=a.workspace
 read=lambda p:json.loads((w/p).read_text())
 plan=read('a6/plan.json');pre=read('prefit.json');lock=read('a6/selection.json')
 assert plan['source_hashes']==pre['source_hashes']==sources();assert plan['folds']==manifest()==pre['folds']
 assert plan['prefit_sha256']==sha(w/'prefit.json')
 for name,h in lock['checkpoints'].items():assert sha(w/'a6'/name)==h,name
 # Existing public and historical numerical sources and every old report unchanged.
 assert not subprocess.check_output(['git','diff','803dc86e563ff048866023bb536830ce0f28ea98','--','multiscale','shoe_multiscale.py','shoebox_reference.py','experiments/query_first_retention','experiments/chronological_retention','experiments/relative_chronological_retention','experiments/discriminative_keys','reports/exp0305a4','reports/exp0305a5','reports/exp0305a2','reports/exp0305a3','experiments/interference_retention','experiments/joint_cold_gate','experiments/joint_request_gate','reports/exp0305','reports/exp01e8','reports/e8_public'],cwd=ROOT)
 assert not plan['audit_opened'] and not lock['audit_opened']
 results=read('a6/results.json');ext=read('extended/results.json');speed=read('benchmark.json');cur=read('cursors.json');hist=read('history.json')
 assert plan['seeds']==[1,7,19] and plan['arms']==['e8','a5','a6']
 assert plan['fitting_bytes']==20250000 and plan['order_seeds']==[8701,8702]
 assert plan['warmup']==250000 and plan['continuation']==2000000
 assert [r['seed'] for r in results['rows']]==[1,7,19]
 for r in results['rows']:
  assert set(r['arms'])=={'e8','a5','a6'}
  for phase in ('warmup','continuation'):
   fits=[x['warmup'] if phase=='warmup' else x for x in r['arms'].values()]
   assert len({x['cursor']['order_sha256'] for x in fits})==1
   assert all(x['cursor']['bytes']==(250000 if phase=='warmup' else 2000000) for x in fits)
  for x in r['arms'].values():
   assert all(z['presentations']==512 and z['worlds']==128 for z in x['development'].values())
   assert x['binary_regression']['presentations']==512
 for r in ext['rows']:
  for x in r['arms'].values():
   assert all(len(z)==64 for z in x['offsets'].values())
   assert len(x['binary_regression']['offsets'])==64
   assert all(z['presentations']==1024 for z in x['padding'].values())
   assert all(z['presentations']==1024 for z in x['binary_regression']['padding'].values())
 assert not ext['audit_opened'] and cur['count']==18 and cur['all_exact'] and hist['exact'] and len(hist['rows'])==14 and speed['public_e8_exact']
 gates={}
 for arm in ('e8','a5','a6'):
  rows=[r['arms'][arm] for r in results['rows']];erows=[r['arms'][arm] for r in ext['rows']];g={}
  for cond in ('pad0','repeat16','diverse16','compete16','overwrite16'):
   g[cond]=dict(primary=threshold([r['development'][cond]['exact'] for r in rows],.95,.9),
    positions=[threshold([r['development'][cond]['positions'][i] for r in rows],.95,.9) for i in (0,1)],
    both_orders=threshold([r['both_orders'][cond]['both_query_orders_exact'] for r in erows],.9,.85),
    offsets=offsets([r['offsets'][cond] for r in erows]))
  g['delays']={c:threshold([r['development'][c]['exact'] for r in rows],.9,.85) for c in ('repeat64','repeat256','diverse64','diverse256')}
  g['padding']={str(d):threshold([r['padding'][str(d)]['exact'] for r in erows],.9,.85) for d in (0,64,128,256)}
  g['no_query_drop']=threshold([r['lesions']['None']-r['lesions']['no_query'] for r in erows],.2,.2)
  g['reverse_hurts']=all(r['lesions']['reverse']<r['lesions']['None'] for r in erows)
  b=[r['binary_regression'] for r in rows];be=[r['binary_regression'] for r in erows]
  g['binary_regression']=dict(primary=threshold([r['exact'] for r in b],.95,.9),positions=[threshold([r['positions'][i] for r in b],.95,.9) for i in (0,1)],
   both_orders=threshold([r['both_orders']['both_query_orders_exact'] for r in be],.9,.85),offsets=offsets([r['offsets'] for r in be]),
   padding={str(d):threshold([r['padding'][str(d)]['exact'] for r in be],.9,.85) for d in (0,64,128,256)},
   no_query_drop=threshold([r['lesions']['None']-r['lesions']['no_query'] for r in be],.2,.2),reverse_hurts=all(r['lesions']['reverse']<r['lesions']['None'] for r in be))
  g['throughput']=dict(relative_native=speed['rows'][arm]['relative_native'],threshold=.8,pass_gate=speed['rows'][arm]['gate_pass'])
  g['development_pass']=passed(g);gates[arm]=g
 updates=read('update_metrics.json');assert updates['overwrite_qualification_valid'];assert max(updates['shortcuts']['overwrite16'].values())<1
 write(a.out,dict(prefit_and_fit_source_locks_verified=True,locked_artifacts_verified=len(lock['checkpoints']),folds_verified=True,
  protected_sources_unchanged=True,a1_audit_opened=False,a2_audit_opened=False,a3_audit_opened=False,a4_audit_opened=False,a5_audit_opened=False,a6_audit_opened=False,cursors_exact=18,historical_trajectories_exact=14,
  gates=gates,overwrite_fixture_shortcut_valid=True,public_default='qualified E8 unchanged'))
 print('Evidence verified; A6 development pass:',gates['a6']['development_pass'])
if __name__=='__main__':main()
