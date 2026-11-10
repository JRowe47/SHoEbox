"""Record validation outcomes before any new test evaluation."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
plan=json.loads((ROOT/'confirmation_plan.json').read_text());rows=[]
jobs=[dict(size=s,profile=p,seed=k) for s in plan['sizes'] for p,seeds in plan['replication'].items() for k in seeds]+plan.get('extra_jobs',[])
for job in jobs:
   size,profile,seed=job['size'],job['profile'],job['seed']
   d=json.loads((ROOT/'results'/size/f'seed{seed}'/profile/'training.json').read_text())
   assert d['targets']==plan['targets']
   v=[x for x in d['validation'] if x['evaluation_targets']==131072 and x['training_targets']>=2800000]
   assert len(v)==3,(size,seed,profile,len(v))
   rows.append(dict(size=size,profile=profile,seed=seed,mean_last3=sum(x['nll'] for x in v)/3,final=v[-1]['nll']))
summaries={}
for size in plan['sizes']:
 controls={x['seed']:x['mean_last3'] for x in rows if x['size']==size and x['profile']=='control'}
 summaries[size]={}
 for profile in plan['replication']:
  x=[x for x in rows if x['size']==size and x['profile']==profile]
  deltas=[q['mean_last3']-controls[q['seed']] for q in x]
  summaries[size][profile]=dict(mean_nll=sum(q['mean_last3'] for q in x)/len(x),paired_deltas=deltas,mean_delta=sum(deltas)/len(deltas))
selected={size:min([p for p,v in summaries[size].items() if len(v['paired_deltas'])==3],key=lambda p:summaries[size][p]['mean_nll']) for size in plan['sizes']}
out=dict(primary_screen_choice='center_split',rule='Primary comparison preselected on 1M validation; per-size selection compares three-seed mean validation over 2.8M, 2.9M and 3M targets; report all confirmations and controls.',validation=rows,summaries=summaries,selected_per_size=selected,test_accessed=False)
(ROOT/'selection.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summaries,indent=2))
