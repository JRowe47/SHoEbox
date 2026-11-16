"""Summarize frozen geometry; performs no inference or fitting."""
import argparse,json
from pathlib import Path
import numpy as np

def summary(xs):
 return dict(count=len(xs),mean=float(np.mean(xs)),minimum=float(min(xs)),maximum=float(max(xs)),quantiles=np.quantile(xs,[.1,.5,.9]).tolist()) if xs else {}
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();raw=json.loads((a.workspace/'conflicts.json').read_text());out={}
 for arm in ('a3','a4'):
  out[arm]={}
  for condition in ('compete16','overwrite16'):
   rs=[r for row in raw['rows'] if row['arm']==arm for r in row['records'] if r['condition']==condition]
   cats={k:[] for k in ('latest_target','old_same_key','other_key')};wrong_newer=[]
   for r in rs:
    target_born=max((o['born'] for o in r['occurrences'] if o['target']),default=-1)
    newer=[]
    for c in r['candidates']:
     owners=[tuple(x) for x in c['owners']]
     cat='latest_target' if (r['query'],r['target']) in owners else ('old_same_key' if any(k==r['query'] for k,v in owners) else 'other_key')
     cats[cat].append(c)
     if cat=='other_key' and c['born']>target_born:newer.append(c)
    wrong_newer.append(any(c['match_probability']>.5 for c in newer))
   out[arm][condition]=dict(answers=len(rs),query_energy=summary([r['query_energy'] for r in rs]),
    newer_other_key_match_above_half=float(np.mean(wrong_newer)),
    categories={name:dict(distance=summary([c['distance'] for c in cs]),key_energy=summary([c['key_energy'] for c in cs]),
        match_probability=summary([c['match_probability'] for c in cs]),match_above_half=float(np.mean([c['match_probability']>.5 for c in cs]))) if cs else {} for name,cs in cats.items()})
 out['scope']='Read-only summaries of the locked teacher-fed trace; no new inference, fit, threshold selection or coefficient search.'
 (a.workspace/'matching_geometry.json').write_text(json.dumps(out,indent=2)+'\n')
 for arm in ('a3','a4'):
  for cond,r in out[arm].items():print(arm,cond,'query energy',r['query_energy']['mean'],'newer wrong match',r['newer_other_key_match_above_half'],[(k,v.get('distance',{}).get('mean'),v.get('key_energy',{}).get('mean'),v.get('match_above_half')) for k,v in r['categories'].items()])
if __name__=='__main__':main()
