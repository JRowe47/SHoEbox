"""Frozen occurrence diagnosis; labels identify records only in analysis output."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from .backend import Model
from .tasks import cases
from .run import write,sha
from experiments.joint_cold_gate.adapter import prefix_labels

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 lock=dict(source_sha256=sha(Path(__file__)),checkpoint_hashes={str(p.relative_to(a.runs)):sha(p) for p in a.runs.glob('*/*.qr')},fitting_bytes=0,audit_opened=False)
 write(a.out.with_suffix('.lock.json'),lock);start=time.process_time();rows=[];total=pad=0
 for seed in (1,7,19):
  for arm in ('e8','a8','a9'):
   path=a.runs/str(seed)/(arm+'.qr');original=path.read_bytes();records=[]
   with Model.loads(original,arm) as m,m.clone() as f:
    for condition in ('compete16','overwrite16'):
     for index,c in enumerate(cases('dev',32,condition=condition)):
      f.reset();known={};labels=prefix_labels(c.ep);answer=0
      for t,b in enumerate(c.ep.data[:-1]):
       f.observe(b);total+=1;pad+=int(b==9)
       lab=labels[t]
       if any(lab[1:5]>=0):
        _,_,sig,_=f.state()
        for ph,packed in enumerate(lab[1:5]):
         if packed>=0:known.setdefault(int(packed)//256,[]).append((int(packed)%256,sig.copy(),ph))
       if not c.ep.mask[t+1]:continue
       q=c.query_keys[answer];target=c.expected[answer];count,_,_,units=f.state();selected,decoded=f.decoded();d=f.diagnostic();pred=int(f.log_probs().argmax());items=[]
       for val,sig,ph in known[q]:
        slots=[k for k in range(count) if np.array_equal(units[k,1:],sig)]
        for k in slots:
         items.append(dict(value=val,target=val==target,slot=k,phase=ph,born=int(units[k,0]),recognized=bool(decoded[k,ph,0]>0),decoded_key=int(decoded[k,ph,1]),decoded_value=int(decoded[k,ph,2]),query_distance=float(decoded[k,ph,4]),shortlisted=any(k==s and ph==p for s,p in selected)))
       probe=f.probe() if arm in ('a8','a9') else None
       query_energy=None if probe is None else float(np.square(probe[9690:9706]).sum())
       key_parameters=None if probe is None else probe[325:5525].reshape(5,16,65)
       candidates=[]
       for k in range(count):
        for ph in range(4):
         if decoded[k,ph,0]<=0:continue
         owners=[]
         for owner,occurrences in known.items():
          for val,sig,phase in occurrences:
           if phase==ph and np.array_equal(units[k,1:],sig):owners.append((owner,val))
         key_energy=None if probe is None else float(np.square(np.tanh(key_parameters[ph]@np.r_[units[k,1:]*8,1])).sum())
         candidates.append(dict(key_energy=key_energy,slot=k,phase=ph,born=int(units[k,0]),distance=float(decoded[k,ph,4]),owners=owners,
           recognized_score=float(decoded[k,ph,0]),decoded_key=int(decoded[k,ph,1]),decoded_value=int(decoded[k,ph,2])))
       d_min=min((c['distance'] for c in candidates),default=0.)
       remaining=0.
       for cand in sorted(candidates,key=lambda x:(-x['born'],x['slot'],x['phase'])):
        z=4.*((d_min+1.96)-cand['distance']);cand['match_probability']=float(np.exp(-np.logaddexp(0.,-z)))
        cand['log_weight']=remaining-float(np.logaddexp(0.,-z));remaining-=float(np.logaddexp(0.,z))
       ranked=sorted(candidates,key=lambda x:(-x['log_weight'],x['slot'],x['phase']))[:2]
       if arm in ('a8','a9') and ranked:
        assert [(r['slot'],r['phase']) for r in ranked]==[tuple(x) for x in selected[:len(ranked)]]
        attention=1. if len(ranked)==1 else float(np.exp(-np.logaddexp(0.,ranked[1]['log_weight']-ranked[0]['log_weight'])))
        assert abs(attention-float(d[0]))<1e-7
       mass=dict(latest_target=0.,old_value=0.,other_key=0.,unannotated=0.)
       for j,(k,ph) in enumerate(selected):
        owners=next((x['owners'] for x in candidates if x['slot']==k and x['phase']==ph),[])
        category='latest_target' if (q,target) in owners else ('old_value' if any(kk==q for kk,v in owners) else ('other_key' if owners else 'unannotated'))
        mass[category]+=float(d[0]) if j==0 else 1.-float(d[0])
       records.append(dict(condition=condition,presentation=index,answer=answer,changed=q==c.updated_key,query=q,target=target,prediction=pred,
        query_correct=int(d[1])==q,attention=float(d[0]),selected_keys=[int(d[2]),int(d[3])],selected_values=[int(d[4]),int(d[5])],occurrences=items,candidates=candidates,selected_mass=mass,query_energy=query_energy))
       answer+=1
   assert original==path.read_bytes()
   summaries={}
   for condition in ('compete16','overwrite16'):
    rs=[r for r in records if r['condition']==condition];changed=[r for r in rs if r['changed']]
    metrics=dict(answers=len(rs),target_present=np.mean([any(o['target'] for o in r['occurrences']) for r in rs]),
     target_recognized=np.mean([any(o['target'] and o['recognized'] for o in r['occurrences']) for r in rs]),
     target_shortlisted=np.mean([any(o['target'] and o['shortlisted'] for o in r['occurrences']) for r in rs]),
     native_teacher_fed=np.mean([r['prediction']==r['target'] for r in rs]))
    if changed:
     both=[r for r in changed if any(o['target'] for o in r['occurrences']) and any(not o['target'] for o in r['occurrences'])]
     metrics.update(changed_answers=len(changed),old_and_new_present=len(both)/len(changed),
      changed_native_teacher_fed=np.mean([r['prediction']==r['target'] for r in changed]),
      predicted_old_value=np.mean([any(not o['target'] and o['value']==r['prediction'] for o in r['occurrences']) for r in changed]),
      both_shortlisted=np.mean([any(o['target'] and o['shortlisted'] for o in r['occurrences']) and any(not o['target'] and o['shortlisted'] for o in r['occurrences']) for r in changed]))
     if both:
      gaps=[min(o['query_distance'] for o in r['occurrences'] if o['target'])-min(o['query_distance'] for o in r['occurrences'] if not o['target']) for r in both]
      metrics.update(new_minus_old_distance_mean=np.mean(gaps),new_closer_fraction=np.mean(np.array(gaps)<0),exact_distance_ties=np.mean(np.array(gaps)==0))
    metrics['nearest_identity_correct']=float(np.mean([any(k==r['query'] for k,v in min(r['candidates'],key=lambda c:(c['distance'],-c['born'],c['slot'],c['phase']),default={'owners':[]})['owners']) for r in rs]))
    metrics['selected_mass']={k:float(np.mean([r['selected_mass'][k] for r in rs])) for k in ('latest_target','old_value','other_key','unannotated')}
    summaries[condition]={k:(float(v) if isinstance(v,np.floating) else v) for k,v in metrics.items()}
   rows.append(dict(seed=seed,arm=arm,summary=summaries,records=records));write(a.out,dict(rows=rows,inference_bytes=total,padding_bytes=pad,cpu_seconds=time.process_time()-start,fitting_bytes=0,audit_opened=False))
   print(seed,arm,summaries,flush=True)
if __name__=='__main__':main()
