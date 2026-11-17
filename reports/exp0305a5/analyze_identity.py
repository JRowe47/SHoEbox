"""Read-only attribution of locked A5 geometry; no model calls or new policy."""
import json,sys
from pathlib import Path
import numpy as np
from experiments.discriminative_keys.test_objective import code
w=Path(sys.argv[1]);geo=json.loads((w/'geometry.json').read_text());out={}
book=np.array([[code(b,r) for r in range(16)] for b in range(256)])
def stats(xs):return dict(count=len(xs),mean=float(np.mean(xs)),min=float(min(xs)),max=float(max(xs))) if xs else {}
for condition in ('compete16','overwrite16'):
 qerr=[];kerr=[];wrong=[];other=[];within=[];wrong_detail=[]
 for row in geo['rows']:
  for r in row['records']:
   if r['condition']!=condition:continue
   q=r['query'];qerr.append(float(np.square(np.array(r['query_vector'])-book[q]).sum()))
   nearest=min(r['candidates'],key=lambda c:(c['query_distance'],-c['born'],c['slot'],c['phase']))
   for c in r['candidates']:
    for key,value in c['owners']:
     kerr.append(float(np.square(np.array(c['key'])-book[key]).sum()))
     if key!=q:other.append(int(np.count_nonzero(book[q]!=book[key])))
     else:within.append(c['query_distance'])
   if not r['nearest_identity_correct']:
    for key,value in nearest['owners']:
     h=int(np.count_nonzero(book[q]!=book[key]));wrong.append(h);wrong_detail.append(dict(seed=row['seed'],presentation=r['presentation'],query=q,wrong_key=key,hamming=h,query_distance=nearest['query_distance']))
 out[condition]=dict(query_reconstruction_squared_error=stats(qerr),record_reconstruction_squared_error=stats(kerr),same_identity_query_record_distance=stats(within),wrong_nearest_code_hamming=stats(wrong),all_other_code_hamming=stats(other),wrong_details=wrong_detail)
out['scope']='Same locked geometry; no new input, fit, codebook or inference candidate evaluation.'
(w/'identity_attribution.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({c:{k:v for k,v in r.items() if k!='wrong_details'} for c,r in out.items() if isinstance(r,dict)},indent=2))
