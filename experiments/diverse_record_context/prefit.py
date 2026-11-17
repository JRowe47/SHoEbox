"""Lock passing correctness receipts, generated folds and shortcuts before fitting."""
import argparse,json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
from .run import sources,write,sha
from .tasks import worlds,cases,make,manifest,interpret,excluded,reserved_a1,reserved_a2,reserved_a3,reserved_a4,reserved_a5,reserved_a6
from .update_metrics import shortcuts

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();w=a.workspace
 if (w/'prefit.json').exists():raise FileExistsError(w/'prefit.json')
 checks={}
 for name in ('prefit_python_final','prefit_native_final','runtime','public_native','history','research','policy_isolation','codebook','streams'):
  receipt=json.loads((w/(name+'_process.json')).read_text());assert receipt['exit_code']==0,name
  checks[name]=dict(receipt_sha256=sha(w/(name+'_process.json')),log_sha256=sha(w/(name+'.txt')))
 assert (w/'prefit_native_final.txt').read_text().count('581 E8 native checks passed')==2
 assert (w/'prefit_native_final.txt').read_text().count('110 E8 request checks passed')==2
 assert (w/'prefit_native_final.txt').read_text().count('83 A1 replacement/lifecycle checks passed')==2
 assert (w/'prefit_native_final.txt').read_text().count('13089 A4 selection checks passed')==2
 assert (w/'prefit_native_final.txt').read_text().count('A5 objective checks passed')==2
 assert (w/'prefit_native_final.txt').read_text().count('A6 presence checks passed')==2
 assert '19 cross-policy rejection checks passed' in (w/'policy_isolation.txt').read_text()
 streams=json.loads((w/'streams.json').read_text());assert streams['passed']
 python_counts=json.loads((w/'python_counts.json').read_text());assert python_counts['passed']
 shortcut={};count=0;both_changes=0
 for split,n in [('train',512),('dev',128)]:
  shortcut[split]={}
  for condition in ('pad0','repeat16','diverse16','compete16','overwrite16'):
   cs=cases(split,n,condition=condition);shortcut[split][condition]=shortcuts(cs)
   assert max(shortcut[split][condition].values())==0
   for c in cs:
    assert interpret(c.ep)==c.expected and c.expected[0]!=c.expected[1];count+=1
  for world in worlds(split,n):
   for changed in (0,1):
    for variant in range(4):
     c=make(world,'overwrite16',variant,updated=changed)
     assert interpret(c.ep)==c.expected and c.expected[0]!=c.expected[1];both_changes+=1
 sets=[{x.group_key for x in worlds(part,n)} for part,n in [('train',512),('dev',128),('audit',256)]]
 assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
 assert not any(excluded(k) for s in sets for k in s)
 binary={w.group_key for w in worlds('dev',128,vocab=2)};assert not binary&(sets[0]|sets[2]|reserved_a1()|reserved_a2()|reserved_a3()|reserved_a4()|reserved_a5()|reserved_a6())
 binary_cases=cases('dev',128,vocab=2)
 binary_shortcuts=shortcuts(binary_cases);assert max(binary_shortcuts.values())==0
 assert all(interpret(c.ep)==c.expected and c.expected[0]!=c.expected[1] for c in binary_cases)
 write(w/'prefit.json',dict(utc=datetime.now(timezone.utc).isoformat(),passed=True,source_hashes=sources(),folds=manifest(),
  stream_sha256=sha(w/'streams.json'),python_counts_sha256=sha(w/'python_counts.json'),per_arm_stream_digests=[dict(arm=r['arm'],phase=r['phase'],digests=r['digests']) for r in streams['rows']],codebook=json.loads((w/'codebook.json').read_text()),correctness=checks,shortcuts=shortcut,interpreter_presentations=count,both_changed_key_interpreter_presentations=both_changes,
  binary_shortcuts=binary_shortcuts,binary_interpreter_presentations=len(binary_cases),binary_dev_overlap_with_three_value_dev=len(binary&sets[1]),audits_scored=False,research_fit_bytes=0))
 print('Pre-fit source/data/correctness/shortcut lock passed.')
if __name__=='__main__':main()
