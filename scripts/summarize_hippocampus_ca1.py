"""Audit matched CA1 continuations and append their results to the main report."""
import hashlib,json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.summarize_hippocampus import compact_measurements,tensors
from scripts.run_memory_selectivity import write_json


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b


def main():
 torch.set_num_threads(1)
 root=ROOT/'artifacts/hippocampus_binding';base=root/'ca1';ev=base/'evaluation'
 path=ROOT/'experiments/text_regions/results/hippocampus_binding/summary.json'
 report=read(path);reference=read(ROOT/'artifacts/hippocampus/A/initial.json')
 initial=[torch.load(base/n/'initial.pt',weights_only=True,map_location='cpu') for n in ('legacy','bounded')]
 fields=('weights','optimizer','language_state','language_stream','rng','next_document')
 proof={field:equal(initial[0][field],initial[1][field]) for field in fields}
 if not all(proof.values()):raise ValueError('Continuation starting states differ.')
 result=dict(initial_state_identity=proof,cohort=dict(seed=1699,start=160000,
     pair_templates=8,single_templates=16,paired_gaps=[0,32,64],retention_gap=64),models={},controls={})
 schedules=[]
 for name in ('legacy','bounded'):
  checkpoint=base/name/'last.pt';saved=torch.load(checkpoint,weights_only=True,map_location='cpu')
  selected=torch.load(base/name/'best.pt',weights_only=True,map_location='cpu')
  history=[json.loads(s) for s in (base/name/'history.jsonl').read_text().splitlines()]
  if saved['step']!=16 or len(history)!=16:raise ValueError('Incomplete continuation.')
  for field in ('weights','optimizer','language_state'):
   if any(not bool(torch.isfinite(t).all()) for t in tensors(saved[field])):raise ValueError('Nonfinite checkpoint.')
  schedules.append([{k:r[k] for k in ('step','next_document','mode','gap','totals')} for r in history])
  val=history[-1]['validation'];parent=reference['cohorts']
  gates=dict(paired=all(val['paired'][str(g)]['accuracy']>=.5 for g in (0,32)),
    single=all(val['cohorts'][f'single_{g}']['accuracy']>=parent[f'single_{g}']['accuracy']-.02 for g in (0,32)),
    correction=all(val['cohorts'][f'correction_{g}']['accuracy']>=parent[f'correction_{g}']['accuracy']-.03 for g in (0,32)),
    language=val['language']['nll']<=reference['language']['nll']+.05)
  row=dict(checkpoint_sha256=digest(checkpoint),parent_sha256=saved['protocol']['parent_sha256'],
    selected_update=selected['step'],selected_validation_score=selected['best'],
    validation=val,initial_validation=read(base/name/'initial.json'),promotion_gates=gates,promoted=all(gates.values()),
    totals=saved['totals'],training_seconds=sum(r['seconds'] for r in history),
    peak_mib=max(r['peak_mib'] for r in history),measurements={})
  for part in ('pairs','retention','language'):
   data=read(ev/(name+'_'+part+'.json'));identity=data['identity']
   if identity['checkpoint_sha256']!=digest(checkpoint) or (identity['seed'],identity['start'])!=(1699,160000):
    raise ValueError('Evaluation identity mismatch.')
   row['measurements'].update(compact_measurements(data['measurements']))
  result['models'][name]=row
 if schedules[0]!=schedules[1]:raise ValueError('Continuation source schedules differ.')
 if result['models']['legacy']['parent_sha256']!=result['models']['bounded']['parent_sha256']:
  raise ValueError('Parent identity mismatch.')
 result['source_schedules_identical']=True
 for name in ('bounded_blocked','bounded_legacy_policy'):
  data=read(ev/(name+'.json'))
  if data['identity']['checkpoint_sha256']!=result['models']['bounded']['checkpoint_sha256']:
   raise ValueError('Intervention checkpoint mismatch.')
  result['controls'][name]=compact_measurements(data['measurements'])
 for name in ('legacy','bounded'):
  diagnostic=read(base/(name+'_audit.json'))
  if diagnostic['checkpoint_sha256']!=result['models'][name]['checkpoint_sha256']:
   raise ValueError('Diagnostic checkpoint mismatch.')
  result['models'][name]['diagnostic']=diagnostic['measurements'][0]
 tests=(root/'ca1_tests.log').read_text()
 if not tests.rstrip().endswith('OK'):raise ValueError('CA1 checks did not pass.')
 report['tests']['ca1']=5;report['ca1']=result
 report['legacy_compatibility']=read(root/'legacy_compatibility.json')
 report['retain_parent']=not result['models']['bounded']['promoted']
 write_json(path,report)
 write_json(path.with_name('ca1_summary.json'),result)
 print(json.dumps({n:dict(pairs=[r['measurements'][f'pairs_{g}']['both_correct'] for g in (0,32,64)],
   single64=r['measurements']['single_64']['accuracy'],correction64=r['measurements']['correction_64']['accuracy'],
   promoted=r['promoted']) for n,r in result['models'].items()}))

if __name__=='__main__':main()
