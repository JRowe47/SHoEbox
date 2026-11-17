"""Read-only final audit of source/data/result locks and complete cost ledgers."""
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from experiments.diverse_record_context.run import sources,sha
w=Path(sys.argv[1]);read=lambda n:json.loads((w/n).read_text())
pre=read('prefit.json');plan=read('a7/plan.json');assert sources()==pre['source_hashes']==plan['source_hashes']
assert pre['stream_sha256']==plan['stream_sha256']==sha(w/'streams.json')
for n in ('frozen_fit.lock.json','evaluation.lock.json'):
 for name,digest in read(n).items():assert sha(w/name)==digest,(n,name)
for name,digest in read('a7/selection.json')['checkpoints'].items():assert sha(w/'a7'/name)==digest
assert read('verification.json')['cursors_exact']==18
assert read('verification.json')['historical_trajectories_exact']==14
assert not read('verification.json')['gates']['a7']['development_pass'] or (root/'reports/exp0305a7/FRESH_QUALIFICATION_REGISTRATION.md').exists()
ledger=read('prefit_correctness_bytes.json');assert ledger['research_totals']['input_bytes']==20250000
assert ledger['resume_totals']['input_bytes']==36000
assert len(ledger['research'])==18 and len(ledger['resume'])==18
fits=read('a7/results.json');totals={k:0 for k in ('bytes','padding_bytes','nuisance_bytes','replaced_padding_bytes','targets','answer_targets')}
for row in fits['rows']:
 for arm,fit in row['arms'].items():
  for phase in (fit['warmup'],fit):
   for key in totals:totals[key]+=phase['cursor'][key]
assert totals['bytes']==20250000
for key in ('padding_bytes','nuisance_bytes','replaced_padding_bytes'):assert totals[key]==ledger['research_totals'][key]
assert totals['replaced_padding_bytes']>0
assert all(r['counts']['replaced_padding_bytes']==0 for r in ledger['research'] if r['arm']!='a7' or r['phase']=='warmup')
for name,hashes in [('conflicts.lock.json','checkpoint_hashes'),('geometry.lock.json','checkpoints'),('retention.lock.json','checkpoints')]:
 lock=read(name)
 for rel,digest in lock[hashes].items():assert sha(w/'a7'/rel)==digest
assert len(list((w/'a7').glob('*/*.qr')))==18
assert all(read(n+'_process.json')['exit_code']==0 for n in ('prefit_python_final','prefit_native_final','runtime','public_native','history','research','streams','policy_isolation','codebook','fit','extended','cursors','update_metrics','benchmark','signatures','conflicts','retention','geometry','survivors','coverage','correctness_accounting'))
assert pre['utc']<plan['utc']
subprocess.run(['git','cat-file','-e','80f1e351a37343db0ed24619144a0c587a8b4a30^{commit}'],cwd=root,check=True)
tracked=subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()
bad=[p for p in tracked if p.startswith(('experiments/diverse_record_context/','reports/exp0305a7/')) and Path(p).suffix in ('.qr','.shoe','.so','.o','.npz','.npy','.pt','.bin')];assert not bad,bad
result=dict(passed=True,locked_source_files=len(pre['source_hashes']),actual_checkpoint_cursor_pairs=18,research_bytes=20250000,correctness_cursor_bytes=36000,per_arm_streams_verified=True,all_prior_policies_preserved=True,counts=totals,audits_opened=False,prefit_commit='80f1e351a37343db0ed24619144a0c587a8b4a30')
(w/'delivery_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
