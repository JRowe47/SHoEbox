"""Read-only final audit of source/data/result locks and complete cost ledgers."""
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from experiments.balanced_record_context.run import sources,sha
w=Path(sys.argv[1]);read=lambda n:json.loads((w/n).read_text())
pre=read('prefit.json');plan=read('a8/plan.json');assert sources()==pre['source_hashes']==plan['source_hashes']
assert pre['stream_sha256']==plan['stream_sha256']==sha(w/'streams.json')
assert pre['world_file_sha256']==sha(w/'pre_fit_worlds.json')
for n in ('frozen_fit.lock.json','evaluation.lock.json'):
 for name,digest in read(n).items():assert sha(w/name)==digest,(n,name)
diagnosis_lock=read('frozen_diagnosis.lock.json')
assert diagnosis_lock['evaluation_lock_sha256']==sha(w/'evaluation.lock.json')
assert len(diagnosis_lock['checkpoints'])==18
for name,digest in diagnosis_lock['checkpoints'].items():assert sha(w/name)==digest,name
for name,digest in diagnosis_lock['sources'].items():
 if Path(name).name not in ('report.py','write_report.py','verify_evidence.py','account_correctness.py'):
  assert sha(root/name)==digest,name
for name,digest in read('a8/selection.json')['checkpoints'].items():assert sha(w/'a8'/name)==digest
assert read('verification.json')['cursors_exact']==18
assert read('verification.json')['historical_trajectories_exact']==14
assert not read('verification.json')['gates']['a8']['development_pass'] or (root/'reports/exp0305a8/FRESH_QUALIFICATION_REGISTRATION.md').exists()
ledger=read('prefit_correctness_bytes.json');assert ledger['research_totals']['input_bytes']==20250000
assert ledger['resume_totals']['input_bytes']==36000
assert len(ledger['research'])==18 and len(ledger['resume'])==18
fits=read('a8/results.json');totals={k:0 for k in ('bytes','padding_bytes','nuisance_bytes','replaced_padding_bytes','targets','answer_targets')}
for row in fits['rows']:
 for arm,fit in row['arms'].items():
  for phase in (fit['warmup'],fit):
   for key in totals:totals[key]+=phase['cursor'][key]
assert totals['bytes']==20250000
for key in ('padding_bytes','nuisance_bytes','replaced_padding_bytes'):assert totals[key]==ledger['research_totals'][key]
assert totals['replaced_padding_bytes']==3*38797
assert all(r['counts']['replaced_padding_bytes']==0 for r in ledger['research'] if r['arm']!='a8' or r['phase']=='warmup')
for name,hashes in [('conflicts.lock.json','checkpoint_hashes'),('geometry.lock.json','checkpoints'),('retention.lock.json','checkpoints')]:
 lock=read(name)
 for rel,digest in lock[hashes].items():assert sha(w/'a8'/rel)==digest
assert len(list((w/'a8').glob('*/*.qr')))==18
assert all(read(n+'_process.json')['exit_code']==0 for n in ('prefit_python_final','prefit_native_final','runtime','public_native','history','research','streams','policy_isolation','codebook','fit','extended','cursors','update_metrics','benchmark','signatures','conflicts','retention','geometry','survivors','coverage','correctness_accounting'))
assert pre['utc']<plan['utc']
assert plan['prefit_commit']==subprocess.check_output(['git','rev-parse','d622eb0'],cwd=root,text=True).strip()
for phase in ('warmup','continuation'):
 stream_rows=[r for r in read('streams.json')['rows'] if r['phase']==phase]
 assert stream_rows[0]['digests']==stream_rows[1]['digests']
 for k in ('mask','labels'):assert len({r['digests'][k] for r in stream_rows})==1
 if phase=='warmup':assert stream_rows[0]['digests']==stream_rows[2]['digests']
actual=read('streams.json')['rows'][-1]
assert actual['cursor']['episodes']==11088
for block in ('pad','diverse'):
 assert {r['phase'] for r in actual['coverage'] if r['regime']==block}==set(range(64))
subprocess.run(['git','cat-file','-e','d622eb08e4ab5ee72315c739f5439255665b373f^{commit}'],cwd=root,check=True)
tracked=subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()
bad=[p for p in tracked if p.startswith(('experiments/balanced_record_context/','reports/exp0305a8/')) and Path(p).suffix in ('.qr','.shoe','.so','.o','.npz','.npy','.pt','.bin')];assert not bad,bad
result=dict(passed=True,locked_source_files=len(pre['source_hashes']),actual_checkpoint_cursor_pairs=18,research_bytes=20250000,correctness_cursor_bytes=36000,per_arm_streams_verified=True,all_prior_policies_preserved=True,counts=totals,audits_opened=False,prefit_commit='d622eb08e4ab5ee72315c739f5439255665b373f')
(w/'delivery_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
