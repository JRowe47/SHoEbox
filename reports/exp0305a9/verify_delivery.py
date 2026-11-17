"""Verify blocked A9 integrity; never runs a model or changes the registration."""
import json,sys,subprocess,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from experiments.selective_record_key_credit.run import sources,sha
w=Path(sys.argv[1]);read=lambda n:json.loads((w/n).read_text());lock=read('blocked_source.lock.json')
assert lock['source_hashes']==sources()
for name,digest in lock['locked_external_files'].items():assert sha(w/name)==digest,name
assert not (w/'prefit.json').exists() and not (w/'a9').exists()
assert not list(w.rglob('*.qr')) and not list(w.rglob('*.shoe'))
f=read('fold_feasibility.json');assert not f['passed'] and f['required_binary_dev_worlds']==128 and f['available_binary_dev_worlds']==96
assert f['data_sha256']==sha(w/'world_feasibility_data.json')
assert not f['audit_answers_evaluated'] and f['research_fitting_bytes']==0
assert read('fold_requirement_recheck_process.json')['exit_code']==1
assert 'only 96 dev worlds at pool 64' in (w/'fold_requirement_recheck.txt').read_text()
for name in ('runtime','research','history','public_native','prefit_native_final','policy_isolation','streams','codebook','fold_feasibility'):
 assert read(name+'_process.json')['exit_code']==0,name
assert read('prefit_python_initial_process.json')['exit_code']==1
p=read('python_counts_failed1.json');assert p['tests']==20 and not p['passed']
assert (w/'prefit_python_initial.txt').read_text().count('... ok')==19
assert (w/'prefit_native_final.txt').read_text().count('57280 A9 masked credit checks passed')==2
assert '23 cross-policy rejection checks passed' in (w/'policy_isolation.txt').read_text()
h=read('history.json');assert h['exact'] and len(h['rows'])==14
s=read('streams.json');assert s['passed'] and len(s['rows'])==6
for phase in ('warmup','continuation'):
 rows={r['arm']:r for r in s['rows'] if r['phase']==phase}
 for field in ('input','target','mask','labels'):assert rows['a8']['digests'][field]==rows['a9']['digests'][field]
 for field in ('order_rng','augmentation_rng','order','current','episodes','order_sha256','bytes','targets','answer_targets','finish_events'):
  assert all(r['cursor'][field]==rows['a8']['cursor'][field] for r in rows.values())
 if phase=='warmup':assert all(r['digests']==rows['e8']['digests'] for r in rows.values())
 else:
  assert rows['a8']['digests']['credit']!=rows['a9']['digests']['credit']
  t=rows['a9']['counts'];assert t['record_key_executed']==14524 and t['record_key_omitted']==14307 and t['local_projection_steps']==817896
  assert all(r['counts']['positive_record_keys']==28831 for r in rows.values())
  for event in rows['a9']['credit_counts']:
   assert event['omitted']==(event['eligible'] if event['regime']=='diverse' else 0)
# Every starting tracked file is preserved except three explicit current-status docs.
allowed={'AGENTS.md','DEVELOPMENT_HANDOFF.md','docs/capabilities/SHoEbox_Minimal_Capability_Matrix.md'}
changed=subprocess.check_output(['git','diff','--name-only','3c24b6623e8f03ae39f2d7d6266609bfcc9344ba'],cwd=root,text=True).splitlines()
assert all(n in allowed or n.startswith(('experiments/selective_record_key_credit/','reports/exp0305a9/')) for n in changed),changed
historical=subprocess.check_output(['git','diff','3c24b6623e8f03ae39f2d7d6266609bfcc9344ba','--','multiscale','shoe_multiscale.py','shoebox_reference.py','experiments/balanced_record_context','reports/exp0305a8'],cwd=root)
assert not historical
for folder in ('experiments/selective_record_key_credit','reports/exp0305a9'):
 bad=[str(p) for p in (root/folder).rglob('*') if p.suffix in ('.qr','.shoe','.so','.o','.npz','.npy','.pt','.bin')];assert not bad,bad
assert (root/'reports/exp0305a9/A9_F1_REGISTRATION.md').exists()
a=read('accounting.json');assert a['research_fitting_bytes']==0 and a['successful_python_control_bytes']==56044
result=dict(passed=True,status='blocked_before_fitting',locked_sources=len(lock['source_hashes']),original_registration_unchanged=True,required_binary_worlds=128,available_binary_worlds=96,source_stream_credit_locks_verified=True,passing_prefit_receipt=False,research_fitting_bytes=0,fitted_checkpoint_pairs=0,all_prior_policies_preserved=True,audits_opened=False,all_required_development_gates_run=False,next_registration='A9_F1_REGISTRATION.md')
(w/'delivery_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
