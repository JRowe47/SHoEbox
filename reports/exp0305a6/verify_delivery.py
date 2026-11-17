"""Data-only final verification; no model construction, fitting or new inference."""
import hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime

ROOT=Path(__file__).resolve().parents[2]
W=Path(sys.argv[1]).resolve()
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
subprocess.run([sys.executable,'-m','experiments.record_presence_cost.verify_evidence',
 '--workspace',str(W),'--out',str(W/'verification_final.json')],cwd=ROOT,check=True)
assert read(W/'verification_final.json')==read(W/'verification.json')
checked=0
for name in ('frozen_fit.lock.json','evaluation.lock.json'):
 d=read(W/name)
 for rel,h in d['files'].items():assert sha(W/rel)==h,(name,rel);checked+=1
 assert not d['audit_opened']
for stem,source in [('conflicts','conflicts.py'),('retention','trace_retention.py'),
                    ('geometry','trace_geometry.py'),('survivors','trace_survivors.py')]:
 d=read(W/(stem+'.lock.json'))
 assert sha(ROOT/'experiments/record_presence_cost'/source)==d['source_sha256']
 for rel,h in d.get('checkpoints',d.get('checkpoint_hashes',{})).items():
  assert sha(W/'a6'/rel)==h,(stem,rel);checked+=1
 assert not d['audit_opened'] and d['fitting_bytes']==0
d=read(W/'coverage.json')
assert d['source_sha256']==sha(ROOT/'reports/exp0305a6/analyze_coverage.py')
assert d['registration_sha256']==sha(ROOT/'reports/exp0305a6/DIAGNOSTIC_ATTRIBUTION_ADDENDUM.md')
assert d['input_sha256']==sha(W/'a6/results.json')
assert d['fitting_bytes']==d['inference_bytes']==0
pre=read(W/'prefit.json');fit=read(W/'fit_process.json')
for stem,h in pre['correctness'].items():
 assert sha(W/(stem+'_process.json'))==h['receipt_sha256']
 assert sha(W/(stem+'.txt'))==h['log_sha256']
 assert read(W/(stem+'_process.json'))['completed_utc']<fit['started_utc']
 checked+=2
assert pre['utc']<fit['started_utc']
prefit_git=subprocess.check_output(['git','show','1ddb724:reports/exp0305a6/prefit/prefit.json'],cwd=ROOT)
assert prefit_git==(W/'prefit.json').read_bytes()
committed=datetime.fromisoformat(subprocess.check_output(['git','show','-s','--format=%cI','1ddb724'],cwd=ROOT,text=True).strip())
assert committed<datetime.fromisoformat(fit['started_utc'])
receipts={p.name:read(p) for p in W.glob('*_process.json')}
timing=receipts['benchmark_process.json'];overlaps=[]
for name,r in receipts.items():
 assert r['exit_code']==0,(name,r['exit_code'])
 if name!='benchmark_process.json' and r['started_utc']<timing['completed_utc'] and r['completed_utc']>timing['started_utc']:
  overlaps.append(name)
assert not overlaps,overlaps
ext=read(W/'extended/results.json');lesions={'None','no_head','no_query','no_state','uniform','reverse','unpair','gate_off','gate_on'}
for row in ext['rows']:
 for v in row['arms'].values():
  assert set(v['lesions'])==lesions and set(v['binary_regression']['lesions'])==lesions
assert len(list((W/'a6').glob('*/*.qr')))==18
assert len(list((W/'a6').glob('*/*.cursor.json')))==18
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
out={'source_folds_and_original_gates_verified':True,'additional_lock_entries_verified':checked,
 'prefit_commit_and_receipts_precede_fitting':True,'snapshot_cursor_pairs':18,
 'all_registered_lesions_present':True,'metered_process_receipts':len(receipts),
 'failed_metered_attempts':0,'timing_overlapping_processes':overlaps,
 'fitting_bytes':20250000,'additional_fitting_bytes':0,'additional_inference_bytes':0,
 'development_pass':False,'public_default':'qualified E8 unchanged','a1_through_a6_audits_opened':False}
(W/'delivery_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
