"""Read-only final check of A9-F1 locks, histories and exact byte/credit ledgers."""
import json, subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from experiments.selective_record_key_credit.run import sources, sha

w = Path(sys.argv[1])
read = lambda name: json.loads((w / name).read_text())
pre, plan = read('prefit.json'), read('a9/plan.json')
assert sources() == pre['source_hashes'] == plan['source_hashes']
assert pre['stream_sha256'] == plan['stream_sha256'] == sha(w / 'streams.json')
assert pre['world_file_sha256'] == sha(w / 'pre_fit_worlds.json')
for lockname in ('frozen_fit.lock.json', 'evaluation.lock.json'):
    for name, digest in read(lockname).items():
        assert sha(w / name) == digest, (lockname, name)
lock = read('frozen_diagnosis.lock.json')
assert lock['evaluation_lock_sha256'] == sha(w / 'evaluation.lock.json')
assert len(lock['checkpoints']) == 18
for name, digest in lock['checkpoints'].items():
    assert sha(w / name) == digest, name
for name, digest in lock['sources'].items():
    if Path(name).name not in ('report.py', 'write_report.py', 'account_correctness.py'):
        assert sha(root / name) == digest, name
for name, digest in read('a9/selection.json')['checkpoints'].items():
    assert sha(w / 'a9' / name) == digest, name
verification = read('verification.json')
assert verification['cursors_exact'] == 18
assert verification['historical_trajectories_exact'] == 14
assert not verification['gates']['a9']['development_pass'] or (root / 'reports/exp0305a9f1/FRESH_QUALIFICATION_REGISTRATION.md').exists()
ledger = read('prefit_correctness_bytes.json')
assert ledger['research_totals']['input_bytes'] == 20250000
assert ledger['resume_totals']['input_bytes'] == 36000
assert len(ledger['research']) == len(ledger['resume']) == 18
totals = dict.fromkeys(('bytes', 'padding_bytes', 'nuisance_bytes', 'replaced_padding_bytes', 'targets', 'answer_targets'), 0)
for row in read('a9/results.json')['rows']:
    for arm, fit in row['arms'].items():
        for phase in (fit['warmup'], fit):
            for key in totals:
                totals[key] += phase['cursor'][key]
assert totals['bytes'] == 20250000
for key in ('padding_bytes', 'nuisance_bytes', 'replaced_padding_bytes'):
    assert totals[key] == ledger['research_totals'][key]
for row in ledger['research']:
    if row['phase'] == 'warmup':
        assert row['counts']['replaced_padding_bytes'] == 0
streams = read('streams.json')['rows']
for phase in ('warmup', 'continuation'):
    rows = {r['arm']: r for r in streams if r['phase'] == phase}
    for key in ('mask', 'labels'):
        assert len({r['digests'][key] for r in rows.values()}) == 1
    for key in ('input', 'target', 'mask', 'labels'):
        assert rows['a8']['digests'][key] == rows['a9']['digests'][key]
    if phase == 'warmup':
        assert rows['e8']['digests'] == rows['a9']['digests']
    else:
        assert rows['e8']['digests']['credit'] == rows['a8']['digests']['credit']
        assert rows['a8']['digests']['credit'] != rows['a9']['digests']['credit']
for name, key in (('conflicts.lock.json', 'checkpoint_hashes'), ('geometry.lock.json', 'checkpoints'), ('geometry_a8.lock.json', 'checkpoints'), ('retention.lock.json', 'checkpoints')):
    for rel, digest in read(name)[key].items():
        assert sha(w / 'a9' / rel) == digest, rel
assert len(list((w / 'a9').glob('*/*.qr'))) == 18
required = ('prefit_python_final', 'prefit_native_final', 'runtime', 'public_native', 'history', 'research', 'streams', 'policy_isolation', 'codebook', 'f1_allocation', 'fit', 'extended', 'cursors', 'update_metrics', 'benchmark', 'signatures', 'conflicts', 'retention', 'geometry', 'geometry_a8', 'survivors', 'coverage', 'correctness_accounting')
assert all(read(name + '_process.json')['exit_code'] == 0 for name in required)
assert pre['utc'] < plan['utc']
local = '00d5cb393c075710925a6fd9add8606915ca3a5a'
published = '7dc8b5c275d00b469f068d788064b63025833120'
assert plan['prefit_commit'] == local
git = lambda *args: subprocess.check_output(['git', *args], cwd=root, text=True).strip()
assert git('rev-parse', local + '^{tree}') == git('rev-parse', published + '^{tree}')
assert git('show', local + ':reports/exp0305a9/prefit/prefit.json') == (w / 'prefit.json').read_text().strip()
assert sha(w / 'streams.json') == sha(w / 'prior_a9/streams.json')
f1 = read('f1_allocation.json')
assert f1['passed']
assert not git('diff', 'fa54cbd1c8c90be756ea4da8cfef2ccb735ed183', '--', 'reports/exp0305a9/A9_F1_REGISTRATION.md', 'reports/exp0305a9/SHoEbox_EXP0305_A9_Report.md', 'reports/exp0305a9/RESUME.md', 'reports/exp0305a9/evidence', 'reports/exp0305a9/accounting.json')
tracked = git('ls-files').splitlines()
assert not [p for p in tracked if p.startswith(('experiments/selective_record_key_credit/', 'reports/exp0305a9f1/')) and Path(p).suffix in ('.qr', '.shoe', '.so', '.o', '.npz', '.npy', '.pt', '.bin')]
result = dict(passed=True, locked_source_files=len(pre['source_hashes']), actual_checkpoint_cursor_pairs=18, research_bytes=20250000, correctness_cursor_bytes=36000, original_primary_streams_unchanged=True, binary_allocation_verified=True, counts=totals, audits_opened=False, local_prefit_commit=local, published_identical_prefit_commit=published)
(w / 'delivery_verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
