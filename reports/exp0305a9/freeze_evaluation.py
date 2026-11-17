"""Seal complete evaluation outcomes before any frozen diagnostic trace."""
import json, sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from experiments.selective_record_key_credit.run import sha, sources, write

w = Path(sys.argv[1])
read = lambda name: json.loads((w / name).read_text())
assert sources() == read('a9/plan.json')['source_hashes']
for name, digest in read('frozen_fit.lock.json').items():
    assert sha(w / name) == digest, name
for name in ('fit', 'extended', 'cursors', 'update_metrics',
             'correctness_accounting', 'benchmark'):
    assert read(name + '_process.json')['exit_code'] == 0, name
paths = ('a9/results.json', 'a9/selection.json', 'extended/results.json',
         'update_metrics.json', 'cursors.json', 'benchmark.json',
         'prefit_correctness_bytes.json')
assert not (w / 'evaluation.lock.json').exists()
assert not (w / 'frozen_diagnosis.lock.json').exists()
write(w / 'evaluation.lock.json', {name: sha(w / name) for name in paths})
# The inherited survivor tracer's narrow glob covers A8 only. This separate
# complete checkpoint lock, verified again after all traces, covers every
# actual A8/A9 diagnostic input without changing the pre-fit source closure.
write(w / 'frozen_diagnosis.lock.json', dict(
    checkpoints={str(p.relative_to(w)): sha(p)
                 for p in sorted((w / 'a9').glob('*/*.qr'))},
    sources={str(p.relative_to(root)): sha(p)
             for p in sorted((root / 'experiments/selective_record_key_credit').glob('*.py'))},
    evaluation_lock_sha256=sha(w / 'evaluation.lock.json'),
    fitting_bytes=0, audit_opened=False))
print('All final outcomes and 18 diagnostic input checkpoints locked.')
