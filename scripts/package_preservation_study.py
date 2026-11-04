"""Audit/export the study and optionally build a complete portable ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.preservation_compare import compare, audit_convergence
from experiments.text_regions.residual_adapter import write_json, write_bytes
from experiments.text_regions.emission_adapter import sha


def archive(path, source, status):
    if path.exists():
        raise ValueError('Archive already exists.')
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    files = {ROOT / p for p in tracked if p and (ROOT / p).is_file()}
    # Include exact dependencies of the old audit as well as all new study files.
    for run in status['runs']:
        files.add(ROOT / run['checkpoint'])
        files.add(ROOT / run['reference'])
        for directory in (source / run['name'], ROOT / run['residual_run'], ROOT / run['prior_run']):
            files.update(p for p in directory.iterdir() if p.is_file())
    for directory in (source, ROOT / 'artifacts/residual_adapter_v1', ROOT / 'artifacts/emission_adapter_v1'):
        files.update(p for p in directory.iterdir() if p.is_file())
    convergence = ROOT / 'artifacts/preservation_convergence_v1'
    if convergence.exists():
        files.update(p for p in convergence.rglob('*') if p.is_file())
    entries = [{'path': str(p.relative_to(ROOT)), 'size_bytes': p.stat().st_size, 'sha256': sha(p)}
               for p in sorted(files)]
    manifest = {'schema_version': 1,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        'files': entries}
    restore = '''# SHoEbox shared-answer preservation study

This package contains source, reports, all twelve frozen model checkpoints,
feature caches, the earlier residual controls, all 144 new fitted corrections,
validation selection locks, and complete synthetic prediction rows.

From this directory (Python 3.10+, PyTorch, NumPy):

```bash
python -m pip install -r requirements.txt
python -m experiments.text_regions.preservation_compare \\
  --source artifacts/shared_preservation_v1 \\
  --convergence artifacts/preservation_convergence_v1 \\
  --out artifacts/reproduced_preservation_comparison.json
python scripts/check_all.py
```

To repeat fitting from cached frozen features:

```bash
python scripts/run_preservation_study.py --out artifacts/preservation_repeat --jobs 3 --threads 2
```

The selected decoder files are `artifacts/shared_preservation_v1/<graph>/selected.pt`.
Use `residual_adapter.load_residuals(path)` to restore them. Each head's `predict`
method takes only observed final/raw features. Query labels are never prediction
inputs. The deployed default model is unchanged. PACKAGE_MANIFEST.json lists
every packaged source and artifact by exact size and SHA-256.
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with ZipFile(temporary, 'w', ZIP_DEFLATED, compresslevel=6) as z:
        for entry in entries:
            z.write(ROOT / entry['path'], 'SHoEbox/' + entry['path'])
        z.writestr('SHoEbox/PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        z.writestr('SHoEbox/RESTORE_PRESERVATION.md', restore)
    with ZipFile(temporary) as z:
        for entry in entries:
            raw = z.read('SHoEbox/' + entry['path'])
            if len(raw) != entry['size_bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise RuntimeError('Archive content verification failed: ' + entry['path'])
    temporary.replace(path)
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': sha(path), 'files': len(entries)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('artifacts/shared_preservation_v1'))
    parser.add_argument('--out', type=Path, default=Path('experiments/text_regions/results/shared_preservation'))
    parser.add_argument('--verification-log', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args(); torch.set_num_threads(2)
    source = ROOT / args.source; target = ROOT / args.out
    log = args.verification_log.read_text()
    if 'All five experiment suites passed.' not in log or len(re.findall(r'^OK$', log, re.M)) != 5:
        parser.error('Verification log must show five passing suites.')
    comparison = compare(source)
    convergence = ROOT / 'artifacts/preservation_convergence_v1'
    if convergence.exists():
        comparison['convergence_diagnostic'] = audit_convergence(source, convergence)
    status = json.loads((source / 'status.json').read_text())
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / 'comparison.json', comparison)
    write_json(target / 'manifest.json', {'source_directory': str(source.relative_to(ROOT)),
        'runs': [{k: r[k] for k in ('name', 'publication_policy', 'source_schedule', 'training_seed',
            'checkpoint', 'checkpoint_sha256', 'reference', 'prior_run', 'residual_run')} for r in status['runs']]})
    shutil.copyfile(args.verification_log, target / 'verification.log')
    verification = {'complete': True, 'verified_models': 12, 'new_fitted_heads': 144,
        'prior_control_heads': 72, 'automated_tests_passed': sum(map(int, re.findall(r'^Ran (\d+) tests', log, re.M))),
        'comparison_sha256': sha(target / 'comparison.json'),
        'verification_log_sha256': sha(target / 'verification.log'), 'checks': comparison['integrity'],
        'raw_artifact_delivery': 'downloadable package; no public raw-row upload'}
    write_json(target / 'verification.json', verification)
    write_bytes(target / 'README.md', b'''# Shared-answer preservation during residual training

This study adds a training-only margin penalty on correctly answered shared
examples. Two fixed penalty coefficients (1 and 10), identity/GELU controls, and
three initializations give 144 fits across twelve frozen graphs. Validation
selects both penalty and strength under the prior shared-answer accuracy guards.

`comparison.json` contains all aggregate and per-head metrics, training diagnostics,
and validation choices. `manifest.json` identifies prerequisite files. The ZIP
download supplies weights, features, and raw rows; these stay outside public git.

```bash
python -m experiments.text_regions.preservation_compare \\
  --source artifacts/shared_preservation_v1 \\
  --out artifacts/reproduced_preservation_comparison.json
```
''')
    print(json.dumps(verification, indent=2))
    if args.archive:
        print(json.dumps(archive(args.archive.resolve(), source, status), indent=2))


if __name__ == '__main__':
    main()
