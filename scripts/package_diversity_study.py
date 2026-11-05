"""Export audited cohort results and package source plus portable fitted artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_archive(path, files, manifest):
    entries = [{'path': n, 'sha256': sha(b), 'size_bytes': len(b)} for n, b in sorted(files.items())]
    temporary = path.with_name(path.name+'.tmp')
    with ZipFile(temporary, 'w', ZIP_DEFLATED, compresslevel=6) as z:
        for name, data in sorted(files.items()):
            z.writestr('SHoEbox/'+name, data)
        z.writestr('SHoEbox/PACKAGE_MANIFEST.json', json.dumps({**manifest, 'files': entries}, indent=2)+'\n')
    with ZipFile(temporary) as z:
        for entry in entries:
            data = z.read('SHoEbox/'+entry['path'])
            if sha(data) != entry['sha256'] or len(data) != entry['size_bytes']:
                raise RuntimeError('Archive byte verification failed: '+entry['path'])
    temporary.replace(path)
    return {'filename': path.name, 'size_bytes': path.stat().st_size,
            'sha256': sha(path.read_bytes()), 'files': len(entries)+1}


def export(comparison_path, verification_log, source, destination):
    comparison = json.loads(comparison_path.read_text())
    if (comparison['verified_models'] != 12 or comparison['fitted_heads'] != 432
            or comparison['legacy_prediction_groups_exact'] != 720
            or comparison['additional_expert_executions'] != 96768000
            or not all(comparison['integrity'].values())):
        raise ValueError('The complete audited study is required.')
    log = verification_log.read_text()
    if 'All five experiment suites passed.' not in log or len(re.findall(r'^OK$', log, re.M)) != 5:
        raise ValueError('A complete passing check_all log is required.')
    count = sum(map(int, re.findall(r'^Ran (\d+) tests? in ', log, re.M)))
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in [('comparison.json', comparison_path.read_bytes()), ('verification.log', verification_log.read_bytes())]:
        (destination/name).write_bytes(data)
    status = json.loads((source/'status.json').read_text())
    keys = ('name', 'publication_policy', 'source_schedule', 'training_seed', 'checkpoint',
            'checkpoint_sha256', 'reference', 'prior_run', 'residual_run')
    manifest = {'schema_version': 1, 'source_directory': str(source.relative_to(ROOT)),
        'runs': [{k: r[k] for k in keys} for r in status['runs']], 'settings': status['settings']}
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    verification = {'schema_version': 1, 'complete': True, 'automated_tests_passed': count,
        'verified_models': 12, 'fitted_heads': 432, 'legacy_prediction_groups_exact': 720,
        'additional_expert_executions': 96768000, 'integrity': comparison['integrity'],
        'comparison_sha256': sha(comparison_path.read_bytes()), 'verification_log_sha256': sha(verification_log.read_bytes())}
    (destination/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    (destination/'README.md').write_text('''# Decoder cohort-diversity reproduction

Twelve frozen graphs, nested 960/3840 training cohorts, 432 residual fits,
and unchanged validation/evaluation caches. `comparison.json` contains the
audited aggregate and per-head metrics. `manifest.json` records the dependencies.
Full tensors, features and synthetic prediction rows are in the downloadable
artifact package. `docs/SHoEbox_Cohort_Reproduction_Report.md` compares these
results with the surviving report of the missing experiment.

```bash
python -m experiments.text_regions.diversity_compare \\
  --source artifacts/cohort_reproduction_v1 \\
  --out artifacts/reproduced_cohort_comparison.json
```
''')
    return verification


def package(source, destination, *, publication_url=None):
    status = json.loads((source/'status.json').read_text())
    if not status['complete'] or len(status['runs']) != 12:
        raise ValueError('Cannot label an unfinished run as a complete package.')
    result_dir = ROOT/'experiments/text_regions/results/cohort_reproduction'
    comparison = json.loads((result_dir/'comparison.json').read_text())
    verification = json.loads((result_dir/'verification.json').read_text())
    if not verification['complete'] or verification['comparison_sha256'] != sha((result_dir/'comparison.json').read_bytes()):
        raise ValueError('Exported verification differs from the audited comparison.')
    for run in comparison['runs']:
        for name, expected in run['files'].items():
            if sha((source/run['name']/name).read_bytes()) != expected:
                raise ValueError('Run artifact changed after the audit.')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--'], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    tracked = subprocess.check_output(['git', 'ls-tree', '-rz', '--name-only', 'HEAD'], cwd=ROOT).decode().split('\0')
    source_files = {p: (ROOT/p).read_bytes() for p in tracked if p}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    tree = subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT).decode().strip()
    common = {'schema_version': 1, 'source_commit': commit, 'source_tree': tree, 'complete': True,
        'publication': {'status': 'published' if publication_url else 'unpublished', 'url': publication_url},
        'reproduced_missing_experiment': True, 'original_missing_commit': 'c4be984e247f738a63433c7a84d4aa7a2c5918ef',
        'fitted_heads': 432, 'verified_models': 12}
    artifacts = {}
    # Include all dependencies of the preserved audits, plus every completed
    # cohort run and per-fit resume checkpoint. No environment or git metadata.
    for folder in ('emission_adapter_v1', 'residual_adapter_v1', 'shared_preservation_v1',
                   'preservation_convergence_v1'):
        for p in (ROOT/'artifacts'/folder).rglob('*'):
            if p.is_file() and not p.name.endswith('.tmp'):
                artifacts[str(p.relative_to(ROOT))] = p.read_bytes()
    manifest = json.loads((ROOT/'experiments/text_regions/results/shared_preservation/manifest.json').read_text())
    for run in manifest['runs']:
        for key in ('checkpoint', 'reference'):
            p = ROOT/run[key]; artifacts[run[key]] = p.read_bytes()
    for p in source.rglob('*'):
        if p.is_file() and not p.name.endswith('.tmp'):
            artifacts[str(p.relative_to(ROOT))] = p.read_bytes()
    restore = '''# SHoEbox reproduced cohort experiment

This package includes the new source, all twelve frozen graphs, fixed baseline
heads, original feature/validation caches, fresh expanded training features,
all 432 fitted corrections, selection records, predictions and audit results.

From the extracted SHoEbox directory:

```bash
python -m pip install -r requirements.txt
python scripts/check_all.py
python -m experiments.text_regions.diversity_compare \\
  --source artifacts/cohort_reproduction_v1 \\
  --out artifacts/reproduced_cohort_comparison.json
```

To fit a fresh repetition, use a new output directory:

```bash
python scripts/run_diversity_study.py \\
  --out artifacts/cohort_repeat --jobs 4 --threads 2
```

Add --resume with the same output directory to continue after an interruption.
Every finished fit is saved atomically. Source, input, settings and frozen-base
checks prevent resuming a different experiment. PACKAGE_MANIFEST.json lists
every byte-verified entry, the source commit and its publication status.
'''.encode()
    destination.mkdir(parents=True, exist_ok=True)
    source_files['RESTORE_COHORT.md'] = restore
    results = [write_archive(destination/'SHoEbox_cohort_reproduction_source.zip', source_files,
                             {**common, 'package_kind': 'source'}),
               write_archive(destination/'SHoEbox_cohort_reproduction_artifacts.zip',
                             {**source_files, **artifacts}, {**common, 'package_kind': 'source_and_artifacts'})]
    (destination/'package_checksums.json').write_text(json.dumps(results, indent=2)+'\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('artifacts/cohort_reproduction_v1'))
    sub = parser.add_subparsers(dest='action', required=True)
    e = sub.add_parser('export'); e.add_argument('--comparison', type=Path, required=True)
    e.add_argument('--verification-log', type=Path, required=True)
    e.add_argument('--out', type=Path, default=Path('experiments/text_regions/results/cohort_reproduction'))
    p = sub.add_parser('package'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--publication-url', help='Verified public commit URL; omit for an unpublished package.')
    args = parser.parse_args(); source = ROOT/args.source
    result = export(args.comparison, args.verification_log, source, ROOT/args.out) if args.action == 'export' else package(source, args.out, publication_url=args.publication_url)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
