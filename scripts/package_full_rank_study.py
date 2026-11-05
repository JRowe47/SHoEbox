"""Export verified aggregate results and package the full-rank decoder phase."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from package_diversity_study import sha, write_archive

ROOT = Path(__file__).resolve().parents[1]


def export(comparison, benchmark, tests, destination):
    result = json.loads(comparison.read_text()); timing = json.loads(benchmark.read_text())
    if (result['verified_models'] != 12 or result['fitted_heads'] != 72 or
            result['inherited_fitted_heads_verified'] != 432 or not all(result['integrity'].values())):
        raise ValueError('Complete audited results are required.')
    log = tests.read_text()
    if 'All five experiment suites passed.' not in log or len(re.findall(r'^OK$', log, re.M)) != 5:
        raise ValueError('Complete passing verification log required.')
    if len(timing['measurements']) != 144 or timing['settings']['graphs'] != 12:
        raise ValueError('Complete decoder timing results required.')
    for r in result['runs']:
        for n, expected in r['files'].items():
            if sha((ROOT/'artifacts/full_rank_v1'/r['name']/n).read_bytes()) != expected:
                raise ValueError('Audited artifact changed.')
        if timing['inputs'][r['name']]['fitted_sha256'] != r['files']['fitted.pt']:
            raise ValueError('Timed and audited heads differ.')
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in [('comparison.json', comparison), ('benchmark.json', benchmark), ('verification.log', tests)]:
        (destination/name).write_bytes(source.read_bytes())
    published_heads = ('base', '256/linear/preserved/selected', '256/linear/penalty_0/selected',
        '256/identity/selected/selected', '256/gelu/selected/selected')
    summary = {'verified_models': 12, 'new_fitted_heads': 72, 'inherited_fitted_heads_verified': 432,
        'compiled_heads': 204, 'compiled_prediction_cases_exact': result['compiled_prediction_cases_exact'],
        'additional_expert_executions': 0,
        'accuracy_means': {g: {h: {k: v['mean'] for k, v in metrics[h].items()}
            for h in published_heads} for g, metrics in result['groups'].items()},
        'decoder_timing_microseconds': timing['groups'], 'timing_scope': timing['scope'],
        'timing_settings': timing['settings'], 'timing_runtime': timing['runtime']}
    (destination/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    verification = {'schema_version': 1, 'complete': True,
        'automated_tests_passed': sum(map(int, re.findall(r'^Ran (\d+) tests? in ', log, re.M))),
        'verified_models': 12, 'fitted_heads': 72, 'inherited_fitted_heads_verified': 432,
        'prediction_groups_exact': result['prediction_groups_exact'],
        'inherited_prediction_groups_exact': result['inherited_prediction_groups_exact'],
        'compiled_prediction_cases_exact': result['compiled_prediction_cases_exact'],
        'comparison_sha256': sha(comparison.read_bytes()), 'benchmark_sha256': sha(benchmark.read_bytes()),
        'verification_log_sha256': sha(tests.read_bytes())}
    (destination/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    (destination/'README.md').write_text('''# Full-rank decoder phase

Seventy-two deterministic full-matrix byte decoder fits, across twelve frozen
graphs and the same nested 960/3,840 training cohorts. The comparison re-audits
all 432 inherited identity/GELU corrections. Aggregate accuracy results and CPU timings
are summarized in summary.json. Detailed per-head comparisons, timing repetitions, tensors,
caches and raw synthetic predictions are in the downloadable artifact package.
See `docs/SHoEbox_Full_Rank_Readout_Report.md`.

```bash
python -m experiments.text_regions.full_rank_compare \\
  --out artifacts/reproduced_full_rank_comparison.json
```
''')
    return verification


def package(destination, publication_url=None):
    results = ROOT/'experiments/text_regions/results/full_rank'
    comparison = json.loads((results/'comparison.json').read_text())
    verification = json.loads((results/'verification.json').read_text())
    if not verification['complete'] or verification['comparison_sha256'] != sha((results/'comparison.json').read_bytes()):
        raise ValueError('Audited export is missing or changed.')
    for row in comparison['runs']:
        for name, expected in row['files'].items():
            if sha((ROOT/'artifacts/full_rank_v1'/row['name']/name).read_bytes()) != expected:
                raise ValueError('Artifact changed since audit.')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    names = subprocess.check_output(['git', 'ls-tree', '-rz', '--name-only', 'HEAD'], cwd=ROOT).decode().split('\0')
    source = {n: (ROOT/n).read_bytes() for n in names if n}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    tree = subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT).decode().strip()
    artifacts = {}
    for folder in ('emission_adapter_v1', 'residual_adapter_v1', 'cohort_reproduction_v1', 'full_rank_v1'):
        for path in (ROOT/'artifacts'/folder).rglob('*'):
            if path.is_file() and not path.name.endswith('.tmp'):
                artifacts[str(path.relative_to(ROOT))] = path.read_bytes()
    for row in comparison['runs']:
        for key in ('checkpoint', 'reference'):
            artifacts[row[key]] = (ROOT/row[key]).read_bytes()
    for name in ('comparison.json', 'benchmark.json', 'verification.log'):
        path = results/name
        artifacts[str(path.relative_to(ROOT))] = path.read_bytes()
    source['RESTORE_FULL_RANK.md'] = b'''# SHoEbox full-rank decoder phase

The source archive contains code, reports and aggregate results. The artifact
archive also includes twelve frozen graphs, baseline heads, all 432 inherited
corrections, all 72 new refits, 204 compiled inference heads, fixed feature and
validation caches, selections and synthetic predictions. No original workspace
path is required.

From the extracted SHoEbox directory:

```bash
python -m pip install -r requirements.txt
python scripts/check_all.py
python -m experiments.text_regions.full_rank_compare \\
  --out artifacts/reproduced_full_rank_comparison.json
```

For a fresh fit and compiled export:

```bash
python scripts/run_full_rank_study.py --out artifacts/full_rank_repeat
python -m experiments.text_regions.full_rank_compare \\
  --source artifacts/full_rank_repeat --export-compiled \\
  --out artifacts/full_rank_repeat_comparison.json
```

Add --resume to the fitting command after an interruption. Complete fits are
saved atomically; changed code, inputs, settings or baselines reject resumption.
CPU timing can be repeated with scripts/benchmark_full_rank_readout.py.

PACKAGE_MANIFEST.json records every file's SHA-256, source commit and publication
status. The default graph and decoder are unchanged. The compiled readout is an
optional feature-only module; deployment usage is in the experiment README.
'''
    common = {'schema_version': 1, 'complete': True, 'source_commit': commit, 'source_tree': tree,
        'publication': {'status': 'published' if publication_url else 'unpublished', 'url': publication_url},
        'verified_models': 12, 'new_full_rank_fits': 72, 'inherited_correction_fits': 432,
        'compiled_heads': 204, 'additional_expert_executions': 0}
    destination.mkdir(parents=True, exist_ok=True)
    packages = [write_archive(destination/'SHoEbox_full_rank_source.zip', source, {**common, 'package_kind': 'source'}),
        write_archive(destination/'SHoEbox_full_rank_artifacts.zip', {**source, **artifacts},
            {**common, 'package_kind': 'source_and_artifacts'})]
    (destination/'package_checksums.json').write_text(json.dumps(packages, indent=2)+'\n')
    return packages


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='action', required=True)
    e = sub.add_parser('export'); e.add_argument('--comparison', type=Path, required=True)
    e.add_argument('--benchmark', type=Path, required=True); e.add_argument('--tests', type=Path, required=True)
    e.add_argument('--out', type=Path, default=Path('experiments/text_regions/results/full_rank'))
    a = sub.add_parser('package'); a.add_argument('--out', type=Path, required=True); a.add_argument('--publication-url')
    args = p.parse_args()
    value = export(args.comparison, args.benchmark, args.tests, ROOT/args.out) if args.action == 'export' else package(args.out, args.publication_url)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
