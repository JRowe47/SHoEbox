"""Package committed source and the verified, surviving preservation artifacts.

The later cohort-diversity report is included; its missing code and tensors are
explicitly excluded from the claim of reproducibility. Uses only the stdlib.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA256 = '59eb82556c59f2469d30aee21b5920f97ecfd1d5e43ec93a361aae812576e78b'
BASELINE_COMMIT = 'faed544b37000fcdb11b5584396c7856b85a4be5'
MISSING_COMMIT = 'c4be984e247f738a63433c7a84d4aa7a2c5918ef'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name:
        raise ValueError('Unsafe archive path: ' + name)
    return name


def baseline_files(archive):
    if sha(archive.read_bytes()) != BASELINE_SHA256:
        raise ValueError('Baseline ZIP does not match the retained published package.')
    with ZipFile(archive) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError('Duplicate baseline entries.')
        manifest = json.loads(z.read('SHoEbox/PACKAGE_MANIFEST.json'))
        if manifest['source_commit'] != BASELINE_COMMIT:
            raise ValueError('Unexpected baseline source commit.')
        artifacts = {}
        for item in manifest['files']:
            name = safe_path(item['path'])
            data = z.read('SHoEbox/' + name)
            if len(data) != item['size_bytes'] or sha(data) != item['sha256']:
                raise ValueError('Baseline manifest mismatch: ' + name)
            if name.startswith('artifacts/'):
                artifacts[name] = data
        artifacts['recovery/BASELINE_PACKAGE_MANIFEST.json'] = (
            json.dumps(manifest, indent=2) + '\n').encode()
    return artifacts


def source_files():
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--'], cwd=ROOT,
                   check=True, stdout=subprocess.DEVNULL)
    names = subprocess.check_output(['git', 'ls-tree', '-rz', '--name-only', 'HEAD'],
                                    cwd=ROOT).decode().split('\0')
    return {safe_path(name): (ROOT / name).read_bytes() for name in names if name}


def restore_note(with_artifacts):
    availability = ('This archive includes the verified preservation tensors, caches, '
                    'selection locks and synthetic prediction rows.' if with_artifacts else
                    'This source archive contains no tensors. Use the recovered-artifacts '
                    'archive for the preservation audit below.')
    return f'''# SHoEbox recovery package

{availability}

The cohort-diversity study is REPORT ONLY. Its 432 fitted heads, expanded
feature caches, new implementation and full comparison JSON were not recovered.
The earlier preservation study is reproducible from the recovered-artifacts ZIP.
See docs/SHoEbox_Recovery_Report.md for the exact recovery boundary.

Run from this extracted SHoEbox directory:

```bash
python -m pip install -r requirements.txt
python scripts/check_all.py
python experiments/hat_minkron/example_live.py
python -m experiments.text_regions.preservation_compare \\
  --source artifacts/shared_preservation_v1 \\
  --convergence artifacts/preservation_convergence_v1 \\
  --out artifacts/reproduced_preservation_comparison.json
```

PACKAGE_MANIFEST.json records the published source commit and every packaged
file's size and SHA-256. The baseline artifact bytes are unchanged. Audit code
allows 1e-12 absolute/relative tolerance for reconstructed float64 normalization
and named loss fields; stored tensors, decisions and predictions stay exact.
'''.encode()


def archive(destination, files, commit, with_artifacts):
    if destination.exists():
        raise ValueError('Refusing to replace an existing package: ' + str(destination))
    files = {**files, 'RESTORE_RECOVERY.md': restore_note(with_artifacts)}
    entries = [{'path': name, 'size_bytes': len(data), 'sha256': sha(data)}
               for name, data in sorted(files.items())]
    manifest = {
        'schema_version': 1, 'source_commit': commit,
        'package_kind': 'recovered_artifacts' if with_artifacts else 'source',
        'baseline_commit': BASELINE_COMMIT,
        'baseline_archive_sha256': BASELINE_SHA256 if with_artifacts else None,
        'cohort_diversity': {'status': 'report_only',
            'missing_source_commit': MISSING_COMMIT,
            'fitted_heads_recovered': 0, 'expanded_feature_caches_recovered': 0},
        'files': entries,
    }
    temporary = destination.with_suffix('.tmp')
    try:
        with ZipFile(temporary, 'w', ZIP_DEFLATED, compresslevel=6) as z:
            for name, data in sorted(files.items()):
                z.writestr('SHoEbox/' + name, data)
            z.writestr('SHoEbox/PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        with ZipFile(temporary) as z:
            expected = {'SHoEbox/' + item['path'] for item in entries}
            expected.add('SHoEbox/PACKAGE_MANIFEST.json')
            if set(z.namelist()) != expected or len(z.namelist()) != len(expected):
                raise RuntimeError('Archive file inventory differs.')
            for item in entries:
                data = z.read('SHoEbox/' + item['path'])
                if len(data) != item['size_bytes'] or sha(data) != item['sha256']:
                    raise RuntimeError('Archive content differs: ' + item['path'])
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {'filename': destination.name, 'size_bytes': destination.stat().st_size,
            'sha256': sha(destination.read_bytes()), 'files': len(entries) + 1,
            'source_commit': commit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    artifacts = baseline_files(args.baseline_archive)
    source = source_files()
    if set(source) & set(artifacts):
        raise ValueError('Artifact/source inventory overlap.')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    args.out.mkdir(parents=True, exist_ok=True)
    results = [archive(args.out / 'SHoEbox_recovered_source.zip', source, commit, False),
               archive(args.out / 'SHoEbox_recovered_artifacts.zip',
                       {**source, **artifacts}, commit, True)]
    (args.out / 'package_checksums.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
