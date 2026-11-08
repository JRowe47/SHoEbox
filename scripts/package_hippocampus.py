"""Package a committed source tree and audited fixed-budget checkpoint evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def make_archive(destination, files, prefix, extras):
    checksums = {}
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(files)):
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, prefix + relative)
            checksums[relative] = digest(path)
        for name, content in extras.items():
            payload = content.encode()
            archive.writestr(prefix + name, payload)
            checksums[name] = hashlib.sha256(payload).hexdigest()
        archive.writestr(prefix + 'SHA256SUMS', ''.join(f'{sha}  {name}\n' for name, sha in sorted(checksums.items())))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC validation failed.')
        for name, sha in checksums.items():
            if hashlib.sha256(archive.read(prefix + name)).hexdigest() != sha:
                raise ValueError('Archive checksum mismatch: ' + name)
    return dict(file=destination.name, bytes=destination.stat().st_size, sha256=digest(destination), entries=len(checksums))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = ROOT / 'artifacts/hippocampus'
    summary_path = ROOT / 'experiments/text_regions/results/hippocampus/summary.json'
    summary = json.loads(summary_path.read_text())
    if not summary['source_trajectories_identical']:
        raise ValueError('Source trajectories must be audited before packaging.')
    progress = json.loads((root / 'evaluation/progress.json').read_text())
    if progress['failed'] or progress['active'] or progress['queued'] or len(progress['completed']) != 21:
        raise ValueError('All endpoint evaluations and controls must complete.')
    for arm in ('A', 'B', 'C'):
        checkpoint = root / arm / 'last.pt'
        if digest(checkpoint) != summary['arms'][arm]['checkpoint_sha256']:
            raise ValueError('Checkpoint identity changed: ' + arm)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True):
        raise ValueError('Commit tracked changes before packaging source.')
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    source = [ROOT / name for name in names if name]
    files = [summary_path, ROOT / 'data/tiny_shakespeare.txt', root / 'parent.pt']
    for arm in ('A', 'B', 'C'):
        files.extend(root / arm / name for name in ('last.pt', 'best.pt', 'history.jsonl', 'initial.json', 'protocol.json'))
    files.extend(root / 'A' / name for name in ('control_reuse.json', 'reproduction_history.jsonl'))
    files.extend(root / name for name in ('cohorts.json', 'scaling.json', 'execution_policy.json',
        'parallel_validation_proof.json', 'recompute_proof.json', 'release_tests.log'))
    files.extend(sorted((root / 'evaluation').glob('*.json')))
    files.extend(sorted((root / 'evaluation').glob('*.jsonl')))
    readme = f'''# SHoEbox hippocampus checkpoints

Source commit: {commit}

Unzip the source archive, then merge the `artifacts/`, `data/` and `experiments/`
directories from this archive into its repository root. Use the repository's
Python requirements and commands in `docs/hippocampus.md`. SHA256SUMS records
every delivered file's SHA-256 digest. The parent is the retained strong
memory-selectivity model, separate from paired control A.

All three arms finish at 64 updates. A's endpoint is reused after exact
30-update reproduction and complete protocol/source-ledger verification;
`A/control_reuse.json` records this. B uses local cues/anchors; C adds shared
associative routing. `best.pt` is selected by validation paired NLL;
`last.pt` is the primary fixed-budget comparison. The summary records the
promotion decision; the feature remains default off.

To evaluate bundled endpoints, run:

```bash
python scripts/finish_hippocampus.py --workers 3
python scripts/summarize_hippocampus.py
```

The manager validates cache identity, then reuses completed evaluations. Delete
only the intended evaluation output when deliberately rerunning it. Training
checkpoints include optimizer, RNG, source cursors and carried language state.
The complete 64-update training budget is unchanged by recomputation.

C ran updates 1-14 directly, 15-18 with 16-byte segmented recomputation, and
19-64 with adaptive recomputation above 768 batch-byte positions. Exact phase
commands are in `docs/hippocampus.md`; `execution_policy.json` records the
schedule. Parallel validation has the same output as serial validation.

Cross-region controlled wiring fixtures and 64-512 mock regional owners are
separate from the learned three-region task. Physical GPU/distributed execution
is the later H5 milestone. No abandoned timing runs are included.
'''
    provenance = json.dumps(dict(source_commit=commit, summary_sha256=digest(summary_path)), indent=2) + '\n'
    results = [make_archive(args.out / 'SHoEbox_Hippocampus_Source.zip', source, 'SHoEbox/',
                           {'HIPPOCAMPUS_BUILD.json': provenance}),
               make_archive(args.out / 'SHoEbox_Hippocampus_Checkpoints.zip', files, 'SHoEbox_hippocampus_checkpoints/',
                           {'README.md': readme, 'BUILD.json': provenance})]
    (args.out / 'archive_manifest.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results))


if __name__ == '__main__':
    main()
