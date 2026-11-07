"""Package complete source, paired checkpoints, audits and file hashes."""
import argparse
import json
from pathlib import Path
import subprocess

from package_payload_credit import archive
from summarize_state_commit import summarize

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'artifacts/state_commit')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    summary = summarize(args.root, [41, 43])
    if any(row['updates'] != 600 for row in summary['rows']):
        raise ValueError('The complete study requires all eight 600-update arms.')
    for row in summary['rows']:
        folder = args.root/str(row['seed'])/row['arm']
        for name in ('best.pt', 'last.pt', 'acquire.pt', 'bridge.pt'):
            if not (folder/name).is_file():
                raise ValueError(f'Missing checkpoint: {folder/name}')
    checks = json.loads((args.root/'validation/all_checks_result.json').read_text())
    if checks['returncode']:
        raise ValueError('The repository check suite must pass before packaging.')
    tracked = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
                                      cwd=ROOT).decode().split('\0')
    source = [(ROOT/name, 'SHoEbox/'+name) for name in sorted(set(tracked))
              if name and (ROOT/name).is_file()]
    evidence = [(path, str(Path('state_commit')/path.relative_to(args.root)))
                for path in sorted(args.root.rglob('*')) if path.is_file() and path.suffix != '.tmp']
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = dict(source_base='150c8a0', partial=False,
        study='Producer state commits, 2x2 state/read modes, seeds 41/43, 600 updates per arm',
        source_sha256=summary['source_sha256'],
        baseline_provenance='state_commit/baseline_provenance.json',
        reproduction='cd SHoEbox && python scripts/run_state_commit.py --out artifacts/state_commit --workers 4')
    print(json.dumps(archive(args.out/'SHoEbox_State_Commit_Artifacts.zip', source+evidence, metadata)))


if __name__ == '__main__':
    main()
