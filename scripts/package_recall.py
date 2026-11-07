"""Package the complete recall source and resumable evidence with file hashes."""
import argparse
import json
from pathlib import Path
import subprocess

from package_payload_credit import archive
from summarize_recall import summarize

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'artifacts/recall')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    summary = summarize(args.root)
    checks = json.loads((args.root/'validation/all_checks_result.json').read_text())
    if checks['returncode'] or not checks['complete']:
        raise ValueError('All repository checks must pass before packaging.')
    for layout in ('distinct', 'pooled'):
        for name in ('best.pt', 'last.pt'):
            if not (args.root/'study'/layout/name).is_file():
                raise ValueError(f'Missing resumable checkpoint: {layout}/{name}')
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=ROOT).decode().split('\0')
    files = [(ROOT/name, 'SHoEbox/'+name) for name in sorted(set(names)) if name and (ROOT/name).is_file()]
    files += [(path, str(Path('recall')/path.relative_to(args.root))) for path in sorted(args.root.rglob('*'))
              if path.is_file() and path.suffix != '.tmp']
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = dict(study='Learnable recall workspace: matched two-slot continuation and seven-slot integration.',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        parent_sha256=summary['parent_sha256'], partial=False,
        reproduction='cd SHoEbox && python scripts/run_recall.py --parent ../recall/study/parent.pt --out artifacts/recall/study')
    print(json.dumps(archive(args.out/'SHoEbox_Recall_Artifacts.zip', files, metadata)))


if __name__ == '__main__':
    main()
