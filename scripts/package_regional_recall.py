"""Package source, corpus, checkpoints and completed evidence with hashes."""
import argparse
import json
from pathlib import Path
import subprocess

from package_payload_credit import archive

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'artifacts/regional_recall')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', help='Published source commit when distinct from the local checkout.')
    args = parser.parse_args()
    result = json.loads((args.root/'checks.json').read_text())
    if not result['complete'] or result['returncode']:
        raise ValueError('All checks must pass before packaging.')
    summary = json.loads((ROOT/'experiments/text_regions/results/regional_recall/summary.json').read_text())
    for layout in ('distinct', 'pooled'):
        for name in ('best.pt', 'last.pt', 'evaluation.json', 'initial.json', 'protocol.json'):
            if not (args.root/layout/name).is_file():
                raise ValueError(f'Missing evidence: {layout}/{name}')
        if summary['arms'][layout]['evaluation']['last']['step'] != summary['arms'][layout]['protocol']['protocol']['steps']:
            raise ValueError('The final budget is incomplete.')
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
                                    cwd=ROOT).decode().split('\0')
    files = [(ROOT/name, 'SHoEbox/'+name) for name in sorted(set(names)) if name and (ROOT/name).is_file()]
    files.append((ROOT/'data/tiny_shakespeare.txt', 'SHoEbox/data/tiny_shakespeare.txt'))
    files += [(path, str(Path('regional_recall')/path.relative_to(args.root)))
              for path in sorted(args.root.rglob('*')) if path.is_file() and path.suffix != '.tmp']
    args.out.mkdir(parents=True, exist_ok=True)
    commit = args.commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    metadata = dict(study='Regional recall and mixed natural-language/memory continuation',
        source_commit=commit, partial=False, parent_sha256=summary['parent_sha256'],
        reproduction='cd SHoEbox && python scripts/run_regional_recall.py --parent ../regional_recall/parent.pt --out artifacts/regional_recall')
    print(json.dumps(archive(args.out/'SHoEbox_Regional_Recall_Artifacts.zip', files, metadata)))


if __name__ == '__main__':
    main()
