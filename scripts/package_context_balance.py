"""Package completed contextual-learning evidence and its exact source."""
import argparse
import json
from pathlib import Path
import subprocess
from package_payload_credit import archive
from summarize_context_balance import summarize

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'artifacts/context_balance')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit')
    a = p.parse_args()
    summary = summarize(a.root)
    names = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'], cwd=ROOT).decode().split('\0')
    files = [(ROOT/n, 'SHoEbox/'+n) for n in sorted(set(names)) if n and (ROOT/n).is_file()]
    files.append((ROOT/'data/tiny_shakespeare.txt', 'SHoEbox/data/tiny_shakespeare.txt'))
    files += [(path, str(Path('SHoEbox/artifacts/context_balance')/path.relative_to(a.root)))
              for path in sorted(a.root.rglob('*')) if path.is_file() and path.suffix != '.tmp']
    a.out.mkdir(parents=True, exist_ok=True)
    commit = a.commit or subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    metadata = dict(study=summary['study'], source_commit=commit, partial=False,
        parent_sha256=summary['arms']['group_clip']['protocol']['protocol']['parent_sha256'],
        resume='cd SHoEbox && python scripts/run_context_balance.py --parent artifacts/context_balance/parent.pt --parallel',
        fresh='cd SHoEbox && python scripts/run_context_balance.py --parent artifacts/context_balance/parent.pt --out artifacts/reproduction --parallel')
    print(json.dumps(archive(a.out/'SHoEbox_Context_Balance_Artifacts.zip', files, metadata)))


if __name__ == '__main__':
    main()
