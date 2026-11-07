"""Package the unified graph milestone and its completed reproducible evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from package_payload_credit import archive

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'artifacts/graph_publication')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--commit')
    args = parser.parse_args()
    summary = json.loads((ROOT/'experiments/text_regions/results/graph_publication/summary.json').read_text())
    if not summary['paired_exposure_verified'] or not summary['checks']['complete'] or summary['checks']['returncode']:
        raise ValueError('Completed checked evidence is required.')
    for arm in ('credit','no_credit'):
        for name in ('best.pt','last.pt','evaluation.json','initial.json','protocol.json','history.jsonl'):
            if not (args.root/arm/name).is_file():
                raise ValueError(f'Missing {arm}/{name}')
    names = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    files = [(ROOT/n,'SHoEbox/'+n) for n in sorted(set(names)) if n and (ROOT/n).is_file()]
    files.append((ROOT/'data/tiny_shakespeare.txt','SHoEbox/data/tiny_shakespeare.txt'))
    files += [(p,str(Path('graph_publication')/p.relative_to(args.root)))
              for p in sorted(args.root.rglob('*')) if p.is_file() and p.suffix != '.tmp']
    args.out.mkdir(parents=True,exist_ok=True)
    commit = args.commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    metadata = dict(study=summary['study'],source_commit=commit,partial=False,
        parent_sha256=summary['parent_sha256'],
        reproduction='cd SHoEbox && python scripts/run_graph_publication.py --parent ../graph_publication/parent.pt --out artifacts/graph_publication --parallel')
    print(json.dumps(archive(args.out/'SHoEbox_Graph_Publication_Artifacts.zip',files,metadata)))


if __name__ == '__main__':
    main()
