"""Package exact source, corpus, checkpoints and completed state-credit evidence."""
import argparse
import json
from pathlib import Path
import subprocess
from package_payload_credit import archive
from summarize_state_credit import summarize

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'artifacts/state_credit')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    a=p.parse_args();summary=summarize(a.root)
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    files=[(ROOT/n,'SHoEbox/'+n) for n in sorted(set(names)) if n and (ROOT/n).is_file()]
    files.append((ROOT/'data/tiny_shakespeare.txt','SHoEbox/data/tiny_shakespeare.txt'))
    files.extend((path,str(Path('SHoEbox/artifacts/state_credit')/path.relative_to(a.root)))
        for path in sorted(a.root.rglob('*')) if path.is_file() and path.suffix!='.tmp')
    a.out.mkdir(parents=True,exist_ok=True)
    metadata=dict(study=summary['study'],source_commit=a.commit,partial=False,
        parent_sha256=summary['environment']['parent_sha256'],
        fresh='cd SHoEbox && python scripts/run_state_credit.py --parent artifacts/state_credit/parent.pt --out artifacts/reproduction',
        evaluate='cd SHoEbox && python scripts/evaluate_state_credit.py',
        checks='cd SHoEbox && python scripts/check_all.py')
    print(json.dumps(archive(a.out/'SHoEbox_State_Credit_Artifacts.zip',files,metadata)))


if __name__=='__main__':
    main()
