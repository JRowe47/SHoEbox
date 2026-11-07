"""Package resumable memory experiments and verify every archived byte."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--revision',required=True)
    a=p.parse_args()
    experiment=ROOT/'artifacts/memory_acquisition'
    if not (experiment/'summary.json').exists():
        raise ValueError('Run the completed-study audit first.')
    summary=json.loads((experiment/'summary.json').read_text())
    for name in ('lr003','retention'):
        arm=summary['arms'][name]
        if arm['completed_updates']!=arm['planned_maximum'] or not arm.get('selected_test',{}).get('complete'):
            raise ValueError(f'Incomplete training or selected evaluation: {name}')
        if not arm.get('final_test',{}).get('complete'):
            raise ValueError(f'Incomplete fixed-budget evaluation: {name}')
    paths=[ROOT/'data/tiny_shakespeare.txt',ROOT/'artifacts/connected_recall/connected/best.pt']
    paths.extend(path for path in sorted(experiment.rglob('*'))
        if path.is_file() and 'resume_check' not in path.parts and path.suffix in ('.pt','.json','.jsonl','.txt'))
    manifest=dict(revision=a.revision,source='https://github.com/JRowe47/SHoEbox',files=[])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in paths:
            data=path.read_bytes();name='SHoEbox/'+str(path.relative_to(ROOT))
            archive.writestr(name,data)
            manifest['files'].append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        archive.writestr('SHoEbox/artifacts/memory_acquisition/MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        archive.writestr('SHoEbox/artifacts/memory_acquisition/REPRODUCE.txt',
            'Use the SHoEbox source at revision '+a.revision+'.\n'
            'Extract this archive alongside the cloned SHoEbox directory.\n'
            'The archive supplies the corpus, starting checkpoint, selected/final checkpoints,\n'
            'source protocols, training histories, causal evaluations and audits.\n'
            'Run commands from SHoEbox; see docs/memory_acquisition.md.\n'
            'The completed evaluation files are identity-checked caches; use a new --out\n'
            'filename to rerun an evaluation, or rerun with the same arguments to recover it.\n'
            'GPU/runtime changes and broader integrated acceptance were outside this phase.\n')
    with zipfile.ZipFile(a.out) as archive:
        assert archive.testzip() is None
        for row in manifest['files']:
            data=archive.read(row['path'])
            assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    print(json.dumps(dict(path=str(a.out),files=len(manifest['files']),bytes=a.out.stat().st_size,
                          all_checksums_verified=True)))


if __name__=='__main__':main()
