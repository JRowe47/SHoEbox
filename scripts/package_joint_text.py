"""Package the exact published tree, checkpoints, raw evidence and source hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--published-commit',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT)
    commit=git('rev-parse','HEAD').decode().strip()
    assert commit==a.published_commit and not git('status','--porcelain').strip()
    root=ROOT/'artifacts/joint_text';lock=json.loads((root/'protocol_lock.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    for run in lock['runs']:
        assert json.loads((root/str(run['seed'])/'audit.json').read_text())['all_passed']
        parent=ROOT/'artifacts/multifact'/str(run['parent_seed'])/'best.pt'
        assert hashlib.sha256(parent.read_bytes()).hexdigest()==lock['parents'][str(run['parent_seed'])], 'Parent checkpoint changed.'
    verification=json.loads((root/'verification.json').read_text())
    assert verification['exit_code']==0
    for key,path in (('log_sha256',root/'regression.log'),
                     ('runner_sha256',ROOT/'scripts/check_all.py'),
                     ('new_tests_sha256',ROOT/'experiments/text_regions/test_joint_text.py')):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==verification[key], 'Regression evidence changed.'
    pilot=ROOT/'artifacts/joint_text_pilot'
    protocol=json.loads((pilot/'protocol.json').read_text())
    for name,digest in protocol['source_hashes'].items():
        assert hashlib.sha256((pilot/'source'/'experiments'/name).read_bytes()).hexdigest()==digest
    source=a.out/'SHoEbox_joint_text_source.zip'
    subprocess.run(['git','archive','--format=zip','--prefix=SHoEbox/','--output',str(source.resolve()),'HEAD'],cwd=ROOT,check=True)
    source_hashes={}
    with zipfile.ZipFile(source) as z:
        for record in git('ls-tree','-r','-z','HEAD').split(b'\0'):
            if not record:continue
            metadata,name=record.split(b'\t',1);name=name.decode();data=z.read('SHoEbox/'+name)
            assert data==git('cat-file','blob',metadata.split()[2].decode())
            source_hashes[name]=hashlib.sha256(data).hexdigest()
    files=[p for p in root.rglob('*') if p.is_file() and p.suffix not in ('.tmp','.pyc') and p.name!='PACKAGE_MANIFEST.json']
    files+=[p for p in pilot.iterdir() if p.is_file() and p.suffix not in ('.tmp','.pyc')]
    files+=[pilot/'source'/'experiments'/name for name in protocol['source_hashes']]
    files+=[ROOT/'artifacts/multifact'/str(run['parent_seed'])/'best.pt' for run in lock['runs']]
    data=ROOT/'data/tiny_shakespeare.txt'
    assert hashlib.sha256(data.read_bytes()).hexdigest()==lock['data_sha256']
    files.append(data)
    files=sorted(set(files))
    manifest=dict(publication_status='published',repository='https://github.com/JRowe47/SHoEbox',source_commit=commit,
        source_tree=git('rev-parse','HEAD^{tree}').decode().strip(),all_phase_targets_passed=summary['all_targets_passed'],
        source_files=source_hashes,source_zip_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        report_files={name:hashlib.sha256((a.out/name).read_bytes()).hexdigest()
            for name in ('SHoEbox_Joint_Text_Report.md','SHoEbox_Joint_Text_Report.pdf')},
        artifact_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    assert manifest['report_files']['SHoEbox_Joint_Text_Report.md']==source_hashes['docs/SHoEbox_Joint_Text_Report.md']
    archive=a.out/'SHoEbox_joint_text_artifacts.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,'SHoEbox/'+str(path.relative_to(ROOT)))
        z.writestr('SHoEbox/artifacts/joint_text/PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest['artifact_files'].items():assert hashlib.sha256(z.read('SHoEbox/'+name)).hexdigest()==digest
    manifest['artifact_zip_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    (a.out/'SHoEbox_joint_text_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(commit=commit,source_files=len(source_hashes),artifact_files=len(files),
        all_phase_targets_passed=summary['all_targets_passed'],archive_bytes=archive.stat().st_size)))


if __name__=='__main__':main()
