"""Package a verified published tree and the complete paired continuation study."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--published-commit',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT)
    commit=git('rev-parse','HEAD').decode().strip()
    assert commit==a.published_commit and not git('status','--porcelain').strip()
    root=ROOT/'artifacts/joint_stability';study=root/'study'
    lock=json.loads((study/'protocol_lock.json').read_text())
    summary=json.loads((study/'summary.json').read_text())
    verification=json.loads((study/'verification.json').read_text())
    assert verification['exit_code']==0
    assert verification['log_sha256']==hashlib.sha256((root/'regression_tests.log').read_bytes()).hexdigest()
    assert verification['runner_sha256']==hashlib.sha256((ROOT/'scripts/check_all.py').read_bytes()).hexdigest()
    assert verification['new_tests_sha256']==hashlib.sha256((ROOT/'experiments/text_regions/test_gradient_balance.py').read_bytes()).hexdigest()
    for run in lock['runs']:
        parent=root/'parents'/str(run['parent_seed'])/'last.pt'
        assert hashlib.sha256(parent.read_bytes()).hexdigest()==lock['parents'][str(run['parent_seed'])]
        for mode in lock['modes']:
            assert json.loads((study/str(run['seed'])/mode/'audit.json').read_text())['all_passed']
    source=a.out/'SHoEbox_joint_stability_source.zip'
    subprocess.run(['git','archive','--format=zip','--prefix=SHoEbox/','--output',str(source.resolve()),'HEAD'],cwd=ROOT,check=True)
    source_hashes={}
    with zipfile.ZipFile(source) as archive:
        for record in git('ls-tree','-r','-z','HEAD').split(b'\0'):
            if not record:continue
            metadata,name=record.split(b'\t',1);name=name.decode();data=archive.read('SHoEbox/'+name)
            assert data==git('cat-file','blob',metadata.split()[2].decode())
            source_hashes[name]=hashlib.sha256(data).hexdigest()
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix not in ('.tmp','.pyc')
                 and p.name not in ('PACKAGE_MANIFEST.json','evaluation.partial.json'))
    data=ROOT/'data/tiny_shakespeare.txt'
    assert hashlib.sha256(data.read_bytes()).hexdigest()==lock['data_sha256']
    files.append(data)
    reports={name:hashlib.sha256((a.out/name).read_bytes()).hexdigest() for name in
        ('SHoEbox_Joint_Stability_Report.md','SHoEbox_Joint_Stability_Report.pdf')}
    assert reports['SHoEbox_Joint_Stability_Report.md']==source_hashes['docs/SHoEbox_Joint_Stability_Report.md']
    manifest=dict(publication_status='published',repository='https://github.com/JRowe47/SHoEbox',
        source_commit=commit,source_tree=git('rev-parse','HEAD^{tree}').decode().strip(),
        all_clipped_targets_passed=summary['all_clipped_targets_passed'],
        source_files=source_hashes,source_zip_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        report_files=reports,artifact_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    destination=a.out/'SHoEbox_joint_stability_artifacts.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,'SHoEbox/'+str(path.relative_to(ROOT)))
        archive.writestr('SHoEbox/artifacts/joint_stability/PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        for name,digest in manifest['artifact_files'].items():
            assert hashlib.sha256(archive.read('SHoEbox/'+name)).hexdigest()==digest
    manifest['artifact_zip_sha256']=hashlib.sha256(destination.read_bytes()).hexdigest()
    (a.out/'SHoEbox_joint_stability_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(commit=commit,source_files=len(source_hashes),artifact_files=len(files),
        artifact_bytes=destination.stat().st_size,all_clipped_targets_passed=summary['all_clipped_targets_passed'])))


if __name__=='__main__':main()
