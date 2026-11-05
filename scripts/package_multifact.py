"""Package the exact published Git tree and fully hashed multi-fact artifacts."""
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
    def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
    commit=git('rev-parse','HEAD').decode().strip()
    assert commit==a.published_commit and not git('status','--porcelain').strip()
    root=ROOT/'artifacts/multifact'
    assert json.loads((root/'run_status.json').read_text())['state']=='complete'
    assert json.loads((root/'summary.json').read_text())['all_targets_passed']
    for pilot in ('multifact_pilot','multifact_field_gate_pilot'):
        folder=ROOT/'artifacts'/pilot
        protocol=json.loads((folder/'protocol.json').read_text())
        for name,digest in protocol['source_hashes'].items():
            assert hashlib.sha256((folder/'source'/'experiments'/name).read_bytes()).hexdigest()==digest
    source=a.out/'SHoEbox_multifact_source.zip'
    subprocess.run(['git','archive','--format=zip','--prefix=SHoEbox/','--output',str(source.resolve()),'HEAD'],cwd=ROOT,check=True)
    hashes={}
    with zipfile.ZipFile(source) as z:
        for record in git('ls-tree','-r','-z','HEAD').split(b'\0'):
            if not record:continue
            metadata,name=record.split(b'\t',1);name=name.decode();data=z.read('SHoEbox/'+name)
            assert data==git('cat-file','blob',metadata.split()[2].decode())
            hashes[name]=hashlib.sha256(data).hexdigest()
    folders=('multifact','multifact_pilot','multifact_field_gate_pilot')
    files=sorted(p for folder in folders for p in (ROOT/'artifacts'/folder).rglob('*')
                 if p.is_file() and p.suffix not in ('.tmp','.pyc') and p.name!='PACKAGE_MANIFEST.json')
    manifest=dict(publication_status='published',repository='https://github.com/JRowe47/SHoEbox',source_commit=commit,
        source_tree=git('rev-parse','HEAD^{tree}').decode().strip(),source_files=hashes,
        source_zip_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        report_files={name:hashlib.sha256((a.out/name).read_bytes()).hexdigest()
                      for name in ('SHoEbox_Multifact_Report.md','SHoEbox_Multifact_Report.pdf')},
        artifact_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    assert manifest['report_files']['SHoEbox_Multifact_Report.md']==hashes['docs/SHoEbox_Multifact_Report.md']
    archive=a.out/'SHoEbox_multifact_artifacts.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,'SHoEbox/'+str(path.relative_to(ROOT)))
        z.writestr('SHoEbox/artifacts/multifact/PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest['artifact_files'].items():assert hashlib.sha256(z.read('SHoEbox/'+name)).hexdigest()==digest
    manifest['artifact_zip_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    (a.out/'SHoEbox_multifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(commit=commit,source=str(source),artifacts=str(archive),verified_source_files=len(hashes))))


if __name__=='__main__':main()
