"""Package the exact published tree and hash every retained experiment artifact."""
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
    folder=ROOT/'artifacts/retained_composition'
    for name in ('training_audit','evaluation_audit','compatibility'):
        assert json.loads((folder/(name+'.json')).read_text())['all_passed']
    assert json.loads((folder/'run_status.json').read_text())['state']=='complete'
    assert 'All five experiment suites passed.' in (folder/'regression.log').read_text()
    source=a.out/'SHoEbox_retained_composition_source.zip'
    subprocess.run(['git','archive','--format=zip','--prefix=SHoEbox/','--output',str(source.resolve()),'HEAD'],cwd=ROOT,check=True)
    hashes={}
    with zipfile.ZipFile(source) as z:
        for record in git('ls-tree','-r','-z','HEAD').split(b'\0'):
            if not record:continue
            metadata,name=record.split(b'\t',1);name=name.decode();data=z.read('SHoEbox/'+name)
            assert data==git('cat-file','blob',metadata.split()[2].decode())
            hashes[name]=hashlib.sha256(data).hexdigest()
    files=sorted(path for name in ('retained_composition','retained_composition_pilot')
                 for path in (ROOT/'artifacts'/name).rglob('*') if path.is_file() and path.suffix!='.tmp')
    manifest=dict(publication_status='published',repository='https://github.com/JRowe47/SHoEbox',
        source_commit=commit,source_tree=git('rev-parse','HEAD^{tree}').decode().strip(),source_files=hashes,
        source_zip_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        artifact_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    artifact=a.out/'SHoEbox_retained_composition_artifacts.zip'
    with zipfile.ZipFile(artifact,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,'SHoEbox/'+str(path.relative_to(ROOT)))
        z.writestr('SHoEbox/artifacts/retained_composition/PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(artifact) as z:
        assert z.testzip() is None
        for name,digest in manifest['artifact_files'].items():assert hashlib.sha256(z.read('SHoEbox/'+name)).hexdigest()==digest
    manifest['artifact_zip_sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest()
    (a.out/'SHoEbox_retained_composition_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(commit=commit,source=str(source),artifacts=str(artifact),verified_source_files=len(hashes))))


if __name__=='__main__':main()
