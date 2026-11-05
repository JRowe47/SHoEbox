"""Package the exact git tree and completed experiment with SHA-256 manifests."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT)
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--published-commit',required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    commit=git('rev-parse','HEAD').decode().strip()
    if commit != args.published_commit:raise ValueError('Package HEAD must equal the published commit.')
    if git('status','--porcelain','--untracked-files=normal').strip():raise ValueError('Package a clean source tree.')
    tree=git('rev-parse','HEAD^{tree}').decode().strip()
    source=args.out/'SHoEbox_gradient_horizon_source.zip'
    subprocess.run(['git','archive','--format=zip','--prefix=SHoEbox/','--output',str(source.resolve()),'HEAD'],cwd=ROOT,check=True)
    members={}
    with zipfile.ZipFile(source) as z:
        for record in git('ls-tree','-r','-z','HEAD').split(b'\0'):
            if not record:continue
            meta,path=record.split(b'\t',1);path=path.decode();sha=meta.split()[2].decode()
            content=z.read('SHoEbox/'+path)
            if content != git('cat-file','blob',sha):raise AssertionError(path)
            members[path]=hashlib.sha256(content).hexdigest()
    artifact_root=ROOT/'artifacts/gradient_horizon'
    status=json.loads((artifact_root/'run_status.json').read_text())
    if status['state'] != 'complete':raise ValueError('Experiment driver is not complete.')
    for name in ('selection.json','evaluation_audit.json'):
        if not json.loads((artifact_root/name).read_text())['all_passed']:raise ValueError(name)
    artifact=args.out/'SHoEbox_gradient_horizon_artifacts.zip'
    files=sorted(path for path in artifact_root.rglob('*') if path.is_file() and path.suffix != '.tmp')
    manifest={'publication_status':'published','repository':'https://github.com/JRowe47/SHoEbox',
              'source_commit':commit,'source_tree':tree,'source_zip_sha256':digest(source),
              'source_members_verified_against_git':len(members),
              'source_files':members,'artifact_files':{str(path.relative_to(ROOT)):digest(path) for path in files}}
    with zipfile.ZipFile(artifact,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in files:z.write(path,'SHoEbox/'+str(path.relative_to(ROOT)))
        z.writestr('SHoEbox/artifacts/gradient_horizon/PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(artifact) as z:
        assert z.testzip() is None
        for name,expected in manifest['artifact_files'].items():
            assert hashlib.sha256(z.read('SHoEbox/'+name)).hexdigest()==expected
    manifest['artifact_zip_sha256']=digest(artifact)
    (args.out/'SHoEbox_gradient_horizon_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'source':str(source),'artifact':str(artifact),'commit':commit,
                      'tree':tree,'source_members':len(members),'artifact_members':len(files)}))


if __name__=='__main__':main()
