"""Package a verified published tree and its recoverable experiment artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--published-commit', required=True)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    def git(*command): return subprocess.check_output(['git', *command], cwd=ROOT)
    commit = git('rev-parse','HEAD').decode().strip()
    assert commit == args.published_commit
    assert not git('status','--porcelain','--untracked-files=normal').strip()
    folder = ROOT/'artifacts/selective_publication'
    for name in ('evaluation_audit.json','training_audit.json','compatibility.json'):
        assert json.loads((folder/name).read_text())['all_passed']
    assert json.loads((folder/'run_status.json').read_text())['state'] == 'complete'
    assert 'All five experiment suites passed.' in (folder/'regression.log').read_text()
    source = args.out/'SHoEbox_selective_publication_source.zip'
    subprocess.run(['git','archive','--format=zip','--prefix=SHoEbox/','--output',str(source.resolve()),'HEAD'],
                   cwd=ROOT, check=True)
    source_hashes = {}
    with zipfile.ZipFile(source) as z:
        for record in git('ls-tree','-r','-z','HEAD').split(b'\0'):
            if not record: continue
            metadata, name = record.split(b'\t',1); name = name.decode()
            content = z.read('SHoEbox/'+name)
            assert content == git('cat-file','blob',metadata.split()[2].decode())
            source_hashes[name] = hashlib.sha256(content).hexdigest()
    files = sorted(path for folder_name in ('selective_publication','publication_pilot')
                   for path in (ROOT/'artifacts'/folder_name).rglob('*')
                   if path.is_file() and path.suffix != '.tmp' and path.name != 'PACKAGE_MANIFEST.json')
    manifest = dict(publication_status='published', repository='https://github.com/JRowe47/SHoEbox',
        source_commit=commit, source_tree=git('rev-parse','HEAD^{tree}').decode().strip(),
        source_zip_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), source_files=source_hashes,
        artifact_files={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    artifact = args.out/'SHoEbox_selective_publication_artifacts.zip'
    with zipfile.ZipFile(artifact,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files: z.write(path,'SHoEbox/'+str(path.relative_to(ROOT)))
        z.writestr('SHoEbox/artifacts/selective_publication/PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(artifact) as z:
        assert z.testzip() is None
        for name, digest in manifest['artifact_files'].items():
            assert hashlib.sha256(z.read('SHoEbox/'+name)).hexdigest() == digest
    manifest['artifact_zip_sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (args.out/'SHoEbox_selective_publication_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'source':str(source),'artifacts':str(artifact),'commit':commit,'verified_source_files':len(source_hashes)}))


if __name__ == '__main__': main()
