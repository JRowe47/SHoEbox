"""Package the committed source tree and complete regional-fact run artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--publication-status',choices=('unpublished','published'),default='unpublished')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    git=lambda *parts: subprocess.check_output(['git',*parts],cwd=ROOT,text=True).strip()
    if git('status','--porcelain'):
        raise SystemExit('Commit the final report, results and source before packaging.')
    commit,tree=git('rev-parse','HEAD'),git('rev-parse','HEAD^{tree}')
    if args.publication_status == 'published':
        remote = git('ls-remote','origin','refs/heads/main').split()
        if not remote or remote[0] != commit:
            raise SystemExit('Published packaging requires the remote main branch to match HEAD.')
    source=args.out/'SHoEbox_regional_facts_source.zip'
    subprocess.run(['git','archive','--format=zip','--prefix=SHoEbox/',
                    '--output',str(source.resolve()),'HEAD'],cwd=ROOT,check=True)
    files=sorted(p for p in (ROOT/'artifacts/regional_facts').rglob('*')
                 if p.is_file() and p.suffix in ('.pt','.json','.log'))
    files.append(ROOT/'artifacts/integrated/selected.pt')
    manifest={'source_commit':commit,'source_tree':tree,
              'publication_status':args.publication_status,
              'published_commit':commit if args.publication_status == 'published' else None,
              'source_archive_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    archive=args.out/'SHoEbox_regional_facts_artifacts.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in files:z.write(path,'SHoEbox/'+str(path.relative_to(ROOT)))
        z.writestr('SHoEbox/REGIONAL_FACTS_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    for path in (source,archive):
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:raise RuntimeError('Archive integrity failure.')
        print(json.dumps({'file':str(path.resolve()),'bytes':path.stat().st_size,
                          'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
