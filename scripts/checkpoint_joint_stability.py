"""Create a verified recovery archive during the paired continuation study."""
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
    a = p.parse_args()
    names = subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'], cwd=ROOT).decode().split('\0')
    files = {ROOT/name for name in names if name}
    files.update(p for p in (ROOT/'artifacts/joint_stability').rglob('*') if p.is_file() and p.suffix not in ('.tmp','.pyc'))
    files.add(ROOT/'data/tiny_shakespeare.txt')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.out.with_suffix('.zip.tmp')
    manifest = dict(status='in_progress', files={})
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            name = 'SHoEbox/'+str(path.relative_to(ROOT))
            content = path.read_bytes()
            if path.suffix == '.json': json.loads(content)
            manifest['files'][name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
        archive.writestr('SHoEbox/RECOVERY_MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        for name, digest in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    temporary.replace(a.out)
    print(json.dumps(dict(path=str(a.out), files=len(files), bytes=a.out.stat().st_size)), flush=True)


if __name__ == '__main__': main()
