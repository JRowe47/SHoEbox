"""Package runnable source and resumable payload-credit evidence with hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def archive(destination, files, metadata):
    manifest = dict(metadata, files={})
    temporary = destination.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as output:
        for path, name in files:
            content = path.read_bytes()
            output.writestr(name, content)
            manifest['files'][name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        output.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    temporary.replace(destination)
    with zipfile.ZipFile(destination) as check:
        if check.testzip() is not None:
            raise ValueError(f'Corrupt package: {destination}')
    return dict(file=str(destination), files=len(files), bytes=destination.stat().st_size,
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'artifacts/payload_credit')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = dict(source_base='d47ddc6', partial=args.partial,
                    study='paired delayed payload credit, seeds 41/43, 600 updates per arm')
    if not args.partial:
        for seed in (41, 43):
            for arm in ('gate_only', 'payload', 'fresh'):
                folder = args.root/str(seed)/arm
                if not json.loads((folder/'completion.json').read_text())['completed']:
                    raise ValueError(f'Incomplete run: {folder}')
                if arm != 'fresh':
                    for checkpoint in ('best', 'last'):
                        for split in ('test', 'long_delay'):
                            if not (folder/f'audit_{checkpoint}_{split}_age64/summary.json').is_file():
                                raise ValueError(f'Missing beyond-window audit: {folder}')
    evidence = [(p, str(Path('payload_credit')/p.relative_to(args.root)))
                for p in sorted(args.root.rglob('*')) if p.is_file() and p.suffix != '.tmp']
    print(json.dumps(archive(args.out/'SHoEbox_Payload_Credit_Artifacts.zip', evidence, metadata)))
    tracked = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
                                      cwd=ROOT).decode().split('\0')
    source = [(ROOT/p, 'SHoEbox/'+p) for p in sorted(set(tracked)) if p and (ROOT/p).is_file()]
    print(json.dumps(archive(args.out/'SHoEbox_Payload_Credit_Source.zip', source, metadata)))


if __name__ == '__main__':
    main()
