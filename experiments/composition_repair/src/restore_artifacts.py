"""Restore verified comparison artifacts or the exact Tiny Shakespeare split."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(data, expected, label):
    if digest(data) != expected:
        raise ValueError(f'SHA-256 mismatch: {label}')


def write_verified(files):
    # Check every destination before changing any of them.
    for relative, data in files.items():
        path = ROOT / relative
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f'Refusing to replace different contents: {path}')
    for relative, data in files.items():
        path = ROOT / relative
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(data)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    print(f'Verified {len(files)} artifacts.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--archive', type=Path)
    group.add_argument('--data-file', type=Path)
    group.add_argument('--download-data', action='store_true')
    args = parser.parse_args()
    if args.archive:
        manifest = json.loads((ROOT / 'ARTIFACT_MANIFEST.json').read_text())
        verify(args.archive.read_bytes(), manifest['archive_sha256'], args.archive)
        files = {}
        with zipfile.ZipFile(args.archive) as archive:
            for relative, expected in manifest['files'].items():
                if Path(relative).is_absolute() or '..' in Path(relative).parts:
                    raise ValueError('Unsafe artifact path')
                data = archive.read(manifest['archive_prefix'] + relative)
                verify(data, expected['sha256'], relative)
                if len(data) != expected['bytes']:
                    raise ValueError(f'Size mismatch: {relative}')
                files[relative] = data
    else:
        manifest = json.loads((ROOT / 'DATA_MANIFEST.json').read_text())
        if args.data_file:
            raw = args.data_file.read_bytes()
        else:
            with urllib.request.urlopen(manifest['source_url'], timeout=30) as response:
                raw = response.read(manifest['raw_bytes'] + 1)
        verify(raw, manifest['source_sha256'], 'Tiny Shakespeare source')
        split = manifest['split_byte_offset']
        files = {'data/train_segment.txt': raw[:split], 'data/heldout.bin': raw[split:]}
        verify(files['data/train_segment.txt'], manifest['train_sha256'], 'train split')
        verify(files['data/heldout.bin'], manifest['heldout_sha256'], 'held-out split')
    write_verified(files)


if __name__ == '__main__':
    main()
