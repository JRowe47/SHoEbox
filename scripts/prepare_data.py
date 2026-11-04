"""Fetch the exact experiment dataset, or use a hash-verified offline copy."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
SHA256 = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
EXPERIMENTS = ('semantic_1m', 'semantic_10m', 'two_child', 'hat_minkron')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Existing Tiny Shakespeare file; its hash must match.')
    args = parser.parse_args()
    cache = ROOT / 'data' / 'tiny_shakespeare.txt'
    if args.source:
        data = args.source.read_bytes()
    elif cache.exists():
        data = cache.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:
            data = response.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != SHA256:
        raise SystemExit(f'Dataset SHA-256 mismatch: expected {SHA256}, received {actual}. No files written.')
    for target in [cache, *(ROOT / 'experiments' / name / 'data' / 'tiny_shakespeare.txt' for name in EXPERIMENTS)]:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(f'Prepared {len(data):,} bytes for all four experiments. SHA-256 verified: {actual}')


if __name__ == '__main__':
    main()
