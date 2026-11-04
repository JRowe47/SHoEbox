"""Run the preserved experiment suites in isolated processes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = [
    ('semantic_1m', ['test_model']),
    ('semantic_10m', ['test_model', 'test_scaled']),
    ('two_child', ['test_hierarchy']),
    ('hat_minkron', ['verify']),
]


def main():
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())['files']
    changed = [name for name, sha in manifest.items()
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha]
    if changed:
        print('Source differs from the recorded experiment: ' + ', '.join(changed), flush=True)
    failed = []
    for name, modules in SUITES:
        print(f'\nRunning {name}', flush=True)
        result = subprocess.run([sys.executable, '-m', 'unittest', '-v', *modules],
                                cwd=ROOT / 'experiments' / name)
        if result.returncode:
            failed.append(name)
    print('\nRunning text_regions', flush=True)
    result = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'experiments.text_regions.verify'], cwd=ROOT)
    if result.returncode:
        failed.append('text_regions')
    if failed:
        print('Failed suites: ' + ', '.join(failed), file=sys.stderr)
        return 1
    print('\nAll five experiment suites passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
