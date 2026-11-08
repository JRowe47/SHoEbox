"""Run the active cortical skeleton checks; --legacy includes archived studies."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', action='store_true')
    args = parser.parse_args()
    result = subprocess.run([sys.executable, '-m', 'unittest', '-v',
                             'shoebox.test_skeleton', 'shoebox.test_fineweb', 'shoebox.test_wide'], cwd=ROOT)
    if result.returncode:
        return result.returncode
    if args.legacy:
        return subprocess.run([sys.executable, 'scripts/check_legacy.py'], cwd=ROOT).returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
