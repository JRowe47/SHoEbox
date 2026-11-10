"""Run cortical skeleton and online CPU checks; --legacy includes archived studies."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', action='store_true')
    args = parser.parse_args()
    result = subprocess.run([sys.executable, '-m', 'unittest', '-v',
                             'shoebox.test_skeleton', 'shoebox.test_fineweb', 'shoebox.test_wide',
                             'shoebox.test_fused_kron', 'shoebox.test_fused_experts',
                             'shoebox.test_sparse_cpu', 'shoebox.test_cpu_train'], cwd=ROOT)
    if result.returncode:
        return result.returncode
    for experts in (12, 64):
        env = dict(os.environ, SHOEBOX_EXPERTS=str(experts),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        result = subprocess.run(
            [sys.executable, 'experiments/online_cpu/check_packaging.py'],
            cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    result = subprocess.run(
        [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
        cwd=ROOT / 'experiments' / 'scaling_cpu', env=env)
    if result.returncode:
        return result.returncode
    if args.legacy:
        return subprocess.run([sys.executable, 'scripts/check_legacy.py'], cwd=ROOT).returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


