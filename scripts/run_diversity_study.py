"""Run or resume all twelve frozen graphs in the cohort-diversity reproduction."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.cohort_diversity import SETTINGS
from experiments.text_regions.residual_adapter import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/cohort_reproduction_v1'))
    parser.add_argument('--jobs', type=int, default=4); parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    out = ROOT / args.out
    if min(args.jobs, args.threads) < 1 or (out.exists() and not args.resume):
        parser.error('Use positive worker counts and a new output directory, or --resume.')
    manifest = json.loads((ROOT/'experiments/text_regions/results/shared_preservation/manifest.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    status = {'schema_version': 1, 'complete': False, 'settings': SETTINGS,
        'jobs': args.jobs, 'threads': args.threads,
        'runs': [{**r, 'status': 'pending'} for r in manifest['runs']]}
    lock = threading.Lock()

    def save():
        write_json(out/'status.json', status)

    def execute(row):
        start = time.monotonic()
        with lock:
            row['status'] = 'running'; save()
        command = [sys.executable, '-m', 'experiments.text_regions.cohort_diversity',
            '--run', row['name'], '--out', str(out/row['name']), '--threads', str(args.threads)]
        if args.resume:
            command.append('--resume')
        with (out/(row['name']+'.log')).open('a' if args.resume else 'w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        with lock:
            row.update(status='complete' if result.returncode == 0 else 'failed',
                returncode=result.returncode, seconds=time.monotonic()-start)
            save()
        print(json.dumps({k: row[k] for k in ('name', 'status', 'seconds', 'returncode')}), flush=True)
        return result.returncode

    save()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        failures = sum(f.result() != 0 for f in as_completed([pool.submit(execute, r) for r in status['runs']]))
    status['complete'] = failures == 0; save()
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
