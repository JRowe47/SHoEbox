"""Run or resume the 72 full-rank refits across twelve retained frozen graphs."""
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
from experiments.text_regions.full_rank_readout import SETTINGS
from experiments.text_regions.residual_adapter import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('artifacts/cohort_reproduction_v1'))
    p.add_argument('--out', type=Path, default=Path('artifacts/full_rank_v1'))
    p.add_argument('--jobs', type=int, default=4); p.add_argument('--threads', type=int, default=2)
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    out = ROOT/args.out
    if min(args.jobs, args.threads) < 1 or (out.exists() and not args.resume):
        p.error('Use positive worker counts and a new output directory, or --resume.')
    manifest = json.loads((ROOT/'experiments/text_regions/results/shared_preservation/manifest.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    status = {'schema_version': 1, 'complete': False, 'settings': SETTINGS,
        'source': str(args.source), 'jobs': args.jobs, 'threads': args.threads,
        'runs': [{**r, 'status': 'pending'} for r in manifest['runs']]}
    lock = threading.Lock()

    def execute(row):
        start = time.monotonic()
        with lock:
            row['status'] = 'running'; write_json(out/'status.json', status)
        command = [sys.executable, '-m', 'experiments.text_regions.full_rank_readout',
            '--source', str(args.source), '--run', row['name'], '--out', str(out/row['name']),
            '--threads', str(args.threads)] + (['--resume'] if args.resume else [])
        with (out/(row['name']+'.log')).open('a' if args.resume else 'w') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        with lock:
            row.update(status='complete' if result.returncode == 0 else 'failed',
                returncode=result.returncode, seconds=time.monotonic()-start)
            write_json(out/'status.json', status)
        print(json.dumps({k: row[k] for k in ('name', 'status', 'seconds', 'returncode')}), flush=True)
        return result.returncode

    write_json(out/'status.json', status)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        failures = sum(f.result() != 0 for f in as_completed([pool.submit(execute, r) for r in status['runs']]))
    status['complete'] = failures == 0; write_json(out/'status.json', status)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
