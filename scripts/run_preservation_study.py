"""Fit shared-margin residual corrections to the twelve existing frozen graphs."""
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
from experiments.text_regions.residual_adapter import write_json
from experiments.text_regions.shared_preservation import PENALTIES, MARGIN_CAP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/shared_preservation_v1'))
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    out = ROOT / args.out
    if out.exists() or min(args.jobs, args.threads) < 1:
        parser.error('Use a new output directory and positive worker/thread counts.')
    manifest = json.loads((ROOT / 'experiments/text_regions/results/residual_adapter/manifest.json').read_text())
    runs = []
    for row in manifest['runs']:
        prior = Path(manifest['source_directory']) / row['name']
        emission = row['prior_run']
        for path in (prior / 'residuals.pt', prior / 'validation.pt', prior / 'results.json', Path(emission) / 'features.pt'):
            if not (ROOT / path).is_file():
                parser.error('Restore prerequisite artifact: ' + str(path))
        command = [sys.executable, '-m', 'experiments.text_regions.shared_preservation',
            '--prior-run', str(prior), '--emission-run', emission,
            '--out', str(out / row['name']), '--threads', str(args.threads)]
        runs.append({**row, 'residual_run': str(prior), 'command': command, 'status': 'pending'})
    out.mkdir(parents=True)
    status = {'schema_version': 1, 'complete': False, 'runs': runs,
        'settings': {'penalties': list(PENALTIES), 'margin_cap': MARGIN_CAP, 'hidden': 16,
            'max_iter': 400, 'l2': .001, 'adapter_seeds': [20261121, 20261122, 20261123],
            'jobs': args.jobs, 'threads': args.threads, 'additional_expert_executions': 0}}
    lock = threading.Lock()

    def save():
        write_json(out / 'status.json', status)

    def execute(run):
        start = time.monotonic()
        with lock:
            run['status'] = 'running'
            save()
        with (out / (run['name'] + '.log')).open('w') as log:
            result = subprocess.run(run['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        with lock:
            run.update(returncode=result.returncode, seconds=time.monotonic() - start,
                status='complete' if result.returncode == 0 else 'failed')
            save()
        print(json.dumps({k: run[k] for k in ('name', 'status', 'seconds', 'returncode')}), flush=True)
        return result.returncode

    save()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        failures = sum(f.result() != 0 for f in as_completed([pool.submit(execute, r) for r in runs]))
    status['complete'] = not failures
    save()
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
