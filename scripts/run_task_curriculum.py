"""Run paired acquisition, fresh-output, truncation, and direct-target arms."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    'curriculum': ('integrated', 'acquisition', 0),
    'fresh': ('fresh', 'acquisition', 0),
    'truncated': ('integrated', 'acquisition', 16),
    'direct': ('integrated', 'none', 0),
}


def run_arm(args, name):
    arm, curriculum, truncation = ARMS[name]
    out = args.out/name
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'experiments.text_regions.task_memory_train',
        '--out', str(out), '--stream-preset', 'retention', '--curriculum', curriculum,
        '--acquire-steps', str(args.acquire_steps), '--bridge-steps', str(args.bridge_steps),
        '--truncate-every', str(truncation), '--arm', arm,
        '--write-mode', 'competitive', '--memory-aware-write', '--bootstrap-empty',
        '--steps', str(args.steps), '--batch', str(args.batch), '--context', '64',
        '--seed', str(args.seed), '--data-seed', str(args.data_seed),
        '--eval-count', str(args.eval_count), '--eval-every', str(args.eval_every),
        '--threads', '1', '--final-eval']
    if (out/'last.pt').exists():
        command += ['--resume', str(out/'last.pt')]
    started = time.perf_counter()
    with (out/'process.log').open('a') as log:
        log.write(json.dumps(command)+'\n'); log.flush()
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if name != 'fresh':
            for checkpoint in ('best', 'last'):
                for split in ('test', 'long_delay'):
                    audit = [sys.executable, '-m', 'experiments.text_regions.task_query_audit',
                        '--checkpoint', str(out/(checkpoint+'.pt')), '--out', str(out/f'audit_{checkpoint}_{split}'),
                        '--split', split, '--count', str(args.eval_count*2), '--age-threshold', '16', '--threads', '1']
                    subprocess.run(audit, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = dict(arm=name, seed=args.seed, seconds=time.perf_counter()-started, completed=True)
    (out/'completion.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=29)
    p.add_argument('--data-seed', type=int, default=229)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--acquire-steps', type=int, default=200)
    p.add_argument('--bridge-steps', type=int, default=200)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--eval-count', type=int, default=16)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--arms', nargs='+', choices=tuple(ARMS), default=list(ARMS))
    args = p.parse_args()
    if min(args.steps, args.acquire_steps, args.bridge_steps, args.batch,
           args.eval_count, args.eval_every, args.workers) < 1:
        p.error('Budgets must be positive.')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_arm, args, name): name for name in args.arms}
        errors = []
        for future in as_completed(pending):
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as error:
                errors.append(pending[future])
                print(json.dumps(dict(arm=pending[future], error=str(error))), flush=True)
    if errors:
        raise SystemExit('Failed arms: '+', '.join(errors))


if __name__ == '__main__':
    main()
