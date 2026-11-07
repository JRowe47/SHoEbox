"""Paired payload-credit study with matched gate-only and fresh-read controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('gate_only', 'payload', 'fresh')


def run_arm(args, seed, name):
    out = args.out / str(seed) / name
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'experiments.text_regions.task_memory_train',
        '--out', str(out), '--stream-preset', 'retention', '--curriculum', 'acquisition',
        '--acquire-steps', str(args.acquire_steps), '--bridge-steps', str(args.bridge_steps),
        '--arm', 'fresh' if name == 'fresh' else 'integrated',
        '--write-mode', 'competitive', '--memory-aware-write', '--bootstrap-empty',
        '--payload-rank', str(args.rank), '--steps', str(args.steps), '--batch', str(args.batch),
        '--context', '64', '--seed', str(seed), '--data-seed', str(seed + 200),
        '--eval-count', str(args.eval_count), '--eval-every', str(args.eval_every),
        '--threads', '1', '--final-eval']
    if name == 'payload':
        command.append('--payload-credit')
    if (out / 'last.pt').exists():
        command += ['--resume', str(out / 'last.pt')]
    started = time.perf_counter()
    with (out / 'process.log').open('a') as log:
        log.write(json.dumps(command) + '\n'); log.flush()
        if not args.audit_only:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if name != 'fresh':
            for checkpoint in ('best', 'last'):
                for split in ('test', 'long_delay'):
                    for threshold in args.age_thresholds:
                        suffix = '' if threshold == 16 else f'_age{threshold}'
                        subprocess.run([sys.executable, '-m', 'experiments.text_regions.task_query_audit',
                            '--checkpoint', str(out / (checkpoint + '.pt')),
                            '--out', str(out / f'audit_{checkpoint}_{split}{suffix}'), '--split', split,
                            '--count', str(args.eval_count * 2), '--age-threshold', str(threshold), '--threads', '1'],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = dict(seed=seed, data_seed=seed+200, arm=name,
                  seconds=time.perf_counter()-started, completed=True)
    completion_name = 'audit_completion.json' if args.audit_only else 'completion.json'
    (out / completion_name).write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[41, 43])
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--acquire-steps', type=int, default=200)
    parser.add_argument('--bridge-steps', type=int, default=200)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--eval-count', type=int, default=16)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--age-thresholds', type=int, nargs='+', default=[16, 64])
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.acquire_steps, args.bridge_steps, args.batch, args.rank,
           args.eval_count, args.eval_every, args.workers) < 1:
        parser.error('Budgets must be positive.')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.arms)) != len(args.arms):
        parser.error('Seeds and arms must be unique; output directories cannot be shared.')
    if min(args.age_thresholds) < 0 or len(set(args.age_thresholds)) != len(args.age_thresholds):
        parser.error('Age thresholds must be nonnegative and unique.')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_arm, args, seed, name): (seed, name)
                   for seed in args.seeds for name in args.arms}
        for future in as_completed(pending):
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as error:
                seed, name = pending[future]
                failures.append((seed, name))
                print(json.dumps(dict(seed=seed, arm=name, error=str(error))), flush=True)
    if failures:
        raise SystemExit(f'Failed arms: {failures}')


if __name__ == '__main__':
    main()
