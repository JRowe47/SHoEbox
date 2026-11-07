"""Matched 2x2 producer-state / read-source study, with exact resumable arms."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('always', 'held', 'always_fresh', 'held_fresh')


def run_arm(args, seed, name):
    out = args.out / str(seed) / name
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'experiments.text_regions.task_memory_train',
        '--out', str(out), '--stream-preset', 'retention', '--curriculum', 'acquisition',
        '--acquire-steps', str(args.acquire_steps), '--bridge-steps', str(args.bridge_steps),
        '--arm', 'fresh' if name.endswith('fresh') else 'integrated',
        '--state-commit', 'publication' if name.startswith('held') else 'always',
        '--write-mode', 'competitive', '--memory-aware-write', '--bootstrap-empty',
        '--payload-rank', '4', '--steps', str(args.steps), '--batch', str(args.batch),
        '--context', '64', '--seed', str(seed), '--data-seed', str(seed + 200),
        '--eval-count', str(args.eval_count), '--eval-every', str(args.eval_every),
        '--threads', '1', '--final-eval']
    if (out / 'last.pt').exists():
        command += ['--resume', str(out / 'last.pt')]
    started = time.perf_counter()
    with (out / 'state_commit_process.log').open('a') as log:
        log.write(json.dumps(command) + '\n'); log.flush()
        if not args.audit_only:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not args.skip_audits:
            for checkpoint in ('best', 'last'):
                for split in ('test', 'long_delay'):
                    subprocess.run([sys.executable, '-m', 'experiments.text_regions.task_query_audit',
                        '--checkpoint', str(out / (checkpoint + '.pt')),
                        '--out', str(out / f'state_audit_{checkpoint}_{split}'), '--split', split,
                        '--count', str(args.eval_count * 2), '--age-threshold', '64',
                        '--producer-state', '--threads', '1'],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = dict(seed=seed, data_seed=seed+200, arm=name,
                  seconds=time.perf_counter()-started, completed=True,
                  trained=not args.audit_only, audited=not args.skip_audits)
    completion = 'state_audit_completion.json' if args.audit_only else 'state_completion.json'
    (out / completion).write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[41, 43])
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--acquire-steps', type=int, default=200)
    p.add_argument('--bridge-steps', type=int, default=200)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--eval-count', type=int, default=16)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--skip-audits', action='store_true')
    args = p.parse_args()
    if min(args.steps, args.acquire_steps, args.bridge_steps, args.batch,
           args.eval_count, args.eval_every, args.workers) < 1:
        p.error('Budgets must be positive.')
    if args.audit_only and args.skip_audits:
        p.error('Audit-only cannot skip audits.')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.arms)) != len(args.arms):
        p.error('Seeds and arms must be unique.')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_arm, args, seed, name): (seed, name)
                   for seed in args.seeds for name in args.arms}
        for future in as_completed(pending):
            seed, name = pending[future]
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as error:
                failures.append((seed, name))
                print(json.dumps(dict(seed=seed, arm=name, error=str(error))), flush=True)
    if failures:
        raise SystemExit(f'Failed arms: {failures}')


if __name__ == '__main__':
    main()
