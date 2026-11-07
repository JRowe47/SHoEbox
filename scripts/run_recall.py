"""Resume a compact recall integration study from a common learned parent.

This is a training shakedown, not a feature-removal gate. Two distinct slots and
their parameter-matched pooled control train every weight and feedback control.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_arm(args, layout):
    out = args.out / layout
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-m', 'experiments.text_regions.task_memory_train',
        '--out', str(out), '--steps', str(args.steps), '--stream-preset', 'retention',
        '--write-mode', 'competitive', '--memory-aware-write', '--bootstrap-empty',
        '--payload-rank', '4', '--recall-slots', str(args.slots), '--recall-layout', layout,
        '--seed', str(args.seed), '--data-seed', str(args.data_seed),
        '--batch', '8', '--context', '64', '--lr', '.001', '--eval-every', '40',
        '--eval-count', str(args.eval_count), '--threads', '1', '--final-eval']
    cmd += ['--resume', str(out/'last.pt')] if (out/'last.pt').exists() else ['--init-from', str(args.out/'parent.pt')]
    with (out/'process.log').open('a') as log:
        log.write(json.dumps(cmd)+'\n'); log.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not args.skip_audits:
            for checkpoint in ('best', 'last'):
                for split in ('test', 'long_delay'):
                    subprocess.run([sys.executable, '-m', 'experiments.text_regions.task_query_audit',
                        '--checkpoint', str(out/(checkpoint+'.pt')), '--split', split,
                        '--count', str(args.eval_count*2), '--age-threshold', '64', '--threads', '1',
                        '--out', str(out/f'audit_{checkpoint}_{split}')],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = dict(layout=layout, complete=True, audited=not args.skip_audits, steps=args.steps)
    (out/'completion.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=160)
    p.add_argument('--slots', type=int, default=2)
    p.add_argument('--seed', type=int, default=59)
    p.add_argument('--data-seed', type=int, default=259)
    p.add_argument('--eval-count', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--skip-audits', action='store_true')
    args = p.parse_args()
    if min(args.steps, args.slots, args.eval_count, args.workers) < 1:
        p.error('Positive budgets required.')
    args.out = args.out.resolve(); args.parent = args.parent.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    parent = args.out/'parent.pt'
    digest = hashlib.sha256(args.parent.read_bytes()).hexdigest()
    if parent.exists():
        if hashlib.sha256(parent.read_bytes()).hexdigest() != digest:
            p.error('Saved parent does not match the requested initialization.')
    elif args.parent != parent:
        shutil.copy2(args.parent, parent)
    (args.out/'provenance.json').write_text(json.dumps(dict(parent_sha256=digest,
        parent_source=str(args.parent), source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT/'experiments/text_regions').glob('*.py'))},
        intent='Trainable recall integration; no removal decision from this small continuation.'), indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_arm, args, layout) for layout in ('distinct', 'pooled')]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)


if __name__ == '__main__':
    main()
