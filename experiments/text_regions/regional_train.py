"""Matched continuation of integrated training with causal fact supervision."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .composition_train import source_hashes
from .fact_supervision import attach_heads
from .integrated_eval import evaluate_capabilities, selection_key
from .integrated_train import load_integrated, save_integrated
from scripts.train_integrated_balanced import balanced_update


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--regional-weight', type=float, default=1.)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--validate-every', type=int, default=25)
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--head-seed', type=int, default=20261108)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.steps, args.validate_every, args.save_every, args.threads) < 1:
        p.error('Counts must be positive.')
    if not math.isfinite(args.regional_weight) or args.regional_weight < 0:
        p.error('Regional weight must be finite and nonnegative.')
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        p.error('Invalid stopping step.')
    torch.set_num_threads(args.threads)
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    if parent.get('schema') != 'shoebox-integrated-v1' or 'fact_supervision' in parent['protocol']:
        p.error('Start this phase from an integrated checkpoint without fact heads.')
    protocol = dict(parent['protocol'])
    protocol.update(steps=args.steps, validate_every=args.validate_every,
                    save_every=args.save_every, threads=args.threads,
                    parent_sha256=hashlib.sha256(args.parent.read_bytes()).hexdigest(),
                    parent_phase_step=parent['step'], source_hashes=source_hashes(),
                    gradient_policy='Clip each unweighted task gradient to norm 1, then average; clip merged norm 1',
                    optimizer='Preserve AdamW moments; constant inherited LR; per-task norm cap 1',
                    fact_supervision={'weight': args.regional_weight, 'head_seed': args.head_seed,
                                      'targets': 'Causal current value or observed factors; configured forward ancestors',
                                      'normalization': 'Mean per eligible node/factor group, then group mean'},
                    continuation='Preserve both stream cursors/states and both RNGs from the parent',
                    phase='Regional retained facts and corrections')
    script = Path(__file__).resolve().parents[2]/'scripts/train_integrated_balanced.py'
    protocol['source_hashes']['../scripts/train_integrated_balanced.py'] = hashlib.sha256(script.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        model, optimizer, rng, (streams, states), saved = load_integrated(args.out/'last.pt', protocol)
        start, history, totals = saved['step'], saved['history'], saved['totals']
    else:
        if (args.out/'initial.pt').exists(): p.error('Output exists; use --resume or a fresh directory.')
        model, optimizer, rng, (streams, states), _ = load_integrated(args.parent)
        attach_heads(model, optimizer, args.head_seed)
        progress = evaluate_capabilities(model, protocol['validation_count'], protocol['seed'])
        start, history = 0, [{'step': 0, 'validation': progress}]
        totals = {key: 0 for key in ('expert_executions', 'source_occurrences', 'external_events',
                                     'composition_episodes', 'correction_final_queries', 'views_final_queries',
                                     'correction_episodes_started', 'views_episodes_started',
                                     'correction_supervised', 'views_supervised')}
        totals.update(training_seconds=0., best_key=list(selection_key(progress)), best_step=0,
                      candidate_key=list(selection_key(progress)[2:]), candidate_step=0,
                      gate_streak=int(not progress['gate_failures']), mastered_step=None)
        for name in ('initial.pt', 'best.pt', 'best_candidate.pt', 'last.pt'):
            save_integrated(args.out/name, model, optimizer, rng, streams, states, 0, protocol, history, totals)
        print(json.dumps({'step': 0, 'scores': progress['scores'], 'failures': progress['gate_failures']}), flush=True)
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    stop = args.stop_after or args.steps
    if stop <= start: p.error('Checkpoint already reached stopping step.')
    model.train()
    for step in range(start+1, stop+1):
        tick = time.perf_counter()
        states, row = balanced_update(model, optimizer, rng, streams, states, protocol)
        row.update(step=step, training_seconds=time.perf_counter()-tick)
        for key in totals:
            if key in row: totals[key] += row[key]
        history.append(row)
        if step == start+1 or step % 10 == 0: print(json.dumps(row), flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            progress = evaluate_capabilities(model, protocol['validation_count'], protocol['seed'])
            row['validation'] = progress
            totals['gate_streak'] = 0 if progress['gate_failures'] else totals['gate_streak']+1
            if totals['gate_streak'] >= 2 and totals['mastered_step'] is None: totals['mastered_step'] = step
            if selection_key(progress) > tuple(totals['best_key']):
                totals['best_key'] = list(selection_key(progress)); totals['best_step'] = step
                save_integrated(args.out/'best.pt', model, optimizer, rng, streams, states, step, protocol, history, totals)
            if selection_key(progress)[2:] > tuple(totals['candidate_key']):
                totals['candidate_key'] = list(selection_key(progress)[2:]); totals['candidate_step'] = step
                save_integrated(args.out/'best_candidate.pt', model, optimizer, rng, streams, states, step, protocol, history, totals)
            # Retain each validation endpoint for independent audits and matched comparisons.
            save_integrated(args.out/f'step{step:04d}.pt', model, optimizer, rng, streams, states, step, protocol, history, totals)
            print(json.dumps({'step': step, 'scores': progress['scores'], 'failures': progress['gate_failures'],
                              'best_step': totals['best_step']}), flush=True)
        if step % args.save_every == 0 or step == stop:
            save_integrated(args.out/'last.pt', model, optimizer, rng, streams, states, step, protocol, history, totals)
            (args.out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    (args.out/'summary.json').write_text(json.dumps({'completed_step': stop, **totals}, indent=2)+'\n')


if __name__ == '__main__': main()
