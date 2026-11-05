"""Run the predeclared complete-episode gradient-horizon comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch

from .composition_train import source_hashes
from .gradient_horizon import horizon_update
from .integrated_eval import evaluate_capabilities, selection_key
from .integrated_train import load_integrated, save_integrated, make_streams


COUNTERS = ('expert_executions', 'source_occurrences', 'external_events', 'composition_episodes',
            'correction_final_queries', 'views_final_queries', 'correction_episodes_started',
            'views_episodes_started', 'correction_supervised', 'views_supervised')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--detach-between', action='store_true')
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--stream-batch', type=int, default=16)
    p.add_argument('--composition-batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=20261109)
    p.add_argument('--validate-every', type=int, default=16)
    p.add_argument('--save-every', type=int, default=4)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.steps, args.stream_batch, args.composition_batch, args.validate_every, args.save_every) < 1:
        p.error('Counts must be positive.')
    if args.composition_batch % 4: p.error('Composition batches must be multiples of four.')
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps: p.error('Invalid stop step.')
    torch.set_num_threads(1)
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    if parent.get('schema') != 'shoebox-integrated-v1' or 'fact_supervision' not in parent['protocol']:
        p.error('Start from a regional-facts checkpoint with all four auxiliary heads.')
    protocol = dict(parent['protocol'])
    protocol.update(steps=args.steps, stream_batch=args.stream_batch, composition_batch=args.composition_batch,
                    seed=args.seed, validate_every=args.validate_every, save_every=args.save_every,
                    threads=1, parent_sha256=hashlib.sha256(args.parent.read_bytes()).hexdigest(),
                    parent_phase_step=parent['step'], source_hashes=source_hashes(),
                    detach_between=args.detach_between, segment=16,
                    fact_supervision={**parent['protocol']['fact_supervision'], 'weight': 1.},
                    phase='Complete-episode gradient horizon',
                    continuation='Preserve weights, optimizer moments and composition RNG; fresh complete episodes and reset runtime',
                    loss_normalization='Query mean and eligible node/factor means over each complete episode',
                    optimizer='Inherited AdamW moments and LR; cap each task gradient at 1 before averaging',
                    evaluation_count=512)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        model, optimizer, rng, (streams, states), saved = load_integrated(args.out/'last.pt', protocol)
        start, history, totals = saved['step'], saved['history'], saved['totals']
    else:
        if (args.out/'initial.pt').exists(): p.error('Output exists; use --resume or a new directory.')
        model, optimizer, rng, _, _ = load_integrated(args.parent)
        streams = make_streams(protocol); states = {name: None for name in streams}
        progress = evaluate_capabilities(model, protocol['validation_count'], protocol['seed'])
        start, history = 0, [{'step': 0, 'validation': progress}]
        totals = {key: 0 for key in COUNTERS}
        totals.update(training_seconds=0., best_key=list(selection_key(progress)), best_step=0)
        for name in ('initial.pt', 'best.pt', 'last.pt'):
            save_integrated(args.out/name, model, optimizer, rng, streams, states, 0, protocol, history, totals)
        print(json.dumps({'step': 0, 'scores': progress['scores'], 'failures': progress['gate_failures']}), flush=True)
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    stop = args.stop_after or args.steps
    if stop <= start: p.error('Checkpoint already reached stopping step.')
    model.train()
    for step in range(start+1, stop+1):
        tick = time.perf_counter()
        row = horizon_update(model, optimizer, rng, streams, protocol)
        row.update(step=step, training_seconds=time.perf_counter()-tick)
        for key in (*COUNTERS, 'training_seconds'): totals[key] += row[key]
        history.append(row)
        print(json.dumps(row), flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            progress = evaluate_capabilities(model, protocol['validation_count'], protocol['seed'])
            row['validation'] = progress
            if selection_key(progress) > tuple(totals['best_key']):
                totals['best_key'] = list(selection_key(progress)); totals['best_step'] = step
                save_integrated(args.out/'best.pt', model, optimizer, rng, streams, states, step, protocol, history, totals)
            save_integrated(args.out/f'step{step:04d}.pt', model, optimizer, rng, streams, states, step, protocol, history, totals)
            print(json.dumps({'step': step, 'scores': progress['scores'],
                              'failures': progress['gate_failures'], 'best_step': totals['best_step']}), flush=True)
        if step % args.save_every == 0 or step == stop:
            save_integrated(args.out/'last.pt', model, optimizer, rng, streams, states, step, protocol, history, totals)
            (args.out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    (args.out/'summary.json').write_text(json.dumps({'completed_step': stop, **totals}, indent=2)+'\n')


if __name__ == '__main__': main()
