"""Joint regional training with separate runtime streams and one common decoder."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch

from ..hat_minkron.train import optimizer_for
from .composition import training_loss
from .composition_data import training_batch
from .composition_train import atomic_save, build_model, load_checkpoint, source_hashes
from .config import GraphConfig
from .episode_data import EpisodeStream
from .integrated_eval import evaluate_capabilities, selection_key
from .state import GraphState
from .view_data import ViewStream


def make_streams(protocol):
    return {'correction': EpisodeStream('correction', protocol['stream_batch'],
                                        protocol['memory_gap'], 4, protocol['seed']),
            'views': ViewStream(batch_size=protocol['stream_batch'], gap=protocol['view_gap'],
                                seed=protocol['seed'], source_schedule='balanced')}


def stream_loss(model, chunk, state, regional_weight=0.):
    # Query labels are loss-only; forward receives observations and causal masks.
    mask = chunk.targets.ne(-100)
    result = model(chunk.inputs, state=state, reset=chunk.reset,
                   availability=chunk.availability, rounds=2, feedback=True,
                   return_node_activity=tuple(model.nodes) if regional_weight else ())
    loss = (torch.nn.functional.cross_entropy(result['logits'][mask], chunk.targets[mask])
            if bool(mask.any()) else result['logits'].sum()*0.)
    result['ce'] = loss
    loss = loss + model.cfg.auxiliary_loss_weight*result['aux'] + model.cfg.hat_balance_weight*result['hat_aux']
    if regional_weight:
        from .fact_supervision import retention_loss
        local, result['retention'] = retention_loss(model, chunk, result['node_activity'])
        loss = loss + regional_weight*local
    return loss, result


def joint_update(model, optimizer, rng, streams, states, protocol):
    """Accumulate three task means before one optimizer step; no replay cache."""
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups: group['lr'] = protocol['lr']
    batch = training_batch(rng, protocol['composition_batch'])
    loss, ce, prefix, result = training_loss(model, batch, rounds=2, feedback=True,
                                            source_reconstruction=.5, regional_supervision=1.)
    weights = protocol['task_weights']
    denominator = sum(weights.values())
    if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite composition loss.')
    (loss*weights['composition']/denominator).backward()
    row = {'composition_nats': float(ce.detach()),
           'composition_accuracy': float(result['logits'][:, -1].argmax(-1).eq(batch.answers).float().mean()),
           'composition_episodes': batch.size,
           'expert_executions': prefix['expert_executions']+result['expert_executions'],
           'source_occurrences': prefix['source_occurrences']+result['source_occurrences'],
           'external_events': prefix['external_events']+result['external_events']}
    del loss, ce, prefix, result
    next_states = {}
    for name, stream in streams.items():
        chunk = stream.next_chunk(protocol['segment'])
        loss, result = stream_loss(model, chunk, states[name])
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite stream loss.')
        (loss*weights[name]/denominator).backward()
        next_states[name] = result['state'].detach()
        row[name+'_nats'] = float(result['ce'].detach())
        row[name+'_supervised'] = int(chunk.targets.ne(-100).sum())
        row[name+'_final_queries'] = int(chunk.final.sum())
        row[name+'_episodes_started'] = int(chunk.reset[:, 0].sum())
        for key in ('expert_executions', 'source_occurrences', 'external_events'): row[key] += result[key]
        del loss, result
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    row.update(gradient_norm=float(norm), lr=protocol['lr'])
    return next_states, row


def save_integrated(path, model, optimizer, rng, streams, states, step, protocol, history, totals):
    atomic_save(path, dict(schema='shoebox-integrated-v1', config=model.cfg.to_dict(),
                          weights=model.state_dict(), optimizer=optimizer.state_dict(),
                          rng=rng.getstate(), torch_rng=torch.get_rng_state(), step=step,
                          protocol=protocol, history=history, totals=totals,
                          streams={name: stream.state_dict() for name, stream in streams.items()},
                          states={name: None if state is None else state.to_dict() for name, state in states.items()}))


def load_integrated(path, expected_protocol=None):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('schema') != 'shoebox-integrated-v1': raise ValueError('Unknown integrated checkpoint schema.')
    if expected_protocol is not None and saved['protocol'] != expected_protocol:
        raise ValueError('Resume requires identical source and protocol.')
    model = build_model(GraphConfig.from_dict(saved['config']), .5, 1.)
    if 'fact_supervision' in saved['protocol']:
        from .fact_supervision import attach_heads
        attach_heads(model, seed=saved['protocol']['fact_supervision']['head_seed'])
    model.load_state_dict(saved['weights'])
    optimizer = optimizer_for(model, saved['protocol']['lr']); optimizer.load_state_dict(saved['optimizer'])
    rng = random.Random(); rng.setstate(saved['rng'])
    torch.set_rng_state(saved['torch_rng'])
    streams = make_streams(saved['protocol'])
    for name, stream in streams.items(): stream.load_state_dict(saved['streams'][name])
    states = {name: None if state is None else GraphState.from_dict(state)
              for name, state in saved['states'].items()}
    return model, optimizer, rng, (streams, states), saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--stream-batch', type=int, default=16)
    p.add_argument('--composition-batch', type=int, default=64)
    p.add_argument('--segment', type=int, default=16)
    p.add_argument('--memory-gap', type=int, default=16)
    p.add_argument('--view-gap', type=int, default=8)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--validate-every', type=int, default=50)
    p.add_argument('--validation-count', type=int, default=64)
    p.add_argument('--save-every', type=int, default=25)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.steps, args.stream_batch, args.composition_batch, args.segment, args.validate_every,
           args.validation_count, args.save_every, args.threads) < 1 or min(args.memory_gap, args.view_gap) < 0:
        p.error('Invalid training dimensions.')
    if args.composition_batch % 4 or args.validation_count % 64:
        p.error('Composition batches require multiples of four; evaluation requires complete 64-tuple cycles.')
    if not math.isfinite(args.lr) or args.lr <= 0: p.error('Learning rate must be finite and positive.')
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps: p.error('Invalid stop step.')
    torch.set_num_threads(args.threads)
    protocol = {k: v for k, v in vars(args).items() if k not in ('parent', 'out', 'resume', 'stop_after')}
    protocol.update(parent_sha256=hashlib.sha256(args.parent.read_bytes()).hexdigest(),
                    source_hashes=source_hashes(), task_weights={'composition': 1., 'correction': 1., 'views': 1.},
                    source_reconstruction=.5, regional_supervision=1., rounds=2, feedback=True,
                    selection='All gates; then composition gates; then minimum accuracy, macro accuracy, lower macro NLL',
                    optimizer='Preserved AdamW moments; constant phase LR; three task means; clip norm 1',
                    streams='Fresh correction and balanced-view episodes; independent carried state; detach every segment',
                    inference='Same seven-node graph and common 256-byte head for every task; no task ID input')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        model, optimizer, rng, runtime, saved = load_integrated(args.out/'last.pt', protocol)
        streams, states = runtime
        start, history, totals = saved['step'], saved['history'], saved['totals']
    else:
        if (args.out/'initial.pt').exists(): p.error('Output exists; use --resume or a new directory.')
        model, optimizer, rng, parent = load_checkpoint(args.parent)
        if not hasattr(model, 'source_reconstruction') or not hasattr(model, 'regional_readout'):
            p.error('Parent requires source and regional training heads.')
        if model.cfg.rounds != 2 or not model.cfg.recurrent_feedback:
            p.error('Parent must be a two-round communicating graph.')
        streams = make_streams(protocol); states = {name: None for name in streams}
        start = 0
        progress = evaluate_capabilities(model, args.validation_count, args.seed)
        history = [{'step': 0, 'validation': progress}]
        totals = {'expert_executions': 0, 'source_occurrences': 0, 'external_events': 0,
                  'training_seconds': 0., 'composition_episodes': 0,
                  'correction_final_queries': 0, 'views_final_queries': 0,
                  'correction_episodes_started': 0, 'views_episodes_started': 0,
                  'correction_supervised': 0, 'views_supervised': 0,
                  'best_key': list(selection_key(progress)), 'best_step': 0,
                  'candidate_key': list(selection_key(progress)[2:]), 'candidate_step': 0,
                  'gate_streak': int(not progress['gate_failures']), 'mastered_step': None}
        for name in ('initial.pt', 'best.pt', 'best_candidate.pt'):
            save_integrated(args.out/name, model, optimizer, rng, streams, states, 0, protocol, history, totals)
        print(json.dumps({'step': 0, 'scores': progress['scores'], 'failures': progress['gate_failures']}), flush=True)
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    stop = args.stop_after or args.steps
    if stop <= start: p.error('Checkpoint already reached stopping step.')
    model.train()
    for step in range(start+1, stop+1):
        tick = time.perf_counter()
        states, row = joint_update(model, optimizer, rng, streams, states, protocol)
        row.update(step=step, training_seconds=time.perf_counter()-tick)
        for key in totals:
            if key in row: totals[key] += row[key]
        history.append(row)
        if step == start+1 or step % 25 == 0: print(json.dumps(row), flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            progress = evaluate_capabilities(model, args.validation_count, args.seed)
            row['validation'] = progress
            totals['gate_streak'] = 0 if progress['gate_failures'] else totals['gate_streak']+1
            if totals['gate_streak'] >= 2 and totals['mastered_step'] is None: totals['mastered_step'] = step
            if selection_key(progress) > tuple(totals['best_key']):
                totals['best_key'] = list(selection_key(progress)); totals['best_step'] = step
                save_integrated(args.out/'best.pt', model, optimizer, rng, streams, states, step, protocol, history, totals)
            if selection_key(progress)[2:] > tuple(totals['candidate_key']):
                totals['candidate_key'] = list(selection_key(progress)[2:]); totals['candidate_step'] = step
                save_integrated(args.out/'best_candidate.pt', model, optimizer, rng, streams, states, step, protocol, history, totals)
            print(json.dumps({'step': step, 'scores': progress['scores'], 'failures': progress['gate_failures'],
                              'best_step': totals['best_step']}), flush=True)
        if step % args.save_every == 0 or step == stop:
            save_integrated(args.out/'last.pt', model, optimizer, rng, streams, states, step, protocol, history, totals)
            (args.out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    (args.out/'summary.json').write_text(json.dumps({'completed_step': stop, **totals}, indent=2)+'\n')


if __name__ == '__main__': main()
