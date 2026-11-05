"""Matched integration follow-up: cap each task gradient before accumulation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.text_regions.composition import training_loss
from experiments.text_regions.composition_data import training_batch
from experiments.text_regions.composition_train import load_checkpoint, source_hashes
from experiments.text_regions.integrated_train import make_streams, stream_loss, save_integrated, load_integrated
from experiments.text_regions.integrated_eval import evaluate_capabilities, selection_key


@torch.no_grad()
def clip_task_gradients(parameters, max_norm=1.):
    """Keep ordinary clipping exact, with FP64 reduction on FP32 overflow.

    Recurrent gradients can be finite while their FP32 sum of squares is not.
    The fallback changes only the norm reduction, not the gradient direction or
    cap. NaN/Inf gradients still fail before any gradient is modified.
    """
    parameters = list(parameters)
    try:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    except RuntimeError as error:
        if 'is non-finite' not in str(error):
            raise
        gradients = [p.grad for p in parameters if p.grad is not None]
        norm = torch.linalg.vector_norm(torch.stack([g.double().norm() for g in gradients]))
        if not bool(torch.isfinite(norm)):
            raise
        coefficient = (max_norm/(norm + 1e-6)).clamp(max=1.)
        for gradient in gradients:
            gradient.mul_(coefficient.to(device=gradient.device, dtype=gradient.dtype))
        return norm


def balanced_update(model, optimizer, rng, streams, states, protocol):
    parameters = list(model.parameters())
    accumulated = [None for _ in parameters]
    weights = protocol['task_weights']; denominator = sum(weights.values())
    for group in optimizer.param_groups: group['lr'] = protocol['lr']
    row = {'expert_executions': 0, 'source_occurrences': 0, 'external_events': 0}
    next_states = {}
    for name in ('composition', 'correction', 'views'):
        optimizer.zero_grad(set_to_none=True)
        if name == 'composition':
            batch = training_batch(rng, protocol['composition_batch'])
            loss, ce, prefix, result = training_loss(model, batch, 2, True, .5, 1.)
            row.update(composition_nats=float(ce.detach()), composition_episodes=batch.size,
                       composition_accuracy=float(result['logits'][:, -1].argmax(-1).eq(batch.answers).float().mean()))
            for key in ('expert_executions', 'source_occurrences', 'external_events'): row[key] += prefix[key]
            del ce, prefix
        else:
            chunk = streams[name].next_chunk(protocol['segment'])
            loss, result = stream_loss(model, chunk, states[name],
                                      protocol.get('fact_supervision', {}).get('weight', 0.))
            next_states[name] = result['state'].detach()
            row[name+'_nats'] = float(result['ce'].detach())
            row[name+'_supervised'] = int(chunk.targets.ne(-100).sum())
            row[name+'_final_queries'] = int(chunk.final.sum())
            row[name+'_episodes_started'] = int(chunk.reset[:, 0].sum())
            for key, value in result.get('retention', {}).items():
                row[name+'_'+key] = value
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite task loss.')
        loss.backward()
        norm = clip_task_gradients(parameters)
        row[name+'_gradient_norm'] = float(norm)
        for index, parameter in enumerate(parameters):
            if parameter.grad is not None:
                contribution = parameter.grad.detach()*(weights[name]/denominator)
                if accumulated[index] is None: accumulated[index] = contribution.clone()
                else: accumulated[index].add_(contribution)
        for key in ('expert_executions', 'source_occurrences', 'external_events'): row[key] += result[key]
        del loss, result
    for parameter, gradient in zip(parameters, accumulated): parameter.grad = gradient
    norm = clip_task_gradients(parameters)
    optimizer.step()
    row.update(gradient_norm=float(norm), lr=protocol['lr'])
    return next_states, row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--baseline-protocol', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.steps < 1 or (args.stop_after is not None and not 1 <= args.stop_after <= args.steps):
        p.error('Invalid update budget.')
    protocol = json.loads(args.baseline_protocol.read_text())
    protocol.update(steps=args.steps, source_hashes=source_hashes(),
                    gradient_policy='Clip each unweighted task gradient to norm 1, then average; clip merged norm 1',
                    optimizer='Preserved AdamW moments and constant LR; per-task gradient cap before accumulation',
                    matched_control=f'First {args.steps} updates of the original joint run; identical initialization and streams')
    protocol['source_hashes']['../scripts/train_integrated_balanced.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if protocol['parent_sha256'] != hashlib.sha256(args.parent.read_bytes()).hexdigest():
        p.error('Parent differs from the matched baseline.')
    torch.set_num_threads(protocol['threads'])
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        model, optimizer, rng, (streams, states), saved = load_integrated(args.out/'last.pt', protocol)
        start, history, totals = saved['step'], saved['history'], saved['totals']
    else:
        if (args.out/'initial.pt').exists(): p.error('Output exists; use --resume or a new directory.')
        model, optimizer, rng, _ = load_checkpoint(args.parent)
        streams = make_streams(protocol); states = {name: None for name in streams}
        progress = evaluate_capabilities(model, protocol['validation_count'], protocol['seed'])
        start = 0; history = [{'step': 0, 'validation': progress}]
        totals = {key: 0 for key in ('expert_executions', 'source_occurrences', 'external_events',
                                     'composition_episodes', 'correction_final_queries', 'views_final_queries',
                                     'correction_episodes_started', 'views_episodes_started',
                                     'correction_supervised', 'views_supervised')}
        totals.update(training_seconds=0., best_key=list(selection_key(progress)), best_step=0,
                      candidate_key=list(selection_key(progress)[2:]), candidate_step=0,
                      gate_streak=int(not progress['gate_failures']), mastered_step=None)
        for name in ('initial.pt', 'best.pt', 'best_candidate.pt'):
            save_integrated(args.out/name, model, optimizer, rng, streams, states, 0, protocol, history, totals)
        print(json.dumps({'step': 0, 'scores': progress['scores']}), flush=True)
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
        if step == start+1 or step % 25 == 0: print(json.dumps(row), flush=True)
        if step % protocol['validate_every'] == 0 or step == args.steps:
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
            print(json.dumps({'step': step, 'scores': progress['scores'], 'failures': progress['gate_failures'],
                              'best_step': totals['best_step']}), flush=True)
        if step % protocol['save_every'] == 0 or step == stop:
            save_integrated(args.out/'last.pt', model, optimizer, rng, streams, states, step, protocol, history, totals)
            (args.out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    (args.out/'summary.json').write_text(json.dumps({'completed_step': stop, **totals}, indent=2)+'\n')


if __name__ == '__main__': main()
