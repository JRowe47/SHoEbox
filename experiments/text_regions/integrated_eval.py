"""One-checkpoint evaluation of composition, correction and factorized facts."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .composition import evaluate as composition_evaluate
from .composition_continue import evaluate_progress, gate_failures
from .probe_train import evaluate_probe
from .view_data import make_view_batch, QUERY_TYPES
from .view_eval import evaluation_mode, accuracy_summary


@torch.no_grad()
def evaluate_views(model, split='validation', count_per_query=64, batch_size=64,
                   gap=8, segment=16, seed=20261107, controls=('all',), rounds=2):
    if min(count_per_query, batch_size, segment, rounds) < 1 or gap < 0:
        raise ValueError('Invalid view evaluation dimensions.')
    records = {}
    with evaluation_mode(model):
        for control in controls:
            records[control] = {}
            for kind in QUERY_TYPES:
                predictions, targets, identities = [], [], []
                nll = 0.; calls = sources = events = 0
                for start in range(0, count_per_query, batch_size):
                    batch = make_view_batch(split, start, min(batch_size, count_per_query-start),
                                            seed, gap, control=control, query_type=kind)
                    state = None
                    for pos in range(0, batch.length, segment):
                        chunk = batch.slice(pos, min(pos+segment, batch.length))
                        result = model(chunk.inputs, state=state, reset=chunk.reset,
                                       availability=chunk.availability, rounds=rounds, feedback=True)
                        state = result['state']
                        calls += result['expert_executions']
                        sources += result['source_occurrences']; events += result['external_events']
                    logits = result['logits'][:, -1]
                    nll += float(F.cross_entropy(logits, batch.answers, reduction='sum'))
                    predictions.extend(logits.argmax(-1).tolist()); targets.extend(batch.answers.tolist())
                    identities.extend(batch.identities)
                records[control][kind] = {
                    **accuracy_summary(predictions, targets), 'nats_per_answer': nll/len(targets),
                    'predictions': predictions, 'targets': targets, 'episode_ids': identities,
                    'expert_executions': calls, 'source_occurrences': sources, 'external_events': events}
    return records


def compact(record):
    """Keep all metrics, drop only per-case arrays for training history."""
    if isinstance(record, dict):
        return {k: compact(v) for k, v in record.items()
                if k not in ('predictions', 'targets', 'episode_ids', 'keys')}
    return record


def evaluate_capabilities(model, count=64, seed=20261107, split='validation'):
    composition = evaluate_progress(model, split)
    memory = {kind: evaluate_probe(model, kind=kind, gap=16, count=count, batch_size=64,
                                  segment=16, seed=seed, split=split, controls=('carried',))
              for kind in ('recall', 'correction')}
    views = evaluate_views(model, split, count_per_query=count, seed=seed)
    # Missing-evidence scoring uses the full factorial cohort, independently of
    # the held-out tuple split, so an absent private answer is not split-inferable.
    control_seed = seed if split == 'validation' else seed+104729
    supported = evaluate_views(model, 'independent', count_per_query=count, seed=control_seed,
                               controls=('a_only', 'b_only', 'c_only'))
    scores = {'composition': composition['metrics']['overall']['accuracy']}
    nats = [composition['nats_per_answer']]
    for kind in memory:
        row = memory[kind]['controls']['carried']
        scores[kind] = row['exact_accuracy']; nats.append(row['answer_nats_per_byte'])
    for kind, row in views['all'].items():
        scores[kind] = row['exact_accuracy']; nats.append(row['nats_per_answer'])
    for control, kinds in (('a_only', ('shared', 'left')), ('b_only', ('shared',)),
                           ('c_only', ('shared', 'right'))):
        for kind in kinds:
            row = supported[control][kind]
            scores[control+'/'+kind] = row['exact_accuracy']; nats.append(row['nats_per_answer'])
    failures = ['composition/'+key for key in gate_failures(composition)]
    failures += [key for key, score in scores.items()
                 if not math.isfinite(score) or score < .9]
    return dict(composition=composition, memory=compact(memory), views=compact(views),
                supported_sources=compact(supported), scores=scores,
                minimum_accuracy=min(scores.values()), macro_accuracy=sum(scores.values())/len(scores),
                macro_nats=sum(nats)/len(nats), gate_failures=failures,
                composition_preserved=not gate_failures(composition))


def selection_key(progress):
    """Preserve composition gates first, then improve the weakest capability."""
    return (not progress['gate_failures'], progress['composition_preserved'],
            progress['minimum_accuracy'], progress['macro_accuracy'], -progress['macro_nats'])


def main():
    from .integrated_train import load_integrated
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=192)
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args()
    if min(args.count, args.threads) < 1: p.error('Counts must be positive.')
    torch.set_num_threads(args.threads)
    model, _, _, _, saved = load_integrated(args.checkpoint)
    seed = saved['protocol']['seed']
    report = {'checkpoint_step': saved['step'], 'parameters': model.parameter_summary(),
              'test_capabilities': evaluate_capabilities(model, args.count, seed, 'test'),
              'composition_controls': composition_evaluate(model, split='test', count=None,
                                                             controls=True, rounds=(1, 2, 4)),
              'memory_controls': {str(gap): {
                  kind: evaluate_probe(model, kind=kind, gap=gap, count=args.count,
                                       batch_size=64, segment=16, seed=seed, split='test')
                  for kind in ('recall', 'correction')} for gap in (16, 32)},
              'heldout_views': evaluate_views(model, 'test', args.count, seed=seed),
              'independent_views': evaluate_views(model, 'independent', args.count, seed=seed+104729,
                                                  controls=('all', 'a_only', 'b_only', 'c_only',
                                                            'no_private', 'reset_query'))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'checkpoint_step': saved['step'],
                      'scores': report['test_capabilities']['scores'],
                      'failures': report['test_capabilities']['gate_failures']}), flush=True)


if __name__ == '__main__': main()
