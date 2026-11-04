"""Test whether query-only sources interfere with facts supplied by B alone.

All conditions retain identical bytes, cases and expert execution counts. Only
A/C availability changes, at observed shared-query markers; their factual bytes
remain unavailable. No target or factor label enters the inference call.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F
from .learning import RegionLearningModel
from .view_data import MARKERS, make_view_batch
from .view_eval import accuracy_summary, evaluation_mode
from .view_score import apply_publication_policy
from .view_train import tensor_fingerprint


VARIANTS = ('b_only', 'final_query_ac', 'all_queries_ac')


def query_availability(batch, variant):
    """Add only observed query markers to an otherwise B-only shared episode.

    Marker positions come from input bytes and availability, never target masks.
    Fact-position metadata is used only to reject an invalid intervention.
    """
    if variant not in VARIANTS or batch.control != 'b_only' or batch.time_offset != 0:
        raise ValueError('Use a known variant and complete B-only episodes.')
    if any(query != 'shared' for query in batch.query_types):
        raise ValueError('This diagnostic requires shared-fact queries.')
    if any(bool(batch.availability[view].any()) for view in ('A', 'C')):
        raise ValueError('A and C must initially have no available observations.')
    marker = MARKERS['shared']
    queries = (batch.inputs['B'] == marker) & batch.availability['B']
    if not bool(queries[:, -1].all()):
        raise ValueError('Every final event must be an available shared-query marker.')
    added = queries.clone() if variant == 'all_queries_ac' else torch.zeros_like(queries)
    if variant == 'final_query_ac':
        added[:, -1] = True
    available = {view: mask.clone() for view, mask in batch.availability.items()}
    for view in ('A', 'C'):
        if bool((added & (batch.inputs[view] != marker)).any()):
            raise ValueError('An intervention may expose only actual shared-query markers.')
        available[view] |= added
        if bool((available[view] & ~queries).any()):
            raise ValueError('An intervention exposed a non-query occurrence.')
        for positions in batch.fact_positions[view].values():
            present = positions >= 0
            rows = present.nonzero().flatten()
            if bool(available[view][rows, positions[rows]].any()):
                raise ValueError('An intervention exposed a factual value.')
    return available


def _case_hash(identities):
    return hashlib.sha256(json.dumps(identities).encode()).hexdigest()


def evaluate_query_interference(model, *, policy='trained', count=512, batch_size=64,
                                seed=0, gap=8, query_every=4, include_predictions=False,
                                on_result=None):
    """Score the three paired conditions, restoring modes and publication policy."""
    if policy not in ('trained', 'original', 'evidence', 'both'):
        raise ValueError('Policy must be trained, original, evidence, or both.')
    if count < 64 or count % 64 or min(batch_size, query_every) < 1 or gap < 0:
        raise ValueError('Use a positive multiple of 64 episodes and positive batch/query interval.')
    policies = ('original', 'evidence') if policy == 'both' else (policy,)
    original_cfg, before = model.cfg, tensor_fingerprint(model)
    report = {'cohort': 'independent', 'query_type': 'shared', 'seed': seed, 'gap': gap,
              'query_every': query_every, 'count': count, 'batch_size': batch_size,
              'publication_policy_requested': policy, 'model_tensor_sha256': before,
              'trained_evidence_gated_ports': original_cfg.evidence_gated_ports,
              'scoring': 'argmax over all 256 bytes at the final shared query',
              'intervention': 'A/C receive only query markers; every A/C factual value remains unavailable',
              'results': {}}
    raw_predictions, expected_ids, expected_budget, expected_inputs = [], None, None, None
    try:
        with evaluation_mode(model):
            for selected_policy in policies:
                model.model.cfg = original_cfg
                apply_publication_policy(model, selected_policy)
                report['results'][selected_policy] = {}
                for variant in VARIANTS:
                    started = time.perf_counter()
                    identities, answers, predictions, templates = [], [], [], []
                    query_correct, query_counts = Counter(), Counter()
                    calls = sources = events = 0
                    ce = 0.
                    inputs_hash = hashlib.sha256()
                    for start in range(0, count, batch_size):
                        batch = make_view_batch('independent', start, min(batch_size, count-start), seed=seed,
                                                gap=gap, query_every=query_every, query_type='shared', control='b_only')
                        available = query_availability(batch, variant)
                        device = next(model.parameters()).device
                        for view in ('A', 'B', 'C'):
                            inputs_hash.update(bytes(batch.inputs[view].flatten().tolist()))
                        result = model({view: values.to(device) for view, values in batch.inputs.items()},
                                       reset=batch.reset.to(device),
                                       availability={view: values.to(device) for view, values in available.items()})
                        predicted = result['logits'].detach().cpu().argmax(-1)
                        final_logits = result['logits'][:, -1].detach().cpu()
                        ce += float(F.cross_entropy(final_logits, batch.answers, reduction='sum'))
                        calls += result['expert_executions'];sources += result['source_occurrences']
                        events += result['external_events']
                        query_mask = (batch.inputs['B'] == MARKERS['shared']) & batch.availability['B']
                        for index in query_mask.any(0).nonzero().flatten().tolist():
                            used = query_mask[:, index]
                            query_counts[str(index)] += int(used.sum())
                            query_correct[str(index)] += int((predicted[used, index] == batch.answers[used]).sum())
                        identities.extend(batch.identities);answers.extend(batch.answers.tolist())
                        predictions.extend(predicted[:, -1].tolist());templates.extend(batch.template_ids['B'].tolist())
                    if expected_ids is None:
                        expected_ids, expected_budget, expected_inputs = identities, (calls, events), inputs_hash.hexdigest()
                    if identities != expected_ids or (calls, events) != expected_budget or inputs_hash.hexdigest() != expected_inputs:
                        raise ValueError('Conditions changed cases, input bytes, or expert/event budgets.')
                    score = accuracy_summary(predictions, answers)
                    if any(row['episodes'] != count//4 for row in score['by_answer'].values()):
                        raise ValueError('Shared answers must be exactly balanced.')
                    confusion = defaultdict(Counter)
                    for answer, predicted in zip(answers, predictions):
                        confusion[chr(answer)][chr(predicted) if predicted in b'ABCD' else 'OTHER'] += 1
                    strata = {}
                    for template in sorted(set(templates)):
                        indices = [i for i, value in enumerate(templates) if value == template]
                        strata[str(template)] = accuracy_summary([predictions[i] for i in indices], [answers[i] for i in indices])
                    crossed = {}
                    for answer in b'ABCD':
                        for template in sorted(set(templates)):
                            indices = [i for i, value in enumerate(templates) if value == template and answers[i] == answer]
                            crossed[chr(answer)+'/'+str(template)] = accuracy_summary(
                                [predictions[i] for i in indices], [answers[i] for i in indices])
                    score.update(nats_per_answer=ce/count, by_B_template=strata, by_answer_B_template=crossed,
                                 confusion={key: dict(value) for key, value in confusion.items()},
                                 query_accuracy={index: query_correct[index]/n for index, n in query_counts.items()},
                                 expert_executions=calls, source_occurrences=sources, external_events=events,
                                 episode_sha256=_case_hash(identities), inputs_sha256=inputs_hash.hexdigest(),
                                 evidence_gated_ports=model.cfg.evidence_gated_ports, seconds=time.perf_counter()-started)
                    if tensor_fingerprint(model) != before:
                        raise RuntimeError('Query evaluation changed model parameters or utility buffers.')
                    report['results'][selected_policy][variant] = score
                    if include_predictions:
                        raw_predictions.append({'policy': selected_policy, 'variant': variant, 'episode_ids': identities,
                                                'answers': answers, 'predictions': predictions, 'B_templates': templates})
                    if on_result is not None:
                        on_result(selected_policy, variant, score)
    finally:
        model.model.cfg = original_cfg
    report['model_tensors_unchanged'] = tensor_fingerprint(model) == before
    if not report['model_tensors_unchanged']:
        raise RuntimeError('Query evaluation changed model parameters or utility buffers.')
    if include_predictions:
        report['predictions'] = raw_predictions
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=512)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--policy', '--publication-policy', dest='policy',
                        choices=('trained', 'original', 'evidence', 'both'), default='trained')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--gap', type=int)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('Thread count must be positive.')
    torch.set_num_threads(args.threads)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if saved.get('view_checkpoint_schema') != 1:
        parser.error('A view-training learner checkpoint is required.')
    model = RegionLearningModel.from_config(saved['learner_config']);model.load_state_dict(saved['weights'])
    protocol = saved['protocol']
    seed = protocol['seed'] if args.seed is None else args.seed
    gap = protocol['gap'] if args.gap is None else args.gap
    def progress(policy, variant, score):
        print(json.dumps({'policy': policy, 'variant': variant, 'accuracy': score['exact_accuracy'],
                          'expert_executions': score['expert_executions'], 'seconds': score['seconds']}), flush=True)
    report = evaluate_query_interference(model, policy=args.policy, count=args.count, batch_size=args.batch,
                                         seed=seed, gap=gap, query_every=protocol['query_every'],
                                         include_predictions=True, on_result=progress)
    predictions = report.pop('predictions')
    report.update(checkpoint=str(args.checkpoint), checkpoint_step=saved['step'], training_protocol=protocol,
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  command=['python', '-m', 'experiments.text_regions.query_interference', *sys.argv[1:]],
                  threads=args.threads, torch_version=str(torch.__version__))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'predictions.jsonl').open('w') as stream:
        for row in predictions:
            stream.write(json.dumps(row, separators=(',', ':'))+'\n')
    (args.out/'results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
