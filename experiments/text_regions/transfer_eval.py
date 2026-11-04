"""Locate fact decodability in frozen regional publications before/after queries.

Readout fits use training combinations and the same information control as the
evaluation. Features contain publication payloads only. Labels and query types
are used outside inference for fitting and reporting. These linear diagnostics
measure decodability, not whether the model itself uses each decoded fact.
"""
import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .view_data import make_view_batch, split_combinations
from .view_eval import (FACTORS, SOURCE_FACTORS, VISIBLE, LinearReadout, accuracy_summary,
                        evaluation_mode, fit_linear_readout)
from .view_train import tensor_fingerprint


CONTROLS = ('all', 'a_only', 'b_only', 'c_only')
MOMENTS = ('prequery', 'final')


@dataclass(frozen=True)
class TransferReadouts:
    heads: dict[str, LinearReadout]
    fits: dict[str, dict]
    fit_split: str
    fit_episode_ids: tuple[str, ...]
    nodes: tuple[str, ...]
    controls: tuple[str, ...]
    fit_count_per_control: int
    seed: int
    gap: int
    query_every: int
    ridge: float
    model_tensor_sha256: str
    inference_policy: dict
    fit_expert_executions: dict[str, int]


def _nodes(model):
    return tuple(node.name for node in model.cfg.nodes)


def _key(control, moment, node, factor):
    return '|'.join((control, moment, node, factor))


def _tasks(control, node):
    supported = VISIBLE[control]
    return tuple(factor for factor in supported if not node.startswith('@') or factor in SOURCE_FACTORS[node[1:]])


def _policy(model):
    utility = getattr(model, 'utility', None)
    return {'evidence_gated_ports': model.cfg.evidence_gated_ports,
            'graph_config': model.cfg.to_dict(), 'utility_enabled': bool(utility is not None and utility.enabled),
            'utility_scale': float(getattr(model, 'utility_scale', 0.))}


def _fit_info(labels, valid):
    counts = {chr(value): int((labels[valid] == value).sum()) for value in b'ABCD'}
    return {'episodes': len(labels), 'valid_episodes': int(valid.sum()),
            'valid_fraction': float(valid.float().mean()), 'by_answer': counts,
            'status': 'fitted' if all(counts.values()) else 'insufficient_valid_classes'}


def capture_transfer(model, batch, segment=16):
    """Run one fresh episode batch and copy actual port values at two cuts.

    Splitting a chunk at the description boundary changes neither the observed
    bytes nor expert work. Invalid publications are zeroed in diagnostic copies
    and excluded by validity masks; no invalid carried payload counts as evidence.
    """
    if segment < 1 or batch.time_offset != 0 or not 0 < batch.description_end < batch.length:
        raise ValueError('Capture requires complete fresh episodes and a positive segment length.')
    nodes = _nodes(model)
    device = next(model.parameters()).device
    cuts = sorted(set(range(segment, batch.length, segment)) | {batch.description_end, batch.length})
    snapshots, state, start, calls, events = {}, None, 0, 0, 0
    with evaluation_mode(model):
        for stop in cuts:
            chunk = batch.slice(start, stop)
            result = model({key: value.to(device) for key, value in chunk.inputs.items()}, state=state,
                           reset=chunk.reset.to(device),
                           availability={key: value.to(device) for key, value in chunk.availability.items()})
            state = result['state']
            calls += result['expert_executions']
            events += result['external_events']
            if stop in (batch.description_end, batch.length):
                moment = 'prequery' if stop == batch.description_end else 'final'
                names = nodes + (tuple('@'+view for view in SOURCE_FACTORS) if moment == 'final' else ())
                snapshots[moment] = {}
                for name in names:
                    port = state.ports.latest[name]
                    valid = port.valid.detach().cpu().clone()
                    values = port.payload.detach().cpu().clone()
                    if valid.dtype != torch.bool or valid.shape != values.shape[:1] or values.ndim != 2:
                        raise ValueError('Expected batched latent publications and boolean validity.')
                    values = torch.where(valid[:, None], values, torch.zeros_like(values))
                    if not bool(torch.isfinite(values).all()):
                        raise ValueError('Nonfinite valid publication in transfer diagnostic.')
                    snapshots[moment][name] = {'payload': values, 'valid': valid}
            start = stop
    return {'logits': result['logits'][:, -1].detach().cpu(), 'snapshots': snapshots,
            'expert_executions': calls, 'external_events': events}


def _append_capture(parts, capture):
    for moment, nodes in capture['snapshots'].items():
        for node, value in nodes.items():
            row = parts.setdefault((moment, node), {'payload': [], 'valid': []})
            row['payload'].append(value['payload'])
            row['valid'].append(value['valid'])


def fit_transfer_readouts(model, *, count=192, batch_size=32, segment=16, seed=0,
                          gap=8, query_every=4, controls=CONTROLS, ridge=.01, split='train'):
    """Fit each control independently on balanced, training-only descriptions."""
    controls = tuple(controls)
    if split != 'train' or count < 96 or count % 96 or min(batch_size, segment, query_every) < 1 or gap < 0:
        raise ValueError('Use training data, a positive multiple of 96 episodes/control and valid dimensions.')
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError('Ridge regularization must be finite and positive.')
    if not controls or len(set(controls)) != len(controls) or any(control not in CONTROLS for control in controls):
        raise ValueError('Distinct supported source controls required.')
    fingerprint = tensor_fingerprint(model)
    policy = _policy(model)
    heads, fits, identities, costs = {}, {}, [], {}
    with evaluation_mode(model):
        for control in controls:
            parts, labels, queries, calls = {}, {factor: [] for factor in FACTORS}, [], 0
            per_query = count//3
            for query in FACTORS:
                for start in range(0, per_query, batch_size):
                    batch = make_view_batch('train', start, min(batch_size, per_query-start), seed=seed,
                                            gap=gap, query_every=query_every, control=control, query_type=query)
                    captured = capture_transfer(model, batch, segment)
                    calls += captured['expert_executions']
                    _append_capture(parts, captured)
                    for factor in FACTORS:
                        labels[factor].append(batch.factors[factor])
                    identities.extend(batch.identities)
                    queries.extend(batch.query_types)
            costs[control] = calls
            labels = {factor: torch.cat(values) for factor, values in labels.items()}
            for (moment, node), values in parts.items():
                features, valid = torch.cat(values['payload']), torch.cat(values['valid'])
                for factor in _tasks(control, node):
                    key = _key(control, moment, node, factor)
                    fits[key] = _fit_info(labels[factor], valid)
                    if fits[key]['status'] == 'fitted':
                        heads[key] = fit_linear_readout(features[valid], labels[factor][valid], ridge)
                    query_mask = torch.tensor([query == factor for query in queries])
                    matched = valid & query_mask
                    fits[key]['query_matched_fit'] = _fit_info(labels[factor][query_mask], valid[query_mask])
                    if fits[key]['query_matched_fit']['status'] == 'fitted':
                        heads[key+'|matched'] = fit_linear_readout(features[matched], labels[factor][matched], ridge)
    if len(set(costs.values())) != 1:
        raise ValueError('Readout-fit controls changed expert execution counts.')
    if tensor_fingerprint(model) != fingerprint or _policy(model) != policy:
        raise RuntimeError('Readout fitting mutated model tensors or inference policy.')
    return TransferReadouts(heads, fits, 'train', tuple(dict.fromkeys(identities)), _nodes(model), controls,
                            count, seed, gap, query_every, ridge, fingerprint, policy, costs)


def _scored_subset(predictions, labels, valid, selection):
    used = valid & selection
    result = accuracy_summary(predictions[used].tolist(), labels[used].tolist())
    result.update(total_episodes=int(selection.sum()), valid_episodes=int(used.sum()),
                  valid_fraction=float(used.sum()/selection.sum()) if bool(selection.any()) else None)
    return result


def evaluate_transfer(model, readouts, *, split='independent', count_per_query=192,
                      batch_size=32, segment=16, seed=0, gap=8, query_every=4, controls=None):
    """Score publications on fresh held-out tuples or independent controls."""
    if split not in ('validation', 'test', 'independent') or count_per_query < 64 or count_per_query % 64:
        raise ValueError('Use a held-out cohort and a positive multiple of 64 episodes per query.')
    if min(batch_size, segment, query_every) < 1 or gap < 0:
        raise ValueError('Invalid evaluation dimensions.')
    controls = tuple((CONTROLS if split == 'independent' else ('all',)) if controls is None else controls)
    if (not controls or len(set(controls)) != len(controls) or any(control not in readouts.controls for control in controls)
            or (split != 'independent' and controls != ('all',))):
        raise ValueError('Use fitted matching controls; restricted combinations permit all views only.')
    fingerprint = tensor_fingerprint(model)
    if (readouts.fit_split != 'train' or readouts.nodes != _nodes(model)
            or readouts.model_tensor_sha256 != fingerprint or readouts.inference_policy != _policy(model)):
        raise ValueError('Transfer readouts must match this frozen model and training-only fit.')
    fit_ids = set(readouts.fit_episode_ids)
    results, expected_ids, expected_calls = {}, None, None
    with evaluation_mode(model):
        for control in controls:
            parts, factor_labels, query_types, identities = {}, {factor: [] for factor in FACTORS}, [], []
            predictions, answers, calls, events, ce_sum = [], [], 0, 0, 0.
            for query in FACTORS:
                for start in range(0, count_per_query, batch_size):
                    batch = make_view_batch(split, start, min(batch_size, count_per_query-start), seed=seed,
                                            gap=gap, query_every=query_every, control=control, query_type=query)
                    if fit_ids.intersection(batch.identities):
                        raise ValueError('Transfer readout fit/evaluation identities overlap.')
                    captured = capture_transfer(model, batch, segment)
                    _append_capture(parts, captured)
                    for factor in FACTORS:
                        factor_labels[factor].append(batch.factors[factor])
                    predictions.extend(captured['logits'].argmax(-1).tolist())
                    answers.extend(batch.answers.tolist())
                    ce_sum += float(F.cross_entropy(captured['logits'], batch.answers, reduction='sum'))
                    query_types.extend(batch.query_types)
                    identities.extend(batch.identities)
                    calls += captured['expert_executions']
                    events += captured['external_events']
            if expected_ids is None:
                expected_ids, expected_calls = identities, calls
            elif expected_ids != identities or expected_calls != calls:
                raise ValueError('Controls changed evaluation identities or expert execution counts.')
            factors = {factor: torch.cat(values) for factor, values in factor_labels.items()}
            direct = accuracy_summary(predictions, answers)
            direct['answer_nats_per_byte'] = ce_sum/len(answers)
            direct['by_query'] = {}
            for factor in FACTORS:
                selected = [index for index, query in enumerate(query_types) if query == factor]
                score = accuracy_summary([predictions[index] for index in selected], [answers[index] for index in selected])
                score['information_present'] = factor in VISIBLE[control]
                direct['by_query'][factor] = score
            nodes = {}
            for (moment, node), values in parts.items():
                features, valid = torch.cat(values['payload']), torch.cat(values['valid'])
                tasks = {}
                for factor in _tasks(control, node):
                    key = _key(control, moment, node, factor)
                    record = {'fit': readouts.fits[key], 'globally_supported': True}
                    head = readouts.heads.get(key)
                    query_mask = torch.tensor([query == factor for query in query_types])
                    if head is not None:
                        decoded = head.predict(features)
                        record.update(pooled=_scored_subset(decoded, factors[factor], valid, torch.ones_like(valid)),
                                      query_matched=_scored_subset(decoded, factors[factor], valid, query_mask))
                    else:
                        record.update(pooled=None, query_matched=None)
                    matched_head = readouts.heads.get(key+'|matched')
                    record['query_matched_fit'] = (None if matched_head is None else
                        _scored_subset(matched_head.predict(features), factors[factor], valid, query_mask))
                    tasks[factor] = record
                nodes.setdefault(moment, {})[node] = {'valid_episodes': int(valid.sum()),
                                                      'episodes': len(valid), 'valid_fraction': float(valid.float().mean()),
                                                      'factors': tasks}
            results[control] = {'direct_output': direct, 'publications': nodes, 'expert_executions': calls,
                                'external_events': events, 'episode_ids': identities}
    unchanged = tensor_fingerprint(model) == fingerprint and _policy(model) == readouts.inference_policy
    if not unchanged:
        raise RuntimeError('Transfer evaluation mutated model tensors.')
    return {'split': split, 'cohort': 'independent_information_controls' if split == 'independent' else 'heldout_factor_combinations',
            'factor_combinations': len(split_combinations(split)), 'count_per_query': count_per_query,
            'seed': seed, 'gap': gap, 'query_every': query_every, 'controls': results,
            'model_tensor_sha256': fingerprint, 'model_tensors_unchanged': unchanged,
            'inference_policy': readouts.inference_policy,
            'fit_split': readouts.fit_split, 'fit_count_per_control': readouts.fit_count_per_control,
            'fit_protocol': {'seed': readouts.seed, 'gap': readouts.gap, 'query_every': readouts.query_every},
            'fit_definitions': {'pooled': 'head fitted and scored across all query types',
                                'query_matched': 'pooled-fit head scored only when the queried factor matches',
                                'query_matched_fit': 'separate head fitted and scored only when the queried factor matches'},
            'feature_definition': 'actual node publication payload only; invalid publications excluded',
            'moments': {'prequery': 'after description event 13, before the first query marker',
                        'final': 'after the final query; includes observed query bytes, not query metadata'},
            'ingress_final': 'source @A/@B/@C own-factor heads; compare with independent pre-query source diagnostics',
            'interpretation': 'frozen linear decodability; does not imply the model uses the decoded fact'}


def load_transfer_readouts(path):
    """Validate and reconstruct a tensor-only training-fit diagnostic artifact."""
    from .config import GraphConfig
    data = torch.load(path, map_location='cpu', weights_only=True)
    fields = set(TransferReadouts.__dataclass_fields__)
    if (not isinstance(data, dict) or set(data) != fields | {'transfer_readout_schema'}
            or type(data['transfer_readout_schema']) is not int or data['transfer_readout_schema'] != 1
            or data['fit_split'] != 'train'):
        raise ValueError('Unsupported training-only transfer readout artifact.')
    data.pop('transfer_readout_schema')
    count = data['fit_count_per_control']
    ids, nodes, controls = data['fit_episode_ids'], data['nodes'], data['controls']
    if type(count) is not int or count < 96 or count % 96:
        raise ValueError('Invalid transfer training-fit size.')
    for values, label in ((ids, 'identities'), (nodes, 'nodes'), (controls, 'controls')):
        if (not isinstance(values, (tuple, list)) or not values or
                any(not isinstance(value, str) or not value for value in values) or len(set(values)) != len(values)):
            raise ValueError('Distinct nonempty transfer '+label+' required.')
    if len(ids) != count or any(not node.isidentifier() for node in nodes) or any(control not in CONTROLS for control in controls):
        raise ValueError('Transfer fit identities or topology are inconsistent.')
    if (type(data['seed']) is not int or type(data['gap']) is not int or data['gap'] < 0
            or type(data['query_every']) is not int or data['query_every'] < 1
            or not isinstance(data['ridge'], (int, float)) or not math.isfinite(data['ridge']) or data['ridge'] <= 0):
        raise ValueError('Invalid transfer fit protocol.')
    digest = data['model_tensor_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest):
        raise ValueError('Invalid model tensor identity.')
    policy = data['inference_policy']
    if (not isinstance(policy, dict) or set(policy) != {'evidence_gated_ports', 'graph_config', 'utility_enabled', 'utility_scale'}
            or type(policy['evidence_gated_ports']) is not bool or type(policy['utility_enabled']) is not bool
            or not isinstance(policy['utility_scale'], (int, float)) or not math.isfinite(policy['utility_scale'])
            or policy['utility_scale'] < 0 or not isinstance(policy['graph_config'], dict)):
        raise ValueError('Invalid transfer inference policy.')
    try:
        config = GraphConfig.from_dict(policy['graph_config'])
        config.validate()
    except (TypeError, KeyError, ValueError, AssertionError) as error:
        raise ValueError('Invalid transfer graph configuration.') from error
    if tuple(node.name for node in config.nodes) != tuple(nodes) or config.evidence_gated_ports != policy['evidence_gated_ports']:
        raise ValueError('Transfer inference policy differs from recorded topology.')
    costs = data['fit_expert_executions']
    if (not isinstance(costs, dict) or set(costs) != set(controls)
            or any(type(cost) is not int or cost < 1 for cost in costs.values()) or len(set(costs.values())) != 1):
        raise ValueError('Invalid or unequal readout-fit expert counts.')
    expected = {_key(control, moment, node, factor)
                for control in controls for moment in MOMENTS
                for node in (tuple(nodes)+(tuple('@'+view for view in SOURCE_FACTORS) if moment == 'final' else ()))
                for factor in _tasks(control, node)}
    if not isinstance(data['fits'], dict) or set(data['fits']) != expected or not isinstance(data['heads'], dict):
        raise ValueError('Transfer head/fit identities do not match the recorded controls and nodes.')

    def validate_fit(info, episodes, matched=False):
        keys = {'episodes', 'valid_episodes', 'valid_fraction', 'by_answer', 'status'}
        if not matched:
            keys.add('query_matched_fit')
        if not isinstance(info, dict) or set(info) != keys or info['episodes'] != episodes:
            raise ValueError('Malformed transfer fit metadata.')
        valid, counts = info['valid_episodes'], info['by_answer']
        if (type(valid) is not int or not 0 <= valid <= episodes or not isinstance(counts, dict)
                or set(counts) != set('ABCD') or any(type(value) is not int or value < 0 for value in counts.values())
                or sum(counts.values()) != valid or not isinstance(info['valid_fraction'], (int, float))
                or not math.isclose(info['valid_fraction'], valid/episodes, abs_tol=1e-7)):
            raise ValueError('Inconsistent transfer fit class counts or validity.')
        status = 'fitted' if all(counts.values()) else 'insufficient_valid_classes'
        if info['status'] != status:
            raise ValueError('Transfer fit status differs from its observed classes.')
        return status == 'fitted'

    expected_heads = {}
    for key, fit in data['fits'].items():
        if validate_fit(fit, count):
            expected_heads[key] = fit['valid_episodes']
        if validate_fit(fit['query_matched_fit'], count//3, True):
            expected_heads[key+'|matched'] = fit['query_matched_fit']['valid_episodes']
    if set(data['heads']) != set(expected_heads):
        raise ValueError('Transfer head tensors do not match successful fits.')
    heads = {}
    for key, values in data['heads'].items():
        if not isinstance(values, dict) or set(values) != {'mean', 'scale', 'weight', 'fitted_examples'}:
            raise ValueError('Malformed transfer linear head.')
        if type(values['fitted_examples']) is not int or values['fitted_examples'] != expected_heads[key]:
            raise ValueError('Transfer head fit count differs from metadata.')
        mean, scale, weight = (values[name] for name in ('mean', 'scale', 'weight'))
        if any(not torch.is_tensor(value) or not value.is_floating_point() or value.requires_grad
               or not bool(torch.isfinite(value).all()) for value in (mean, scale, weight)):
            raise ValueError('Transfer head tensors must be finite detached floating-point values.')
        if (mean.ndim != 1 or len(mean) < 1 or scale.shape != mean.shape or weight.shape != (len(mean)+1, 4)
                or not bool((scale > 0).all())):
            raise ValueError('Invalid transfer head dimensions or scales.')
        heads[key] = LinearReadout(mean.double(), scale.double(), weight.double(), values['fitted_examples'])
    data.update(heads=heads, fit_episode_ids=tuple(ids), nodes=tuple(nodes), controls=tuple(controls))
    return TransferReadouts(**data)


def main():
    from .learning import RegionLearningModel
    from .view_score import apply_publication_policy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fit-count', type=int, default=192)
    parser.add_argument('--count-per-query', type=int, default=192)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--segment', type=int, default=16)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--gap', type=int)
    parser.add_argument('--ridge', type=float, default=.01)
    parser.add_argument('--publication-policy', choices=('trained', 'original', 'evidence'), default='trained')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('Thread count must be positive.')
    torch.set_num_threads(args.threads)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if saved.get('view_checkpoint_schema') != 1:
        parser.error('A view-training learner checkpoint is required.')
    model = RegionLearningModel.from_config(saved['learner_config'])
    model.load_state_dict(saved['weights'])
    apply_publication_policy(model, args.publication_policy)
    protocol = saved['protocol']
    seed = protocol['seed'] if args.seed is None else args.seed
    gap = protocol['gap'] if args.gap is None else args.gap
    options = dict(batch_size=args.batch, segment=args.segment, seed=seed, gap=gap,
                   query_every=protocol['query_every'])
    readouts = fit_transfer_readouts(model, count=args.fit_count, ridge=args.ridge, **options)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({'transfer_readout_schema': 1, **asdict(readouts)}, args.out/'transfer_readouts.pt')
    result = {'checkpoint_step': saved['step'], 'training_protocol': protocol,
              'inference_policy': _policy(model), 'publication_policy_requested': args.publication_policy,
              'readout_fit_expert_executions': readouts.fit_expert_executions, 'evaluations': {}}
    for split in ('test', 'independent'):
        result['evaluations'][split] = evaluate_transfer(model, readouts, split=split,
                                                        count_per_query=args.count_per_query, **options)
        (args.out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'split': split, 'model_tensors_unchanged': True,
                          'direct_accuracy': {control: value['direct_output']['exact_accuracy']
                                              for control, value in result['evaluations'][split]['controls'].items()}}), flush=True)


if __name__ == '__main__':
    main()
