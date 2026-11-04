"""Balanced held-out view metrics and frozen source-only linear diagnostics.

Graph predictions use the ordinary label-free forward pass. Source readouts
are fitted separately, on training combinations only, using independent ingress
activity captured at the common description boundary. No optimizer or utility
commit is called by these evaluators.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math

import torch
from torch.nn import functional as F


FACTORS = ('shared', 'left', 'right')
VALUES = tuple(b'ABCD')
CONTROLS = ('all', 'a_only', 'b_only', 'c_only', 'no_private', 'reset_query')
VISIBLE = {'all': FACTORS, 'carried': FACTORS, 'a_only': ('shared', 'left'), 'b_only': ('shared',),
           'c_only': ('shared', 'right'), 'no_private': ('shared',), 'reset_query': ()}
SOURCE_FACTORS = {'A': ('shared', 'left'), 'B': ('shared',), 'C': ('shared', 'right')}


@contextmanager
def evaluation_mode(model):
    """Restore every module's mode, including deliberately mixed train/eval."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


def accuracy_summary(predictions, answers):
    predictions, answers = list(predictions), list(answers)
    if len(predictions) != len(answers) or any(value not in VALUES for value in answers):
        raise ValueError('Byte answers and predictions must have equal lengths.')
    counts = Counter(answers)
    correct = sum(predicted == answer for predicted, answer in zip(predictions, answers))
    by_class = {chr(value): {'episodes': counts[value],
                            'correct': sum(p == value and a == value for p, a in zip(predictions, answers))}
                for value in VALUES}
    class_accuracy = [row['correct']/row['episodes'] for row in by_class.values() if row['episodes']]
    return {'episodes': len(answers), 'exact_correct': correct,
            'exact_accuracy': correct/len(answers) if answers else None,
            'balanced_accuracy': sum(class_accuracy)/len(class_accuracy) if class_accuracy else None,
            'class_balanced_chance': .25, 'majority_class_chance': max(counts.values())/len(answers) if answers else None,
            'by_answer': by_class}


def conditional_chance(combinations, query, control):
    """Bayes answer prior given visible factors and the declared split only.

    Identifiers and random filler are nuisance variables, excluded by design.
    This distinguishes true missing-information controls from correlations
    introduced by a restricted held-out combination set.
    """
    if query not in FACTORS or control not in VISIBLE or not combinations:
        raise ValueError('Known query, control and nonempty combination set required.')
    groups = defaultdict(Counter)
    for combination in combinations:
        if len(combination) != 3:
            raise ValueError('Each combination must contain shared, left and right values.')
        values = dict(zip(FACTORS, combination))
        groups[tuple(values[name] for name in VISIBLE[control])][values[query]] += 1
    return sum(max(counts.values()) for counts in groups.values())/len(combinations)


@dataclass(frozen=True)
class LinearReadout:
    mean: torch.Tensor
    scale: torch.Tensor
    weight: torch.Tensor
    fitted_examples: int

    def predict(self, features):
        features = features.detach().cpu().double()
        design = torch.cat(((features-self.mean)/self.scale, torch.ones(len(features), 1)), 1)
        return torch.tensor(VALUES)[(design @ self.weight).argmax(-1)]


@dataclass(frozen=True)
class SourceReadouts:
    heads: dict[str, LinearReadout]
    fit_split: str
    fit_episode_ids: tuple[str, ...]
    ridge: float
    description_boundary: str = 'last available ingress at common description end'


def load_source_readouts(path):
    """Restore schema-1 tensor-only readouts without enabling pickle objects."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    expected = {'source_readout_schema', 'heads', 'fit_split', 'fit_episode_ids', 'ridge', 'description_boundary'}
    if not isinstance(payload, dict) or set(payload) != expected or type(payload['source_readout_schema']) is not int or payload['source_readout_schema'] != 1:
        raise ValueError('Unsupported source-readout checkpoint schema.')
    identities = payload['fit_episode_ids']
    if (payload['fit_split'] != 'train' or not isinstance(identities, (list, tuple)) or not identities
            or any(not isinstance(identity, str) or not identity for identity in identities)
            or len(set(identities)) != len(identities)):
        raise ValueError('Source readouts require distinct recorded training-fit identities.')
    ridge = payload['ridge']
    if not isinstance(ridge, (int, float)) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError('Invalid source-readout regularization.')
    boundary = payload['description_boundary']
    if boundary != SourceReadouts.__dataclass_fields__['description_boundary'].default:
        raise ValueError('Source readouts use an incompatible description boundary.')
    if not isinstance(payload['heads'], dict) or not payload['heads']:
        raise ValueError('Source-readout checkpoint contains no heads.')
    heads = {}
    for name, values in payload['heads'].items():
        parts = name.split('/') if isinstance(name, str) else []
        if (len(parts) != 3 or parts[0] not in ('ingress', 'shared', 'private')
                or parts[1] not in SOURCE_FACTORS or parts[2] not in _readout_tasks('/'.join(parts[:2]))):
            raise ValueError('Unknown source-readout identity.')
        if not isinstance(values, dict) or set(values) != {'mean', 'scale', 'weight', 'fitted_examples'}:
            raise ValueError('Malformed linear readout.')
        if type(values['fitted_examples']) is not int or values['fitted_examples'] != len(identities):
            raise ValueError('Readout sample count differs from recorded training identities.')
        mean, scale, weight = (values[key] for key in ('mean', 'scale', 'weight'))
        if any(not torch.is_tensor(value) or not value.is_floating_point()
               or value.requires_grad or not bool(torch.isfinite(value).all()) for value in (mean, scale, weight)):
            raise ValueError('Readout tensors must be finite detached floating-point values.')
        if (mean.ndim != 1 or len(mean) < 1 or scale.shape != mean.shape
                or weight.shape != (len(mean)+1, len(VALUES)) or not bool((scale > 0).all())):
            raise ValueError('Readout tensor dimensions or feature scales are invalid.')
        heads[name] = LinearReadout(mean.double(), scale.double(), weight.double(), values['fitted_examples'])
    return SourceReadouts(heads, 'train', tuple(identities), float(ridge), boundary)


def fit_linear_readout(features, labels, ridge=1e-2):
    """Deterministic closed-form fit; labels never reach the graph model."""
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError('Ridge regularization must be finite and positive.')
    x, labels = features.detach().cpu().double(), labels.detach().cpu().long()
    if x.ndim != 2 or labels.shape != (len(x),) or len(x) < 1 or not bool(torch.isfinite(x).all()):
        raise ValueError('Finite nonempty [N,D] features and matching labels required.')
    if not bool((labels[:, None] == torch.tensor(VALUES)).any(1).all()):
        raise ValueError('Readout labels must be one of the four answer bytes.')
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-6)
    design = torch.cat(((x-mean)/scale, torch.ones(len(x), 1)), 1)
    target = (labels[:, None] == torch.tensor(VALUES)).double()
    penalty = torch.eye(design.shape[1], dtype=torch.double)*ridge
    penalty[-1, -1] = 0
    weight = torch.linalg.solve(design.T @ design + penalty, design.T @ target)
    return LinearReadout(mean, scale, weight, len(x))


def _source_features(model, source_payloads):
    features = {}
    objective = getattr(model, 'view_objective', None)
    for source, payload in source_payloads.items():
        if source not in SOURCE_FACTORS:
            continue
        features[f'ingress/{source}'] = payload.detach().cpu()
        if objective is not None and source in objective.views:
            features[f'shared/{source}'] = objective.shared[source](payload).detach().cpu()
            features[f'private/{source}'] = objective.private[source](payload).detach().cpu()
    return features


def _readout_tasks(feature_name):
    kind, source = feature_name.split('/')
    return ('shared',) if kind == 'shared' else (
        tuple(name for name in SOURCE_FACTORS[source] if name != 'shared') if kind == 'private'
        else SOURCE_FACTORS[source])


def _run_batch(model, batch, segment, capture_sources=False):
    if segment < 1:
        raise ValueError('Segment length must be positive.')
    device = next(model.parameters()).device
    state, result = None, None
    source_payloads, source_available = {}, {}
    calls, events = 0, 0
    boundary = batch.description_end
    if not 0 < boundary <= batch.length:
        raise ValueError('Description boundary must lie within the episode.')
    for start in range(0, batch.length, segment):
        stop = min(start+segment, batch.length)
        chunk = batch.slice(start, stop)
        capture = capture_sources and start < boundary
        result = model({key: values.to(device) for key, values in chunk.inputs.items()}, state=state,
                       reset=chunk.reset.to(device),
                       availability={key: values.to(device) for key, values in chunk.availability.items()},
                       return_view_activity=capture)
        state = result['state']
        calls += result['expert_executions']
        events += result['external_events']
        if capture:
            end = min(stop, boundary)-start
            for source, activity in result['view_activity'].items():
                if activity.get('provenance') != 'independent_ingress':
                    raise ValueError('Source diagnostics require independent ingress provenance.')
                payload, available = activity['payload'][:, :end], activity['available'][:, :end]
                any_available = available.any(1)
                positions = torch.arange(end, device=device)[None]
                index = torch.where(available, positions, -1).max(1).values.clamp_min(0)
                current = payload[torch.arange(len(payload), device=device), index]
                previous = source_payloads.get(source, torch.zeros_like(current))
                source_payloads[source] = torch.where(any_available[:, None], current, previous)
                source_available[source] = source_available.get(source, torch.zeros_like(any_available)) | any_available
    return result['logits'][:, -1].detach().cpu(), calls, events, source_payloads, source_available


def _collect_sources(model, payloads, availability, batch, feature_parts, label_parts, mask_parts):
    for name, values in _source_features(model, payloads).items():
        feature_parts.setdefault(name, []).append(values)
    for factor in FACTORS:
        label_parts.setdefault(factor, []).append(batch.factors[factor].detach().cpu())
    for source, mask in availability.items():
        mask_parts.setdefault(source, []).append(mask.detach().cpu())


def fit_source_readouts(model, *, count=128, batch_size=32, segment=16, seed=0,
                        gap=8, query_every=4, ridge=1e-2, split='train'):
    """Fit frozen ridge heads on a declared training-only description corpus.

    The query is fixed to shared because all labels are taken from source
    factors at the common pre-query boundary. Fitting repeats neither all three
    query types nor any held-out combination. All model parameters stay fixed.
    """
    from .view_data import make_view_batch
    if split != 'train':
        raise ValueError('Frozen source readouts may be fitted only on the train split.')
    if min(count, batch_size, segment, query_every) < 1 or count % 32 or gap < 0:
        raise ValueError('Use a positive multiple of 32 train descriptions and positive batch/segment sizes.')
    features, labels, masks, identities = {}, {}, {}, []
    with evaluation_mode(model):
        for start in range(0, count, batch_size):
            batch = make_view_batch('train', start, min(batch_size, count-start), seed=seed,
                                    gap=gap, query_every=query_every, control='all', query_type='shared')
            _, _, _, payloads, available = _run_batch(model, batch, segment, capture_sources=True)
            _collect_sources(model, payloads, available, batch, features, labels, masks)
            identities.extend(batch.identities)
        joined_labels = {key: torch.cat(parts) for key, parts in labels.items()}
        joined_masks = {key: torch.cat(parts) for key, parts in masks.items()}
        heads = {}
        for name, parts in features.items():
            values = torch.cat(parts)
            source = name.split('/')[1]
            used = joined_masks[source]
            for factor in _readout_tasks(name):
                heads[f'{name}/{factor}'] = fit_linear_readout(values[used], joined_labels[factor][used], ridge)
    return SourceReadouts(heads, 'train', tuple(identities), ridge)


def _source_metrics(model, features, labels, masks, readouts):
    from .view_data import DESCRIPTION_END
    if not features:
        return None
    features = {key: torch.cat(parts) for key, parts in features.items()}
    labels = {key: torch.cat(parts) for key, parts in labels.items()}
    masks = {key: torch.cat(parts) for key, parts in masks.items()}
    scores = {}
    if readouts is not None:
        for name, head in readouts.heads.items():
            kind, source, factor = name.split('/')
            feature_name = f'{kind}/{source}'
            if feature_name not in features or factor not in _readout_tasks(feature_name):
                raise ValueError('A frozen readout does not match the model source representations.')
            used = masks[source]
            score = accuracy_summary(head.predict(features[feature_name][used]).tolist(), labels[factor][used].tolist())
            score['fitted_examples'] = head.fitted_examples
            scores[name] = score
    trained = {}
    objective = getattr(model, 'view_objective', None)
    if objective is not None:
        device = next(objective.parameters()).device
        for source in objective.views:
            name = f'private/{source}'
            private_factors = tuple(factor for factor in SOURCE_FACTORS[source] if factor != 'shared')
            if name not in features or len(private_factors) != 1:
                continue
            used = masks[source]
            logits = objective.private_readout[source](features[name][used].to(device))
            trained[source] = accuracy_summary(logits.argmax(-1).cpu().tolist(), labels[private_factors[0]][used].tolist())
            trained[source]['objective_enabled'] = objective.cfg.private_enabled
            trained[source]['factor'] = private_factors[0]
            trained[source]['evaluation_timing'] = {
                'description_end_exclusive': DESCRIPTION_END,
                'latest_possible_event_index': DESCRIPTION_END-1,
                'source_position': 'last available ingress before the description boundary',
                'query_markers_observed': False}
            trained[source]['timing_matches_training_pool'] = False
            trained[source]['training_pooling'] = objective.cfg.pooling
            trained[source]['timing_note'] = (
                'This pre-query score uses event 13 or an earlier valid source event. '
                'The runner normally trains last-pooled private heads at event 15 after the '
                'first query markers in a 16-event chunk, and at later chunk ends. '
                'Use the frozen readouts for matched fit/evaluation timing.')
    return {'feature_boundary': 'last available independent ingress at common description end',
            'evaluation_control': 'all', 'frozen_linear': scores, 'trained_private_heads': trained,
            'fit_split': None if readouts is None else readouts.fit_split,
            'ridge': None if readouts is None else readouts.ridge,
            'frozen_scoring': 'argmax over four declared factor values',
            'trained_head_scoring': 'argmax over all 256 bytes'}


def evaluate_views(model, *, split='independent', count_per_query=192, batch_size=32,
                   segment=16, seed=0, gap=8, query_every=4, controls=None,
                   source_readouts=None, include_predictions=False):
    """Score balanced fresh episodes without supplying answers to the model.

    The independent all-64-combination cohort supports information controls.
    Restricted held-out cohorts use all views only; their factor correlations
    are reported separately from the information-withholding experiment.
    """
    from .view_data import make_view_batch, split_combinations
    if split not in ('validation', 'test', 'independent'):
        raise ValueError('Evaluation requires a fresh validation, test or independent cohort.')
    if min(count_per_query, batch_size, segment, query_every) < 1 or count_per_query % 64 or gap < 0:
        raise ValueError('Use a positive multiple of 64 episodes per query and positive batch/segment sizes.')
    controls = tuple((CONTROLS if split == 'independent' else ('all',)) if controls is None else controls)
    controls = tuple('all' if control == 'carried' else control for control in controls)
    if not controls or len(set(controls)) != len(controls) or any(control not in CONTROLS for control in controls):
        raise ValueError('Controls must be distinct known information interventions.')
    if split != 'independent' and any(control != 'all' for control in controls):
        raise ValueError('Information controls require the independent all-combination cohort.')
    combinations = tuple(split_combinations(split))
    if source_readouts is not None and source_readouts.fit_split != 'train':
        raise ValueError('Held-out source diagnostics require train-only fitted readouts.')
    results = {}
    features, labels, masks = {}, {}, {}
    expected_cost, expected_identities = None, None
    with evaluation_mode(model):
        for control in controls:
            by_query, all_predictions, all_answers, sample_keys = {}, [], [], []
            supported_predictions, supported_answers, absent_predictions, absent_answers = [], [], [], []
            calls = events = 0
            for query in FACTORS:
                predictions, answers, ce_sum = [], [], 0.
                for start in range(0, count_per_query, batch_size):
                    batch = make_view_batch(split, start, min(batch_size, count_per_query-start), seed=seed,
                                            gap=gap, query_every=query_every, control=control, query_type=query)
                    capture = control == 'all' and (source_readouts is not None or getattr(model, 'view_objective', None) is not None)
                    logits, cost, observations, payloads, available = _run_batch(model, batch, segment, capture)
                    calls += cost
                    events += observations
                    predictions.extend(logits.argmax(-1).tolist())
                    answers.extend(batch.answers.tolist())
                    ce_sum += float(F.cross_entropy(logits, batch.answers, reduction='sum'))
                    sample_keys.extend((query, identity) for identity in batch.identities)
                    expected_support = query in VISIBLE[control]
                    if not bool((batch.answer_available[:, -1] == expected_support).all()):
                        raise ValueError('Declared information support differs from episode evidence.')
                    if capture:
                        if source_readouts is not None and set(source_readouts.fit_episode_ids).intersection(batch.identities):
                            raise ValueError('Source readout fit/evaluation identities overlap.')
                        _collect_sources(model, payloads, available, batch, features, labels, masks)
                score = accuracy_summary(predictions, answers)
                if any(row['episodes'] != count_per_query//4 for row in score['by_answer'].values()):
                    raise ValueError('Evaluation answers must be exactly balanced within every query type.')
                score.update(information_present=query in VISIBLE[control],
                             conditional_split_prior_chance=conditional_chance(combinations, query, control),
                             answer_nats_per_byte=ce_sum/count_per_query)
                by_query[query] = score
                all_predictions.extend(predictions)
                all_answers.extend(answers)
                if query in VISIBLE[control]:
                    supported_predictions.extend(predictions)
                    supported_answers.extend(answers)
                else:
                    absent_predictions.extend(predictions)
                    absent_answers.extend(answers)
            if expected_cost is None:
                expected_cost, expected_identities = calls, sample_keys
            elif calls != expected_cost or sample_keys != expected_identities:
                raise ValueError('Information controls must retain identical episodes and expert execution counts.')
            score = accuracy_summary(all_predictions, all_answers)
            score.update(by_query=by_query, macro_query_accuracy=sum(row['exact_accuracy'] for row in by_query.values())/3,
                         supported=accuracy_summary(supported_predictions, supported_answers),
                         absent_information=accuracy_summary(absent_predictions, absent_answers),
                         expert_executions=calls, external_events=events,
                         episode_fingerprint=sha256(json.dumps(sample_keys).encode()).hexdigest())
            absent = [row['conditional_split_prior_chance'] for row in by_query.values() if not row['information_present']]
            score['absent_information']['conditional_split_prior_chance'] = sum(absent)/len(absent) if absent else None
            if include_predictions:
                score.update(predictions=all_predictions, answers=all_answers, episode_keys=sample_keys)
            results[control] = score
        sources = _source_metrics(model, features, labels, masks, source_readouts)
    return {'split': split, 'cohort': 'independent_information_controls' if split == 'independent' else 'heldout_factor_combinations',
            'seed': seed, 'gap': gap, 'query_every': query_every, 'count_per_query': count_per_query,
            'factor_combinations': len(combinations), 'answer_vocabulary': 'ABCD',
            'scoring': 'argmax over all 256 bytes at final query; classes and query types exactly balanced',
            'controls': results, 'source_readouts': sources}
