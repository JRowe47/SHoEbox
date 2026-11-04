"""Verify paired baseline-feature gradient ablations from saved predictions.

Each pair declares one publication policy used for both training and inference.
The original and detached runs may differ only in baseline_detach_features.
No checkpoints are loaded and no predictions are generated.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path

from .evidence_compare import (COUNT_KEYS, QUERIES, VISIBLE, InputFiles, _contrast,
                               _load_evaluation, _load_run, _require,
                               _training_protocol, _variation, source_coverage)


DETACH = 'baseline_detach_features'
OBSERVED = 'baseline_observed_only'
BASELINE_FLAGS = (DETACH, OBSERVED)
POLICIES = {'original': False, 'gated': True}


class BaselineFiles(InputFiles):
    """Normalize legacy defaults in memory while hashing the exact input bytes."""
    def json(self, path):
        value = super().json(path)
        protocol = value if Path(path).name == 'protocol.json' else value.get('protocol')
        if isinstance(protocol, dict):
            for flag in BASELINE_FLAGS:
                protocol.setdefault(flag, False)
                if 'learner_config' in protocol:
                    protocol['learner_config'].setdefault(flag, False)
        return value


def _protocol_signature(protocol, remove_seed=False, flag=DETACH):
    result = _training_protocol(protocol, remove_seed=remove_seed)
    result.pop(flag)
    return result


def _learner_signature(config, remove_seed=False, flag=DETACH):
    result = deepcopy(config)
    result.pop(flag)
    result['model'].setdefault('evidence_gated_ports', False)
    if remove_seed: result['model'].pop('seed', None)
    return result


def _baseline_run(spec, policy, detached, files, label, observed=None):
    run = _load_run(spec, policy, files, label)
    protocol, summary = run['protocol'], run['summary']
    config = protocol['learner_config']
    _require(type(protocol[DETACH]) is bool and protocol[DETACH] == detached,
             f'{label}: wrong baseline detach flag')
    _require(type(config[DETACH]) is bool and config[DETACH] == detached,
             f'{label}: protocol/config baseline detach flags differ')
    _require(type(protocol[OBSERVED]) is bool and (observed is None or protocol[OBSERVED] == observed),
             f'{label}: wrong baseline observed-only flag')
    _require(type(config[OBSERVED]) is bool and config[OBSERVED] == protocol[OBSERVED],
             f'{label}: protocol/config baseline observed-only flags differ')
    _require(type(config['model'].get('evidence_gated_ports', False)) is bool,
             f'{label}: invalid model publication policy')
    utility_enabled = config.get('utility', {}).get('enabled')
    _require(type(utility_enabled) is bool and utility_enabled == protocol['utility'],
             f'{label}: protocol/config utility flags differ')
    view_config = config.get('view_objective', {}).get('cfg', {})
    for term in ('agreement', 'prediction', 'variance', 'covariance', 'private', 'separation'):
        enabled = view_config.get(term+'_enabled')
        _require(type(enabled) is bool and enabled == protocol['view_objective'],
                 f'{label}: protocol/config view objective flags differ for {term}')
    _require(config.get('baseline_enabled') is True, f'{label}: baseline head must be enabled')
    weight = config.get('baseline_weight')
    _require(type(weight) in (int, float) and math.isfinite(weight) and weight > 0,
             f'{label}: baseline loss must have positive finite weight')
    counts = summary['training']
    expected_calls = counts['external_events']*protocol['rounds']*summary['parameters']['expert_calls_per_round_per_external_event']
    _require(counts['expert_executions'] == expected_calls, f'{label}: training expert budget is inconsistent')
    for query in QUERIES:
        count = counts['by_query'][query]['count']
        _require(type(count) is int and count >= 0, f'{label}: invalid training query count')
    return run


def _private_metrics(evaluation):
    """Add equally weighted left/right strata using the already verified cases."""
    for cohort, controls in evaluation['public']['cohorts'].items():
        for control, score in controls.items():
            for stratum, queries in (
                    ('private', ('left', 'right')),
                    ('supported_private', tuple(q for q in ('left', 'right') if q in VISIBLE[control])),
                    ('absent_private', tuple(q for q in ('left', 'right') if q not in VISIBLE[control]))):
                hits = [hit for query in queries for hit in evaluation['vectors'][f'{cohort}/{control}/{query}']]
                count, correct = len(hits), sum(hits)
                accuracy = correct/count if count else None
                key = f'{cohort}/{control}/{stratum}'
                evaluation['vectors'][key] = hits
                evaluation['public']['metrics'][key] = accuracy
                score[stratum] = {'episodes': count, 'exact_correct': correct,
                                   'exact_accuracy': accuracy, 'balanced_accuracy': accuracy}
    return evaluation


def _match_runs(original, changed, flag, name):
    a, b = original['protocol'], changed['protocol']
    detail = 'baseline detach' if flag == DETACH else 'baseline observability'
    _require(a['initial_tensor_sha256'] == b['initial_tensor_sha256'], f'{name}: matched initial tensor fingerprints differ')
    _require(_protocol_signature(a, flag=flag) == _protocol_signature(b, flag=flag), f'{name}: training protocols differ beyond {detail}')
    _require(_learner_signature(a['learner_config'], flag=flag) == _learner_signature(b['learner_config'], flag=flag),
             f'{name}: learner configurations differ beyond {detail}')
    _require(a['stream'] == b['stream'], f'{name}: source streams differ')
    _require(original['summary']['parameters'] == changed['summary']['parameters'], f'{name}: parameter counts differ')
    counts = {key: original['summary']['training'][key] for key in COUNT_KEYS}
    _require(counts == {key: changed['summary']['training'][key] for key in COUNT_KEYS},
             f'{name}: training exposure/execution counts differ')
    query_counts = {q: original['summary']['training']['by_query'][q]['count'] for q in QUERIES}
    _require(query_counts == {q: changed['summary']['training']['by_query'][q]['count'] for q in QUERIES},
             f'{name}: training query exposure differs')
    for key in ('state_bytes', 'utility_updates'):
        _require(original['summary'].get(key) == changed['summary'].get(key), f'{name}: {key} differs')
    return counts, query_counts


def count_baseline_supervision(chunks, baseline_view='A', observed_only=True):
    """Independent causal audit from source events, never the learner's mask.

    Factor identity, source availability and reset order determine knowledge.
    Equal-valued facts in a different factor do not reveal the queried factor.
    The state persists across truncated chunks and resets before each event.
    """
    import torch
    fields = ('eligible', 'observed', 'used', 'suppressed')
    result = {key: 0 for key in fields}
    result.update(by_query={query: {key: 0 for key in fields} for query in QUERIES},
                  recorded_segments=0, unrecorded_segments=0)
    known = None
    for chunk in chunks:
        available = chunk.availability[baseline_view]
        if known is None: known = {q: torch.zeros(available.shape[0], dtype=torch.bool) for q in QUERIES}
        _require(len(known['shared']) == available.shape[0], 'Baseline audit stream batch size changed')
        query_masks = {q: torch.tensor([query == q for query in chunk.query_types]) for q in QUERIES}
        for time in range(chunk.length):
            for query in QUERIES:
                known[query] &= ~chunk.reset[:, time]
                known[query] |= available[:, time] & (chunk.fact_positions[baseline_view][query] == chunk.time_offset+time)
            eligible = (chunk.targets[:, time] != -100) & available[:, time]
            observed = eligible & torch.stack([known[q] & query_masks[q] for q in QUERIES]).any(0)
            used = observed if observed_only else eligible
            masks = {'eligible': eligible, 'observed': observed, 'used': used, 'suppressed': eligible & ~used}
            for key, mask in masks.items():
                result[key] += int(mask.sum())
                for query in QUERIES: result['by_query'][query][key] += int((mask & query_masks[query]).sum())
        result['recorded_segments'] += 1
    return result


def _audit_supervision(run, counts, cache, required=False):
    from .view_data import ViewStream, VIEWS
    protocol = run['protocol']; view = protocol['learner_config'].get('baseline_view')
    _require(view in VIEWS, f"{run['label']}: baseline audit requires a declared source view")
    key = json.dumps({'stream': protocol['stream'], 'segment': protocol['segment'], 'counts': counts, 'view': view}, sort_keys=True)
    if key not in cache:
        stream = ViewStream(**protocol['stream'])
        cache[key] = count_baseline_supervision((stream.next_chunk(protocol['segment']) for _ in range(counts['segments'])), view)
    expected = deepcopy(cache[key])
    if not protocol[OBSERVED]:
        for row in [expected, *expected['by_query'].values()]:
            row['used'], row['suppressed'] = row['eligible'], 0
    recorded = run['summary']['training'].get('baseline_supervision')
    if recorded is None:
        _require(not required, f"{run['label']}: missing recorded baseline supervision audit")
        return {'baseline_view': view, 'observed_only': protocol[OBSERVED], 'recording': 'legacy_reconstructed_only',
                'expected': expected, 'recorded': None}
    fields = ('eligible', 'observed', 'used', 'suppressed')
    _require(type(recorded.get('recorded_segments')) is int and type(recorded.get('unrecorded_segments')) is int and
             recorded['recorded_segments'] == counts['segments'] and recorded['unrecorded_segments'] == 0,
             f"{run['label']}: incomplete baseline supervision audit")
    _require(set(recorded.get('by_query', {})) == set(QUERIES), f"{run['label']}: incomplete baseline supervision query audit")
    for label, actual, wanted in [('total', recorded, expected),
                                  *((q, recorded['by_query'][q], expected['by_query'][q]) for q in QUERIES)]:
        for field in fields:
            _require(type(actual.get(field)) is int and actual[field] == wanted[field],
                     f"{run['label']}: baseline supervision {label}/{field} differs from causal reconstruction")
    return {'baseline_view': view, 'observed_only': protocol[OBSERVED], 'recording': 'complete_verified',
            'expected': expected, 'recorded': recorded}


def _summarize(members, changed_arm='detached'):
    metrics = members[0]['evaluations']['original']['metrics']
    arms = ('original', changed_arm) + (('detached',) if changed_arm != 'detached' and any('detached' in m['evaluations'] for m in members) else ())
    contrasts = (changed_arm+'_minus_original',) + (('detached_minus_original',) if len(arms) == 3 else ())
    result = {'publication_policy': members[0]['publication_policy'], 'auxiliary': members[0]['auxiliary'],
              'utility_mediation_possible': members[0]['auxiliary']['utility'],
              'seeds': [member['seed'] for member in members], 'n_seed_pairs': len(members),
              'aggregation': 'Unweighted per-seed mean, minimum and maximum; accuracy units are fractions. Publication policies are separate.',
              'evaluations': {arm: {metric: _variation({member['seed']: member['evaluations'][arm]['metrics'][metric]
                                                        for member in members if arm in member['evaluations']})
                                    for metric in metrics} for arm in arms},
              'minimum_supported_accuracy': {arm: _variation({member['seed']: member['evaluations'][arm]['minimum_supported_accuracy']
                                                              for member in members if arm in member['evaluations']}) for arm in arms}}
    for contrast in contrasts:
        result[contrast] = {metric: _variation({member['seed']: member[contrast][metric]['accuracy_change']
                                               for member in members if contrast in member}) for metric in metrics}
    return result


def _compare_manifest(manifest_path, count_per_query=512, include_coverage=True, changed_arm='detached'):
    """Common integrity engine for one declared baseline intervention per contrast."""
    _require(changed_arm in ('detached', 'observed'), 'Unknown baseline intervention')
    masking = changed_arm == 'observed'; flag = OBSERVED if masking else DETACH
    change_key = changed_arm+'_minus_original'; suffix = 'baseline_observability' if masking else 'baseline_detach'
    _require(type(count_per_query) is int and count_per_query > 0 and count_per_query % 64 == 0,
             'Count per query must be a positive multiple of 64')
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get('schema_version') == 1 and isinstance(manifest.get('pairs'), list) and manifest['pairs'],
             'Expected a version-1 manifest with nonempty pairs')
    base = Path(manifest.get('base_directory', '.'))
    if not base.is_absolute(): base = manifest_path.parent/base
    files = BaselineFiles(base)
    _require(json.loads(files.read(manifest_path)) == manifest, 'Manifest changed before hashing')
    pairs, names, groups, signatures, coverage_cache, audit_cache = [], set(), defaultdict(list), {}, {}, {}
    for item in manifest['pairs']:
        name = item['name']
        _require(isinstance(name, str) and name and name not in names, 'Pair names must be unique')
        names.add(name)
        publication = item['publication_policy']
        _require(publication in POLICIES, f'{name}: publication policy must be original or gated')
        policy = POLICIES[publication]
        original = _baseline_run(item['original'], policy, False, files, name+'/original', observed=False if masking else None)
        changed = _baseline_run(item[changed_arm], policy, not masking, files, name+'/'+changed_arm, observed=True if masking else None)
        a = original['protocol']
        counts, query_counts = _match_runs(original, changed, flag, name)
        runs = {'original': original, changed_arm: changed}
        if masking and 'detached' in item:
            runs['detached'] = _baseline_run(item['detached'], policy, True, files, name+'/detached context', observed=False)
            _match_runs(original, runs['detached'], DETACH, name+'/detached context')
        evaluations = {arm: _private_metrics(_load_evaluation(item[arm]['evaluation_directory'], run, policy, count_per_query, files))
                       for arm, run in runs.items()}
        reference = evaluations['original']
        for evaluation in evaluations.values():
            for key, message in (('metadata', 'evaluation protocols differ'), ('signatures', 'paired case IDs/answers differ'),
                                 ('budgets', 'evaluation budgets differ')):
                _require(reference[key] == evaluation[key], f'{name}: {message}')
        auxiliary = {'view_objective': a['view_objective'], 'utility': a['utility']}
        group = f"publication_{publication}_views_{'on' if auxiliary['view_objective'] else 'off'}_utility_{'on' if auxiliary['utility'] else 'off'}"
        signature = (_protocol_signature(a, True, flag), _learner_signature(a['learner_config'], True, flag),
                     {key: value for key, value in a['stream'].items() if key != 'seed'}, a['parameters'],
                     {cohort: {key: value for key, value in meta.items() if key != 'seed'}
                      for cohort, meta in reference['metadata'].items()})
        if group in signatures: _require(signatures[group] == signature, f'{group}: protocols differ across seeds')
        else: signatures[group] = signature
        seed = a['seed']
        _require(not any(pair['seed'] == seed for pair in groups[group]), f'{group}: duplicate seed pair')
        coverage = None
        if include_coverage:
            cache_key = json.dumps({'stream': a['stream'], 'segment': a['segment'], 'counts': counts}, sort_keys=True)
            if cache_key not in coverage_cache: coverage_cache[cache_key] = source_coverage(a, counts)
            coverage = coverage_cache[cache_key]
            _require(coverage['episodes_completed'] == sum(query_counts.values()), f'{name}: reconstructed final-query exposure differs')
        pair = {'name': name, 'seed': seed, 'publication_policy': publication, 'auxiliary': auxiliary,
                'matched': {'initial_tensor_sha256': a['initial_tensor_sha256'], 'completed_steps': original['summary']['step'],
                            'training_counts': counts, 'training_query_counts': query_counts, 'parameters': a['parameters'],
                            'same_training_and_inference_publication_policy': True,
                            'training_protocol_equal_except_'+suffix: True,
                            'learner_configuration_equal_except_'+suffix: True, 'source_streams_equal': True,
                            'paired_case_ids_answers_fingerprints': True, 'matched_evaluation_budgets': True},
                'baseline': {'enabled': True, 'weight': a['learner_config']['baseline_weight'],
                             'original_detach_features': False, changed_arm+'_detach_features': not masking},
                'evaluation_protocol': reference['metadata'],
                'evaluations': {arm: value['public'] for arm, value in evaluations.items()},
                change_key: _contrast(evaluations[changed_arm], reference), 'source_coverage': coverage}
        if masking:
            pair['baseline'].update(original_observed_only=False, observed_observed_only=True,
                                    denominator='original eligible source-present supervised events')
            pair['baseline_supervision_audit'] = {arm: _audit_supervision(run, counts, audit_cache, required=arm == 'observed')
                                                 for arm, run in runs.items()}
            if 'detached' in runs:
                pair['detached_minus_original'] = _contrast(evaluations['detached'], reference)
                pair['detached_context_note'] = 'Separately matched detach intervention with observed-only masking disabled; not the primary observability contrast.'
        pairs.append(pair); groups[group].append(pair)
    result = {'schema_version': 1, 'comparison': 'baseline_observability_ablation' if masking else 'baseline_feature_gradient_ablation',
              'count_per_query': count_per_query, 'episodes_per_control': 3*count_per_query,
              'verified_pair_count': len(pairs), 'pairs': pairs,
              'groups': {name: _summarize(members, changed_arm) for name, members in groups.items()},
              'file_manifest': sorted(files.entries.values(), key=lambda row: row['path']),
              'integrity': {'input_file_sha256_recorded': True, 'scores_recomputed_from_predictions': True,
                            'query_and_answer_balance_verified': True, 'complete_test_and_independent_controls': True,
                            'legacy_missing_baseline_detach_normalized_false': True,
                            'same_training_and_inference_publication_policy': True,
                            'source_coverage_reconstructed': include_coverage, 'improvement_required': False},
              'contrast_definition': 'Observed minus original; only source-aware baseline supervision changes.' if masking else
                                     'Detached minus original; paired cases at identical training and inference publication policy.',
              'intervention_scope': 'Supervise the baseline head and encoder only on causally observed targets, with the original eligible-event denominator and utility eligibility unchanged.' if masking else
                                    'Detach baseline feature gradients for every supervised baseline target. The baseline head still trains; all other settings are matched.'}
    if masking:
        result['integrity'].update(legacy_missing_baseline_observed_normalized_false=True,
                                   baseline_supervision_counts_causally_reconstructed=True,
                                   baseline_detach_disabled_in_primary_contrast=True)
    return result


def compare_manifest(manifest_path, count_per_query=512, include_coverage=True):
    """Backward-compatible detached-versus-original comparison entry point."""
    return _compare_manifest(manifest_path, count_per_query, include_coverage, changed_arm='detached')


def run_cli(compare_function, description, argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count-per-query', type=int, default=512)
    parser.add_argument('--skip-coverage', action='store_true', help='Skip deterministic training-stream exposure reconstruction.')
    args = parser.parse_args(argv)
    result = compare_function(args.manifest, args.count_per_query, not args.skip_coverage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix+'.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n'); temporary.replace(args.out)
    print(json.dumps({'verified_pairs': result['verified_pair_count'], 'groups': {name: group['seeds'] for name, group in result['groups'].items()},
                      'count_per_query': args.count_per_query, 'output': str(args.out)}))


def main(argv=None):
    return run_cli(compare_manifest, __doc__, argv)


if __name__ == '__main__': main()
