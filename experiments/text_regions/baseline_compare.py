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
POLICIES = {'original': False, 'gated': True}


class BaselineFiles(InputFiles):
    """Normalize legacy defaults in memory while hashing the exact input bytes."""
    def json(self, path):
        value = super().json(path)
        protocol = value if Path(path).name == 'protocol.json' else value.get('protocol')
        if isinstance(protocol, dict):
            protocol.setdefault(DETACH, False)
            if 'learner_config' in protocol:
                protocol['learner_config'].setdefault(DETACH, False)
        return value


def _protocol_signature(protocol, remove_seed=False):
    result = _training_protocol(protocol, remove_seed=remove_seed)
    result.pop(DETACH)
    return result


def _learner_signature(config, remove_seed=False):
    result = deepcopy(config)
    result.pop(DETACH)
    result['model'].setdefault('evidence_gated_ports', False)
    if remove_seed: result['model'].pop('seed', None)
    return result


def _baseline_run(spec, policy, detached, files, label):
    run = _load_run(spec, policy, files, label)
    protocol, summary = run['protocol'], run['summary']
    config = protocol['learner_config']
    _require(type(protocol[DETACH]) is bool and protocol[DETACH] == detached,
             f'{label}: wrong baseline detach flag')
    _require(type(config[DETACH]) is bool and config[DETACH] == detached,
             f'{label}: protocol/config baseline detach flags differ')
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


def _summarize(members):
    metrics = members[0]['evaluations']['original']['metrics']
    return {'publication_policy': members[0]['publication_policy'], 'auxiliary': members[0]['auxiliary'],
            'utility_mediation_possible': members[0]['auxiliary']['utility'],
            'seeds': [member['seed'] for member in members], 'n_seed_pairs': len(members),
            'aggregation': 'Unweighted per-seed mean, minimum and maximum; accuracy units are fractions. Publication policies are separate.',
            'evaluations': {arm: {metric: _variation({member['seed']: member['evaluations'][arm]['metrics'][metric]
                                                      for member in members})
                                  for metric in metrics} for arm in ('original', 'detached')},
            'minimum_supported_accuracy': {arm: _variation({member['seed']: member['evaluations'][arm]['minimum_supported_accuracy']
                                                            for member in members}) for arm in ('original', 'detached')},
            'detached_minus_original': {metric: _variation({member['seed']: member['detached_minus_original'][metric]['accuracy_change']
                                                           for member in members}) for metric in metrics}}


def compare_manifest(manifest_path, count_per_query=512, include_coverage=True):
    """Validate paired runs, case identity, budgets and saved scores before comparison."""
    _require(type(count_per_query) is int and count_per_query > 0 and count_per_query % 64 == 0,
             'Count per query must be a positive multiple of 64')
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get('schema_version') == 1 and isinstance(manifest.get('pairs'), list) and manifest['pairs'],
             'Expected a version-1 manifest with nonempty pairs')
    base = Path(manifest.get('base_directory', '.'))
    if not base.is_absolute(): base = manifest_path.parent/base
    files = BaselineFiles(base); files.read(manifest_path)
    pairs, names, groups, signatures, coverage_cache = [], set(), defaultdict(list), {}, {}
    for item in manifest['pairs']:
        name = item['name']
        _require(isinstance(name, str) and name and name not in names, 'Pair names must be unique')
        names.add(name)
        publication = item['publication_policy']
        _require(publication in POLICIES, f'{name}: publication policy must be original or gated')
        policy = POLICIES[publication]
        original = _baseline_run(item['original'], policy, False, files, name+'/original')
        detached = _baseline_run(item['detached'], policy, True, files, name+'/detached')
        a, b = original['protocol'], detached['protocol']
        _require(a['initial_tensor_sha256'] == b['initial_tensor_sha256'], f'{name}: matched initial tensor fingerprints differ')
        _require(_protocol_signature(a) == _protocol_signature(b), f'{name}: training protocols differ beyond baseline detach')
        _require(_learner_signature(a['learner_config']) == _learner_signature(b['learner_config']),
                 f'{name}: learner configurations differ beyond baseline detach')
        _require(a['stream'] == b['stream'], f'{name}: source streams differ')
        _require(original['summary']['parameters'] == detached['summary']['parameters'], f'{name}: parameter counts differ')
        counts = {key: original['summary']['training'][key] for key in COUNT_KEYS}
        _require(counts == {key: detached['summary']['training'][key] for key in COUNT_KEYS},
                 f'{name}: training exposure/execution counts differ')
        query_counts = {q: original['summary']['training']['by_query'][q]['count'] for q in QUERIES}
        _require(query_counts == {q: detached['summary']['training']['by_query'][q]['count'] for q in QUERIES},
                 f'{name}: training query exposure differs')
        for key in ('state_bytes', 'utility_updates'):
            _require(original['summary'].get(key) == detached['summary'].get(key), f'{name}: {key} differs')
        evaluations = {arm: _private_metrics(_load_evaluation(item[arm]['evaluation_directory'], run, policy, count_per_query, files))
                       for arm, run in (('original', original), ('detached', detached))}
        reference, changed = evaluations['original'], evaluations['detached']
        for key, message in (('metadata', 'evaluation protocols differ'), ('signatures', 'paired case IDs/answers differ'),
                             ('budgets', 'evaluation budgets differ')):
            _require(reference[key] == changed[key], f'{name}: {message}')
        auxiliary = {'view_objective': a['view_objective'], 'utility': a['utility']}
        group = f"publication_{publication}_views_{'on' if auxiliary['view_objective'] else 'off'}_utility_{'on' if auxiliary['utility'] else 'off'}"
        signature = (_protocol_signature(a, True), _learner_signature(a['learner_config'], True),
                     {key: value for key, value in a['stream'].items() if key != 'seed'},
                     a['parameters'],
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
                            'training_protocol_equal_except_baseline_detach': True,
                            'learner_configuration_equal_except_baseline_detach': True, 'source_streams_equal': True,
                            'paired_case_ids_answers_fingerprints': True, 'matched_evaluation_budgets': True},
                'baseline': {'enabled': True, 'weight': a['learner_config']['baseline_weight'],
                             'original_detach_features': False, 'detached_detach_features': True},
                'evaluation_protocol': reference['metadata'],
                'evaluations': {arm: value['public'] for arm, value in evaluations.items()},
                'detached_minus_original': _contrast(changed, reference), 'source_coverage': coverage}
        pairs.append(pair); groups[group].append(pair)
    return {'schema_version': 1, 'comparison': 'baseline_feature_gradient_ablation',
            'count_per_query': count_per_query, 'episodes_per_control': 3*count_per_query,
            'verified_pair_count': len(pairs), 'pairs': pairs,
            'groups': {name: _summarize(members) for name, members in groups.items()},
            'file_manifest': sorted(files.entries.values(), key=lambda row: row['path']),
            'integrity': {'input_file_sha256_recorded': True, 'scores_recomputed_from_predictions': True,
                          'query_and_answer_balance_verified': True, 'complete_test_and_independent_controls': True,
                          'legacy_missing_baseline_detach_normalized_false': True,
                          'same_training_and_inference_publication_policy': True,
                          'source_coverage_reconstructed': include_coverage, 'improvement_required': False},
            'contrast_definition': 'Detached minus original; paired cases at identical training and inference publication policy.',
            'intervention_scope': 'Detach baseline feature gradients for every supervised baseline target. The baseline head still trains; all other settings are matched.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count-per-query', type=int, default=512)
    parser.add_argument('--skip-coverage', action='store_true', help='Skip deterministic training-stream exposure reconstruction.')
    args = parser.parse_args(argv)
    result = compare_manifest(args.manifest, args.count_per_query, not args.skip_coverage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix+'.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n'); temporary.replace(args.out)
    print(json.dumps({'verified_pairs': result['verified_pair_count'], 'groups': {name: group['seeds'] for name, group in result['groups'].items()},
                      'count_per_query': args.count_per_query, 'output': str(args.out)}))


if __name__ == '__main__': main()
