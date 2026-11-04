"""Verify matched evidence-policy experiments and separate training/inference effects.

The input manifest names run directories and evaluation directories explicitly.
Every evaluation must include results.json and predictions.jsonl. No model is
loaded, predictions are never regenerated, and improvements are not assumed.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path


QUERIES = ('shared', 'left', 'right')
CONTROLS = ('all', 'a_only', 'b_only', 'c_only', 'no_private', 'reset_query')
VISIBLE = {'all': QUERIES, 'a_only': ('shared', 'left'), 'b_only': ('shared',),
           'c_only': ('shared', 'right'), 'no_private': ('shared',), 'reset_query': ()}
COUNT_KEYS = ('segments', 'external_events', 'source_occurrences', 'expert_executions', 'optimizer_steps')
EXTRA_PROTOCOL_KEYS = ('learner_config', 'parameters', 'stream')
ROLE_NAMES = {'oo': 'original_training_original_inference',
              'og': 'original_training_gated_inference',
              'gg': 'gated_training_gated_inference',
              'go': 'gated_training_original_inference'}
CONTRASTS = {'inference_policy_effect_on_original_training': ('og', 'oo'),
             'training_effect_at_gated_inference': ('gg', 'og'),
             'combined_change': ('gg', 'oo'),
             'training_effect_at_original_inference': ('go', 'oo'),
             'inference_policy_effect_on_gated_training': ('gg', 'go')}


def _require(condition, message):
    if not condition: raise ValueError(message)


def _equal_number(actual, expected):
    if actual is None or expected is None: return actual is expected
    return isinstance(actual, (int, float)) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


class InputFiles:
    def __init__(self, base):
        self.base, self.entries = Path(base).resolve(), {}

    def path(self, path):
        path = Path(path)
        return path.resolve() if path.is_absolute() else (self.base/path).resolve()

    def read(self, path):
        path = self.path(path)
        before = path.stat(); data = path.read_bytes(); after = path.stat()
        _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                 f'Input changed while reading: {path.name}')
        relative = Path(os.path.relpath(path, self.base)).as_posix()
        entry = {'path': relative, 'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        _require(relative not in self.entries or self.entries[relative] == entry,
                 f'Input changed between uses: {relative}')
        self.entries[relative] = entry
        return data.decode()

    def json(self, path): return json.loads(self.read(path))


def _training_protocol(protocol, remove_policy=False, remove_seed=False):
    result = deepcopy({key: value for key, value in protocol.items() if key not in EXTRA_PROTOCOL_KEYS})
    result.setdefault('evidence_gated_ports', False)
    if result.get('base_initializer') is not None:
        # An initializer location is an I/O detail; its exact bytes are not.
        result['base_initializer'] = {'sha256': result['base_initializer']['sha256']}
    if remove_policy: result.pop('evidence_gated_ports')
    if remove_seed:
        result.pop('seed', None); result.pop('initial_tensor_sha256', None)
    return result


def _learner_signature(config, remove_seed=False):
    result = deepcopy(config)
    result['model'].pop('evidence_gated_ports', None)
    if remove_seed: result['model'].pop('seed', None)
    return result


def _load_run(spec, expected_policy, files, label):
    directory = files.path(spec['run_directory'])
    protocol, summary = files.json(directory/'protocol.json'), files.json(directory/'summary.json')
    policy = protocol.get('evidence_gated_ports', False)
    _require(type(policy) is bool and policy == expected_policy, f'{label}: wrong training policy')
    model_policy = protocol['learner_config']['model'].get('evidence_gated_ports', False)
    _require(model_policy == policy, f'{label}: protocol/config training policy differs')
    _require(protocol['initial_tensor_sha256'] == summary['initial_tensor_sha256'], f'{label}: initial tensor hash differs')
    _require(protocol['parameters'] == summary['parameters'], f'{label}: parameter summaries differ')
    _require(summary['step'] == protocol['steps'] == summary['training']['segments'], f'{label}: incomplete training horizon')
    for key in COUNT_KEYS:
        _require(type(summary['training'][key]) is int and summary['training'][key] >= 0,
                 f'{label}: invalid training count {key}')
    _require(set(summary['training']['by_query']) == set(QUERIES), f'{label}: incomplete training query counts')
    for key in ('view_objective', 'utility'):
        _require(type(protocol[key]) is bool, f'{label}: invalid auxiliary flag {key}')
    return {'protocol': protocol, 'summary': summary, 'spec': spec, 'policy': policy, 'label': label}


def _score(reported, cases, label):
    correct = sum(case['prediction'] == case['answer'] for case in cases)
    n = len(cases); accuracy = correct/n if n else None
    _require(reported['episodes'] == n and reported['exact_correct'] == correct,
             f'{label}: prediction counts disagree with results')
    _require(_equal_number(reported['exact_accuracy'], accuracy), f'{label}: accuracy disagrees with predictions')
    counts, hits = Counter(case['answer'] for case in cases), Counter(case['answer'] for case in cases if case['prediction'] == case['answer'])
    balanced = sum(hits[value]/counts[value] for value in counts)/len(counts) if counts else None
    if 'balanced_accuracy' in reported:
        _require(_equal_number(reported['balanced_accuracy'], balanced), f'{label}: balanced accuracy disagrees')
    if 'by_answer' in reported:
        _require(set(reported['by_answer']) == set('ABCD'), f'{label}: missing answer classes')
        for value in b'ABCD':
            row = reported['by_answer'][chr(value)]
            _require(row['episodes'] == counts[value] and row['correct'] == hits[value], f'{label}: class counts disagree')
    return {'episodes': n, 'exact_correct': correct, 'exact_accuracy': accuracy, 'balanced_accuracy': balanced}


def _load_evaluation(directory, run, inference_policy, count_per_query, files):
    directory = files.path(directory)
    result = files.json(directory/'results.json')
    label = f"{run['label']}/{'gated' if inference_policy else 'original'} inference"
    _require(result['checkpoint_step'] == run['summary']['step'], f'{label}: checkpoint step mismatch')
    _require(result['model_tensor_sha256'] == run['summary']['final_tensor_sha256'], f'{label}: checkpoint tensor fingerprint mismatch')
    _require(_training_protocol(result['protocol']) == _training_protocol(run['protocol']), f'{label}: evaluation training protocol differs')
    declared = result.get('inference_policy', {}).get('evidence_gated_ports', run['policy'])
    _require(type(declared) is bool and declared == inference_policy, f'{label}: inference policy mismatch')
    _require(set(result['evaluations']) == {'test', 'independent'}, f'{label}: test and independent cohorts are required')
    records = {}
    for line in files.read(directory/'predictions.jsonl').splitlines():
        record = json.loads(line); key = (record['cohort'], record['control'])
        _require(key not in records, f'{label}: duplicate prediction record')
        records[key] = record
    expected = {('test', 'all')} | {('independent', control) for control in CONTROLS}
    _require(set(records) == expected, f'{label}: incomplete prediction cohorts/controls')
    metadata, scores, signatures, budgets, vectors, metrics = {}, {}, {}, {}, {}, {}
    all_cases = {}
    for cohort in ('test', 'independent'):
        data = result['evaluations'][cohort]
        expected_controls = ('all',) if cohort == 'test' else CONTROLS
        _require(set(data['controls']) == set(expected_controls), f'{label}: incomplete {cohort} controls')
        _require(data['split'] == cohort and data['count_per_query'] == count_per_query, f'{label}: evaluation count or split differs')
        _require(data['factor_combinations'] == (16 if cohort == 'test' else 64), f'{label}: wrong factor cohort')
        for key in ('seed', 'gap', 'query_every'):
            _require(data[key] == run['protocol'][key], f'{label}: evaluation {key} differs from declared training-range protocol')
        _require(data['answer_vocabulary'] == 'ABCD', f'{label}: answer vocabulary differs')
        metadata[cohort] = {key: value for key, value in data.items() if key not in ('controls', 'source_readouts', 'seconds')}
        scores[cohort] = {}; all_cases[cohort] = {}
        for control in expected_controls:
            row, record = data['controls'][control], records[(cohort, control)]
            predictions, answers, keys = record['predictions'], record['answers'], record['episode_keys']
            _require(len(predictions) == len(answers) == len(keys) == 3*count_per_query, f'{label}: unexpected case count')
            _require(all(type(value) is int and 0 <= value < 256 for value in predictions), f'{label}: predictions are not bytes')
            _require(all(value in b'ABCD' for value in answers), f'{label}: answers are not declared values')
            _require(all(isinstance(key, list) and len(key) == 2 and key[0] in QUERIES and isinstance(key[1], str) and key[1]
                         for key in keys), f'{label}: malformed episode keys')
            _require(len({key[1] for key in keys}) == len(keys), f'{label}: duplicate episode identifiers')
            fingerprint = _fingerprint(keys)
            _require(row['episode_fingerprint'] == fingerprint, f'{label}: episode fingerprint mismatch')
            signature = (keys, answers)
            if cohort in signatures: _require(signatures[cohort] == signature, f'{label}: controls do not share identical cases')
            else: signatures[cohort] = signature
            cases = [{'query': key[0], 'id': key[1], 'prediction': prediction, 'answer': answer}
                     for key, prediction, answer in zip(keys, predictions, answers)]
            checked = _score(row, cases, label+'/'+cohort+'/'+control)
            _require(set(row['by_query']) == set(QUERIES), f'{label}: missing query strata')
            checked['by_query'] = {}
            strata = {'overall': cases}
            for query in QUERIES:
                selected = [case for case in cases if case['query'] == query]
                _require(len(selected) == count_per_query and Counter(case['answer'] for case in selected) ==
                         {value: count_per_query//4 for value in b'ABCD'}, f'{label}: query/answer allocation is unbalanced')
                qrow = row['by_query'][query]; present = query in VISIBLE[control]
                _require(qrow['information_present'] == present, f'{label}: incorrect information support')
                _require(_equal_number(qrow['conditional_split_prior_chance'], 1. if present else .25), f'{label}: wrong conditional prior')
                qscore = _score(qrow, selected, label+'/'+cohort+'/'+control+'/'+query)
                qscore.update(information_present=present, conditional_split_prior_chance=qrow['conditional_split_prior_chance'])
                if 'answer_nats_per_byte' in qrow:
                    nats = qrow['answer_nats_per_byte']
                    _require(isinstance(nats, (int, float)) and math.isfinite(nats) and nats >= 0, f'{label}: invalid answer loss')
                    qscore.update(answer_nats_per_byte=nats, answer_bits_per_byte=nats/math.log(2))
                checked['by_query'][query] = qscore; strata[query] = selected
            strata['supported'] = [case for case in cases if case['query'] in VISIBLE[control]]
            strata['absent_information'] = [case for case in cases if case['query'] not in VISIBLE[control]]
            for stratum in ('supported', 'absent_information'):
                checked[stratum] = _score(row[stratum], strata[stratum], label+'/'+cohort+'/'+control+'/'+stratum)
            budget = (row['external_events'], row['expert_executions'])
            events = len(cases)*(17+data['gap'])
            expected_calls = events*run['protocol']['rounds']*run['summary']['parameters']['expert_calls_per_round_per_external_event']
            _require(budget == (events, expected_calls), f'{label}: evaluation execution budget mismatch')
            if cohort in budgets: _require(budgets[cohort] == budget, f'{label}: control execution counts differ')
            else: budgets[cohort] = budget
            checked.update(external_events=budget[0], expert_executions=budget[1], episode_fingerprint=fingerprint)
            scores[cohort][control] = checked; all_cases[cohort][control] = cases
            for stratum, selected in strata.items():
                key = f'{cohort}/{control}/{stratum}'
                hits = [case['prediction'] == case['answer'] for case in selected]
                vectors[key] = hits; metrics[key] = sum(hits)/len(hits) if hits else None
    _require(not {key[1] for key in signatures['test'][0]} & {key[1] for key in signatures['independent'][0]},
             f'{label}: test and independent identifiers overlap')
    supported = {f'{control}/{query}': row['exact_accuracy']
                 for control, score in scores['independent'].items() for query, row in score['by_query'].items()
                 if row['information_present']}
    return {'public': {'checkpoint_tensor_sha256': result['model_tensor_sha256'], 'inference_evidence_gated_ports': inference_policy,
                       'cohorts': scores, 'supported_scorecard': supported,
                       'minimum_supported_accuracy': min(supported.values()), 'metrics': metrics},
            'metadata': metadata, 'signatures': signatures, 'budgets': budgets, 'vectors': vectors}


def _contrast(new, old):
    changes = {}
    for metric, values in new['vectors'].items():
        previous = old['vectors'][metric]
        _require(len(values) == len(previous), 'Contrast case counts differ')
        count = len(values)
        wins = sum(a and not b for a, b in zip(values, previous))
        losses = sum(b and not a for a, b in zip(values, previous))
        changes[metric] = {'episodes': count, 'accuracy_change': (wins-losses)/count if count else None,
                           'newly_correct': wins, 'newly_wrong': losses}
    return changes


def source_coverage(protocol, training_counts):
    """Reconstruct the declared stream only; no model forward or optimization."""
    from .view_data import ViewStream, VIEWS
    stream = ViewStream(**protocol['stream'])
    table = {query: {} for query in QUERIES}
    events = occurrences = completed = supervised = unsupported = started = 0
    for _ in range(training_counts['segments']):
        chunk = stream.next_chunk(protocol['segment'])
        events += chunk.targets.numel(); occurrences += sum(int(value.sum()) for value in chunk.availability.values())
        completed += int(chunk.final.sum()); supervised += int((chunk.targets != -100).sum())
        unsupported += int(((chunk.targets != -100) & ~chunk.answer_available).sum())
        if chunk.time_offset == 0:
            for row, query in enumerate(chunk.query_types):
                subset = ''.join(view for index, view in enumerate(VIEWS) if bool(chunk.retained_views[row, index]))
                entry = table[query].setdefault(subset, {'episodes': 0, 'by_answer': {value: 0 for value in 'ABCD'}})
                entry['episodes'] += 1; entry['by_answer'][chr(int(chunk.answers[row]))] += 1; started += 1
    _require(events == training_counts['external_events'] and occurrences == training_counts['source_occurrences'],
             'Reconstructed source coverage differs from recorded exposure')
    _require(unsupported == 0, 'Training includes query labels without supporting evidence')
    return {'episodes_started': started, 'episodes_completed': completed, 'supervised_query_events': supervised,
            'unsupported_supervised_query_events': unsupported, 'external_events': events,
            'source_occurrences': occurrences, 'query_by_retained_sources': table}


def _variation(seed_values):
    present = {str(seed): value for seed, value in seed_values.items() if value is not None}
    values = list(present.values())
    return {'n_seeds': len(values), 'mean': sum(values)/len(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None, 'per_seed': present}


def compare_manifest(manifest_path, count_per_query=512, include_coverage=True):
    """Validate all inputs before returning per-pair and per-seed summaries."""
    _require(type(count_per_query) is int and count_per_query > 0 and count_per_query % 64 == 0,
             'Count per query must be a positive multiple of 64')
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get('schema_version') == 1 and isinstance(manifest.get('pairs'), list) and manifest['pairs'],
             'Expected a version-1 manifest with nonempty pairs')
    base = Path(manifest.get('base_directory', '.'))
    if not base.is_absolute(): base = manifest_path.parent/base
    files = InputFiles(base); files.read(manifest_path)
    pairs, names, group_signatures, groups, coverage_cache = [], set(), {}, defaultdict(list), {}
    for item in manifest['pairs']:
        name = item['name']; _require(isinstance(name, str) and name and name not in names, 'Pair names must be unique')
        names.add(name)
        original = _load_run(item['original'], False, files, name+'/original training')
        gated = _load_run(item['gated'], True, files, name+'/gated training')
        a, b = original['protocol'], gated['protocol']
        _require(a['initial_tensor_sha256'] == b['initial_tensor_sha256'], f'{name}: matched initial tensor fingerprints differ')
        _require(_training_protocol(a, True) == _training_protocol(b, True), f'{name}: training protocols differ beyond evidence policy')
        _require(_learner_signature(a['learner_config']) == _learner_signature(b['learner_config']), f'{name}: learner configurations differ')
        _require(a['stream'] == b['stream'], f'{name}: source streams differ')
        _require(original['summary']['parameters'] == gated['summary']['parameters'], f'{name}: parameter counts differ')
        counts = {key: original['summary']['training'][key] for key in COUNT_KEYS}
        _require(counts == {key: gated['summary']['training'][key] for key in COUNT_KEYS}, f'{name}: training exposure/execution counts differ')
        for query in QUERIES:
            _require(original['summary']['training']['by_query'][query]['count'] == gated['summary']['training']['by_query'][query]['count'],
                     f'{name}: training query exposure differs')
        _require(set(item['original']['evaluations']) == {'original', 'gated'}, f'{name}: both original-training inference policies required')
        _require(set(item['gated']['evaluations']) in ({'gated'}, {'original', 'gated'}), f'{name}: gated-training gated inference required')
        roles = {'oo': _load_evaluation(item['original']['evaluations']['original'], original, False, count_per_query, files),
                 'og': _load_evaluation(item['original']['evaluations']['gated'], original, True, count_per_query, files),
                 'gg': _load_evaluation(item['gated']['evaluations']['gated'], gated, True, count_per_query, files)}
        if 'original' in item['gated']['evaluations']:
            roles['go'] = _load_evaluation(item['gated']['evaluations']['original'], gated, False, count_per_query, files)
        reference = roles['oo']
        for role in roles.values():
            _require(role['metadata'] == reference['metadata'], f'{name}: evaluation protocols differ')
            _require(role['signatures'] == reference['signatures'], f'{name}: paired case IDs/answers differ')
            _require(role['budgets'] == reference['budgets'], f'{name}: evaluation budgets differ')
        contrasts = {key: _contrast(roles[new], roles[old]) for key, (new, old) in CONTRASTS.items() if new in roles and old in roles}
        interaction = None
        if 'go' in roles:
            interaction = {metric: (roles['gg']['public']['metrics'][metric]-roles['go']['public']['metrics'][metric]) -
                           (roles['og']['public']['metrics'][metric]-roles['oo']['public']['metrics'][metric])
                           if roles['gg']['public']['metrics'][metric] is not None else None for metric in reference['vectors']}
        seed = a['seed']; auxiliary = {'view_objective': a['view_objective'], 'utility': a['utility']}
        group = f"views_{'on' if auxiliary['view_objective'] else 'off'}_utility_{'on' if auxiliary['utility'] else 'off'}"
        stream_signature = {key: value for key, value in a['stream'].items() if key != 'seed'}
        eval_signature = {cohort: {key: value for key, value in meta.items() if key != 'seed'} for cohort, meta in reference['metadata'].items()}
        group_signature = (_training_protocol(a, True, True), _learner_signature(a['learner_config'], True), stream_signature, eval_signature)
        if group in group_signatures: _require(group_signatures[group] == group_signature, f'{group}: protocols differ across seeds')
        else: group_signatures[group] = group_signature
        _require(not any(pair['seed'] == seed for pair in groups[group]), f'{group}: duplicate seed pair')
        coverage = None
        if include_coverage:
            key = json.dumps({'stream': a['stream'], 'segment': a['segment'], 'counts': counts}, sort_keys=True)
            if key not in coverage_cache: coverage_cache[key] = source_coverage(a, counts)
            coverage = coverage_cache[key]
            _require(coverage['episodes_completed'] == sum(original['summary']['training']['by_query'][query]['count'] for query in QUERIES),
                     f'{name}: reconstructed final-query exposure differs')
        pair = {'name': name, 'seed': seed, 'auxiliary': auxiliary,
                'matched': {'initial_tensor_sha256': a['initial_tensor_sha256'], 'completed_steps': original['summary']['step'],
                            'training_counts': counts, 'parameters': original['summary']['parameters'],
                            'training_protocol_equal_except_evidence_policy': True,
                            'paired_case_ids_answers_fingerprints': True, 'matched_evaluation_budgets': True},
                'evaluation_protocol': reference['metadata'],
                'evaluations': {ROLE_NAMES[key]: value['public'] for key, value in roles.items()},
                'contrasts': contrasts, 'training_inference_interaction': interaction, 'source_coverage': coverage}
        pairs.append(pair); groups[group].append(pair)
    summaries = {}
    for group, members in groups.items():
        evaluated = sorted({role for member in members for role in member['evaluations']})
        contrasts = sorted({contrast for member in members for contrast in member['contrasts']})
        metrics = next(iter(members[0]['evaluations'].values()))['metrics']
        summaries[group] = {'auxiliary': members[0]['auxiliary'], 'seeds': [member['seed'] for member in members],
            'n_seed_pairs': len(members), 'aggregation': 'Unweighted per-seed mean, minimum and maximum; accuracy units are fractions.',
            'evaluations': {role: {metric: _variation({member['seed']: member['evaluations'][role]['metrics'][metric]
                                                     for member in members if role in member['evaluations']})
                                   for metric in metrics} for role in evaluated},
            'minimum_supported_accuracy': {role: _variation({member['seed']: member['evaluations'][role]['minimum_supported_accuracy']
                                                             for member in members if role in member['evaluations']}) for role in evaluated},
            'contrasts': {contrast: {metric: _variation({member['seed']: member['contrasts'][contrast][metric]['accuracy_change']
                                                       for member in members if contrast in member['contrasts']})
                                     for metric in metrics} for contrast in contrasts},
            'training_inference_interaction': {metric: _variation({member['seed']: member['training_inference_interaction'][metric]
                                                                   for member in members if member['training_inference_interaction'] is not None})
                                                for metric in metrics}}
    return {'schema_version': 1, 'count_per_query': count_per_query, 'episodes_per_control': 3*count_per_query,
            'verified_pair_count': len(pairs), 'pairs': pairs, 'groups': summaries,
            'file_manifest': sorted(files.entries.values(), key=lambda row: row['path']),
            'integrity': {'input_file_sha256_recorded': True, 'scores_recomputed_from_predictions': True,
                          'query_and_answer_balance_verified': True, 'complete_test_and_independent_controls': True,
                          'improvement_required': False, 'source_coverage_reconstructed': include_coverage},
            'contrast_definitions': {'inference_policy_effect_on_original_training': 'OG - OO',
                                     'training_effect_at_gated_inference': 'GG - OG', 'combined_change': 'GG - OO',
                                     'training_effect_at_original_inference': 'GO - OO',
                                     'inference_policy_effect_on_gated_training': 'GG - GO',
                                     'training_inference_interaction': '(GG - GO) - (OG - OO)',
                                     'notation': 'First letter: training policy; second: inference policy; O=original, G=evidence-gated.'}}


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
