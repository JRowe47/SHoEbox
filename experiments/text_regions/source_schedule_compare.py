"""Verify source-schedule ablations at matched model and expert budgets.

Balanced versus random-dropout training deliberately changes source occurrences
and auxiliary eligibility. These are reconstructed from causal source events,
not assumed equal. Both baseline interventions remain disabled. Prediction
validation and paired score arithmetic are shared with the earlier comparators.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from .baseline_compare import (BaselineFiles, POLICIES, _audit_supervision, _baseline_run,
                               _private_metrics, _summarize)
from .evidence_compare import (COUNT_KEYS, QUERIES, _contrast, _load_evaluation,
                               _require, _training_protocol, _variation)


VIEWS = ('A', 'B', 'C')
SUBSETS = ('A', 'B', 'AB', 'C', 'AC', 'BC', 'ABC')
PRIVATE = {'A': 'left', 'C': 'right'}
COMPUTE_KEYS = tuple(key for key in COUNT_KEYS if key != 'source_occurrences')


class ScheduleFiles(BaselineFiles):
    def json(self, path):
        value = super().json(path)
        protocol = value if Path(path).name == 'protocol.json' else value.get('protocol')
        if isinstance(protocol, dict):
            protocol.setdefault('source_schedule', 'dropout')
            if 'stream' in protocol: protocol['stream'].setdefault('source_schedule', 'dropout')
        return value


def _empty_exposure():
    row = lambda: {'episodes_started': 0, 'final_queries': 0}
    return {'schema_version': 1, 'recorded_segments': 0, 'unrecorded_segments': 0,
            'episodes_started': 0, 'final_queries': 0, 'supported_final_queries': 0,
            'by_query': {query: {**row(), 'subsets': {subset: row() for subset in SUBSETS}} for query in QUERIES},
            'source_occurrences': {view: 0 for view in VIEWS}, 'private_eligible': {'A': 0, 'C': 0},
            'shared_available_samples': {'A': 0, 'C': 0}, 'shared_pair_eligible': {'A_C': 0}}


def count_source_exposure(chunks):
    """Count source/auxiliary eligibility without trainer mask or target helpers.

Private labels are eligible at the last available occurrence in each chunk
only after that source consumes its private fact. Shared losses pool available
source occurrences, so their sample eligibility depends on availability alone.
"""
    import torch
    counts, digest, known = _empty_exposure(), hashlib.sha256(), None
    for batch in chunks:
        size = batch.targets.shape[0]
        if known is None:
            known = {view: {query: torch.zeros(size, dtype=torch.bool) for query in QUERIES} for view in VIEWS}
        _require(len(known['A']['shared']) == size, 'Source audit batch size changed')
        # Exclude availability-dependent labels/masks; all other episode data
        # must remain identical between the two source schedules.
        case = {'identities': batch.identities, 'queries': batch.query_types,
                'factors': {q: batch.factors[q].tolist() for q in QUERIES},
                'templates': {v: batch.template_ids[v].tolist() for v in VIEWS},
                'fact_positions': {v: {q: batch.fact_positions[v][q].tolist() for q in QUERIES} for v in VIEWS}}
        if batch.time_offset == 0: digest.update(json.dumps(case, sort_keys=True).encode())
        counts['recorded_segments'] += 1
        present = {v: batch.availability[v].any(1) for v in VIEWS}
        last_private = {v: torch.zeros(size, dtype=torch.bool) for v in PRIVATE}
        for view in VIEWS: counts['source_occurrences'][view] += int(batch.availability[view].sum())
        for view in PRIVATE: counts['shared_available_samples'][view] += int(present[view].sum())
        counts['shared_pair_eligible']['A_C'] += int((present['A'] & present['C']).sum())
        query_masks = {q: torch.tensor([kind == q for kind in batch.query_types]) for q in QUERIES}
        for time in range(batch.length):
            event = {'time': batch.time_offset+time, 'inputs': {v: batch.inputs[v][:, time].tolist() for v in VIEWS},
                     'targets': batch.targets[:, time].tolist(), 'reset': batch.reset[:, time].tolist(),
                     'final': batch.final[:, time].tolist()}
            digest.update(json.dumps(event, sort_keys=True).encode())
            for view in VIEWS:
                available = batch.availability[view][:, time]
                for query in QUERIES:
                    known[view][query] &= ~batch.reset[:, time]
                    known[view][query] |= available & (batch.fact_positions[view][query] == batch.time_offset+time)
                if view in PRIVATE:
                    last_private[view] = torch.where(available, known[view][PRIVATE[view]], last_private[view])
            supported = torch.stack([known[v][q] & query_masks[q] for v in VIEWS for q in QUERIES]).any(0)
            counts['supported_final_queries'] += int((batch.final[:, time] & supported).sum())
        for view in PRIVATE: counts['private_eligible'][view] += int(last_private[view].sum())
        for index, query in enumerate(batch.query_types):
            subset = ''.join(view for i, view in enumerate(VIEWS) if bool(batch.retained_views[index, i]))
            _require(subset in SUBSETS, 'Empty or invalid retained source subset')
            events = {'episodes_started': int(batch.time_offset == 0 and bool(batch.reset[index, 0])),
                      'final_queries': int(batch.final[index].sum())}
            for key, value in events.items():
                counts[key] += value
                counts['by_query'][query][key] += value
                counts['by_query'][query]['subsets'][subset][key] += value
    return counts, digest.hexdigest()


def _schedule_permutations(seed):
    """Independent schedule specification; does not call the data selector."""
    choices = {'shared': list(range(1, 8)), 'left': [1, 3, 5, 7], 'right': [4, 5, 6, 7]}
    for query, values in choices.items():
        key = repr((seed, 'balanced-source-order', query)).encode()
        generator = random.Random(int.from_bytes(hashlib.blake2b(key, digest_size=16).digest(), 'big'))
        generator.shuffle(values)
    return choices


def replay_schedule(protocol, training_counts):
    from .view_data import ViewStream, make_view_batch
    import torch
    spec = dict(protocol['stream']); spec.setdefault('source_schedule', 'dropout')
    stream = ViewStream(**spec)
    permutations = _schedule_permutations(spec['seed'])
    ordinals = {q: 0 for q in QUERIES}
    allocation = {q: {subset: {value: 0 for value in 'ABCD'} for subset in SUBSETS} for q in QUERIES}
    tuples = {q: {} for q in QUERIES}
    schedule = spec['source_schedule']
    external_events = optimizer_steps = 0
    def checked_chunks():
        nonlocal external_events, optimizer_steps
        reference = retained = None
        for _ in range(training_counts['segments']):
            chunk = stream.next_chunk(protocol['segment'])
            external_events += chunk.targets.numel()
            optimizer_steps += int(bool((chunk.targets != -100).any()) or protocol['view_objective'])
            if chunk.time_offset == 0:
                retained = chunk.retained_views.clone()
                # Render the same cases with all sources available, independently
                # of the schedule's retention helper, to audit whole-source masks.
                reference = make_view_batch('train', stream.batch_start, spec['batch_size'], seed=spec['seed'],
                                            gap=spec['gap'], query_every=spec['query_every'],
                                            query_type=spec.get('query_type'), dropout=0.)
                for row, query in enumerate(chunk.query_types):
                    subset = ''.join(view for i, view in enumerate(VIEWS) if bool(chunk.retained_views[row, i]))
                    _require(subset in SUBSETS, 'Empty or invalid retained source subset')
                    answer = chr(int(chunk.factors[query][row]))
                    factor_tuple = ''.join(chr(int(chunk.factors[q][row])) for q in QUERIES)
                    allocation[query][subset][answer] += 1
                    tuples[query].setdefault(factor_tuple, {name: 0 for name in SUBSETS})[subset] += 1
                if schedule == 'balanced':
                    for row, query in enumerate(chunk.query_types):
                        ordinal = ordinals[query]; order = permutations[query]
                        bits = order[((ordinal//4) % 8 + ordinal//32) % len(order)]
                        expected = torch.tensor([bool(bits & (1 << i)) for i in range(3)])
                        _require(torch.equal(chunk.retained_views[row], expected), 'Balanced subset differs from independent ordinal schedule')
                        ordinals[query] += 1
            _require(reference is not None, 'Source stream did not begin at an episode boundary')
            _require(torch.equal(chunk.retained_views, retained), 'Retained source subset changed within an episode')
            start, stop = chunk.time_offset, chunk.time_offset+chunk.length
            for index, view in enumerate(VIEWS):
                expected = reference.availability[view][:, start:stop] & chunk.retained_views[:, index, None]
                _require(torch.equal(chunk.availability[view], expected), 'Source availability is not a whole-source retained subset')
            yield chunk
    exposure, fingerprint = count_source_exposure(checked_chunks())
    _require(external_events == training_counts['external_events'], 'Reconstructed external event budget differs from training counts')
    _require(optimizer_steps == training_counts['optimizer_steps'], 'Reconstructed optimizer step budget differs from training counts')
    _require(sum(exposure['source_occurrences'].values()) == training_counts['source_occurrences'],
             'Reconstructed source occurrences differ from training counts')
    _require(exposure['supported_final_queries'] == exposure['final_queries'], 'Source schedule includes unsupported training queries')
    diagnostics = {}
    for query in QUERIES:
        supported = [name for name in SUBSETS if query == 'shared' or ('A' if query == 'left' else 'C') in name]
        visits = [row[subset] for row in tuples[query].values() for subset in supported]
        episodes = exposure['by_query'][query]['episodes_started']
        if schedule == 'balanced' and episodes % 4 == 0:
            _require(all(len(set(allocation[query][subset].values())) == 1 for subset in SUBSETS),
                     'Balanced source subsets are correlated with answer values')
        if schedule == 'balanced' and episodes == 1280:
            expected_range = (5, 6) if query == 'shared' else (10, 10)
            _require(len(tuples[query]) == 32 and (min(visits), max(visits)) == expected_range,
                     'Balanced 1280-query tuple/subset coverage differs from the declared schedule')
        diagnostics[query] = {'episodes_started': episodes, 'supported_subsets': supported,
                              'answer_by_subset': allocation[query], 'factor_tuple_by_subset': tuples[query],
                              'tuple_subset_min': min(visits) if visits else None,
                              'tuple_subset_max': max(visits) if visits else None}
    return {'source_exposure': exposure, 'training_case_sha256': fingerprint,
            'balanced_selection_verified': schedule == 'balanced', 'allocation': diagnostics}


def _exact_counts(actual, expected, label):
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and set(actual) == set(expected), label+': audit fields differ')
        for key, value in expected.items(): _exact_counts(actual[key], value, label+'/'+key)
    else:
        _require(type(actual) is int and actual == expected, label+': recorded count differs from causal reconstruction')


def _audit_exposure(run, cache, required):
    protocol, training = run['protocol'], run['summary']['training']
    key = json.dumps({'stream': protocol['stream'], 'segment': protocol['segment'],
                      'counts': {k: training[k] for k in COUNT_KEYS}}, sort_keys=True)
    if key not in cache: cache[key] = replay_schedule(protocol, training)
    reconstructed = cache[key]; expected = reconstructed['source_exposure']
    for query in QUERIES:
        _require(expected['by_query'][query]['final_queries'] == training['by_query'][query]['count'],
                 f"{run['label']}: final query exposure differs from reconstruction")
    recorded = training.get('source_exposure')
    if recorded is None:
        _require(not required, f"{run['label']}: missing recorded source exposure")
    else: _exact_counts(recorded, expected, run['label']+'/source_exposure')
    return {'recording': 'complete_verified' if recorded is not None else 'legacy_reconstructed_only',
            'expected': expected, 'recorded': recorded, 'training_case_sha256': reconstructed['training_case_sha256'],
            'balanced_selection_verified': reconstructed['balanced_selection_verified'], 'allocation': reconstructed['allocation']}


def _protocol_signature(protocol, remove_seed=False, remove_policy=False):
    result = _training_protocol(protocol, remove_seed=remove_seed, remove_policy=remove_policy)
    result.pop('source_schedule'); result.pop('dropout')
    return result


def _stream_signature(stream, remove_seed=False):
    result = {key: value for key, value in stream.items() if key not in ('source_schedule', 'dropout')}
    if remove_seed: result.pop('seed', None)
    return result


def _config_signature(config, remove_seed=False, remove_policy=False):
    result = deepcopy(config)
    result['model'].setdefault('evidence_gated_ports', False)
    if remove_seed: result['model'].pop('seed', None)
    if remove_policy: result['model'].pop('evidence_gated_ports')
    return result


def _load_schedule_run(spec, policy, schedule, files, name):
    run = _baseline_run(spec, policy, False, files, name, observed=False)
    protocol = run['protocol']; dropout = .25 if schedule == 'dropout' else 0.
    _require(protocol['source_schedule'] == schedule and protocol['stream']['source_schedule'] == schedule,
             name+': source schedule differs from declared arm')
    _require(type(protocol['dropout']) in (float, int) and protocol['dropout'] == dropout and
             type(protocol['stream']['dropout']) in (float, int) and protocol['stream']['dropout'] == dropout,
             name+': wrong dropout for schedule contrast')
    mapping = {'batch': 'batch_size', 'seed': 'seed', 'gap': 'gap', 'query_every': 'query_every'}
    _require(all(protocol[key] == protocol['stream'][stream_key] for key, stream_key in mapping.items()),
             name+': protocol and source stream differ')
    objective = protocol['learner_config']['view_objective']
    _require(tuple(objective.get('views', ('A', 'C'))) == ('A', 'C') and objective['cfg'].get('pooling', 'last') == 'last',
             name+': source exposure audit requires A/C objectives with last-available pooling')
    return run


def compare_manifest(manifest_path, count_per_query=512):
    _require(type(count_per_query) is int and count_per_query > 0 and count_per_query % 64 == 0,
             'Count per query must be a positive multiple of 64')
    path = Path(manifest_path).resolve(); manifest = json.loads(path.read_text())
    _require(manifest.get('schema_version') == 1 and isinstance(manifest.get('pairs'), list) and manifest['pairs'],
             'Expected a version-1 manifest with nonempty pairs')
    base = Path(manifest.get('base_directory', '.'))
    if not base.is_absolute(): base = path.parent/base
    files = ScheduleFiles(base); _require(json.loads(files.read(path)) == manifest, 'Manifest changed before hashing')
    pairs, groups, group_signatures, names = [], defaultdict(list), {}, set()
    exposure_cache, baseline_cache, interactions = {}, {}, defaultdict(dict)
    for item in manifest['pairs']:
        name = item['name']; _require(isinstance(name, str) and name and name not in names, 'Pair names must be unique'); names.add(name)
        publication = item['publication_policy']; _require(publication in POLICIES, 'Unknown publication policy')
        runs = {arm: _load_schedule_run(item[arm], POLICIES[publication], schedule, files, name+'/'+arm)
                for arm, schedule in (('original', 'dropout'), ('balanced', 'balanced'))}
        a, b = (runs[arm]['protocol'] for arm in ('original', 'balanced'))
        _require(a['initial_tensor_sha256'] == b['initial_tensor_sha256'], name+': initial tensors differ')
        _require(_protocol_signature(a) == _protocol_signature(b), name+': training protocols differ beyond source schedule')
        _require(_stream_signature(a['stream']) == _stream_signature(b['stream']), name+': training streams differ beyond source schedule')
        _require(_config_signature(a['learner_config']) == _config_signature(b['learner_config']), name+': learner configurations differ')
        _require(a['parameters'] == b['parameters'], name+': parameter budgets differ')
        counts = {arm: {key: run['summary']['training'][key] for key in COUNT_KEYS} for arm, run in runs.items()}
        _require(all(counts['original'][key] == counts['balanced'][key] for key in COMPUTE_KEYS), name+': training compute budgets differ')
        query_counts = {arm: {q: run['summary']['training']['by_query'][q]['count'] for q in QUERIES} for arm, run in runs.items()}
        _require(query_counts['original'] == query_counts['balanced'], name+': final query budgets differ')
        for key in ('state_bytes', 'utility_updates'):
            _require(runs['original']['summary'].get(key) == runs['balanced']['summary'].get(key), name+': '+key+' differs')
        audits = {arm: _audit_exposure(run, exposure_cache, required=arm == 'balanced') for arm, run in runs.items()}
        _require(audits['original']['training_case_sha256'] == audits['balanced']['training_case_sha256'],
                 name+': training cases/content differ beyond source availability')
        baseline_audits = {arm: _audit_supervision(run, counts[arm], baseline_cache, required=arm == 'balanced') for arm, run in runs.items()}
        evaluations = {arm: _private_metrics(_load_evaluation(item[arm]['evaluation_directory'], run, POLICIES[publication], count_per_query, files))
                       for arm, run in runs.items()}
        original, balanced = evaluations['original'], evaluations['balanced']
        for key, message in (('metadata', 'evaluation protocols differ'), ('signatures', 'paired case IDs/answers differ'), ('budgets', 'evaluation budgets differ')):
            _require(original[key] == balanced[key], name+': '+message)
        auxiliary = {'view_objective': a['view_objective'], 'utility': a['utility']}
        aux_name = f"views_{'on' if auxiliary['view_objective'] else 'off'}_utility_{'on' if auxiliary['utility'] else 'off'}"
        group = 'publication_'+publication+'_'+aux_name; seed = a['seed']
        signature = (_protocol_signature(a, True), _config_signature(a['learner_config'], True), _stream_signature(a['stream'], True),
                     a['parameters'], {cohort: {key: value for key, value in meta.items() if key != 'seed'} for cohort, meta in original['metadata'].items()})
        if group in group_signatures: _require(group_signatures[group] == signature, group+': protocols differ across seeds')
        else: group_signatures[group] = signature
        _require(not any(p['seed'] == seed for p in groups[group]), group+': duplicate seed pair')
        pair = {'name': name, 'seed': seed, 'publication_policy': publication, 'auxiliary': auxiliary,
                'matched': {'initial_tensor_sha256': a['initial_tensor_sha256'], 'completed_steps': runs['original']['summary']['step'],
                            'compute_counts': {key: counts['original'][key] for key in COMPUTE_KEYS}, 'training_counts': counts,
                            'training_query_counts': query_counts['original'], 'parameters': a['parameters'],
                            'training_protocol_equal_except_source_schedule': True, 'baseline_detach_and_observed_only_disabled': True,
                            'ordered_training_cases_equal': True, 'paired_case_ids_answers_fingerprints': True, 'matched_evaluation_budgets': True,
                            'source_occurrences_intentionally_not_required_equal': True},
                'source_schedules': {'original': {'source_schedule': 'dropout', 'dropout': .25},
                                     'balanced': {'source_schedule': 'balanced', 'dropout': 0.}},
                'source_exposure_audit': audits, 'baseline_supervision_audit': baseline_audits,
                'evaluation_protocol': original['metadata'], 'evaluations': {arm: value['public'] for arm, value in evaluations.items()},
                'balanced_minus_original': _contrast(balanced, original)}
        pairs.append(pair); groups[group].append(pair)
        interaction_signature = (_protocol_signature(a, remove_policy=True), _config_signature(a['learner_config'], remove_policy=True),
                                 _stream_signature(a['stream']), a['parameters'], pair['matched']['compute_counts'],
                                 original['metadata'], original['signatures'], original['budgets'])
        interactions[aux_name].setdefault(seed, {})[publication] = (pair, interaction_signature)
    policy_interactions = {}
    for auxiliary, seeds in interactions.items():
        complete = {seed: policies for seed, policies in seeds.items() if set(policies) == set(POLICIES)}
        values = defaultdict(dict)
        for seed, policies in complete.items():
            original, gated = policies['original'], policies['gated']
            _require(original[0]['matched']['initial_tensor_sha256'] == gated[0]['matched']['initial_tensor_sha256'],
                     f'{auxiliary}/{seed}: policy interaction initial tensors differ')
            _require(original[1] == gated[1], f'{auxiliary}/{seed}: policy interaction protocols or cases differ')
            for metric, row in gated[0]['balanced_minus_original'].items():
                other = original[0]['balanced_minus_original'][metric]['accuracy_change']
                values[metric][seed] = row['accuracy_change']-other if other is not None else None
        policy_interactions[auxiliary] = {'n_seed_pairs': len(complete), 'seeds': sorted(complete),
            'definition': '(Balanced - dropout) at gated publication minus (balanced - dropout) at original publication.',
            'balanced_effect_gated_minus_original': {metric: _variation(rows) for metric, rows in values.items()}}
    return {'schema_version': 1, 'comparison': 'supported_source_schedule_ablation', 'verified_pair_count': len(pairs),
            'count_per_query': count_per_query, 'episodes_per_control': 3*count_per_query, 'pairs': pairs,
            'groups': {key: _summarize(members, changed_arm='balanced') for key, members in groups.items()},
            'policy_interactions': policy_interactions, 'file_manifest': sorted(files.entries.values(), key=lambda row: row['path']),
            'integrity': {'input_file_sha256_recorded': True, 'scores_recomputed_from_predictions': True,
                          'query_and_answer_balance_verified': True, 'complete_test_and_independent_controls': True,
                          'source_exposure_and_auxiliary_eligibility_reconstructed': True, 'balanced_ordinal_selection_independently_verified': True,
                          'ordered_training_content_matched': True, 'baseline_loss_switches_disabled': True,
                          'legacy_missing_source_schedule_normalized_dropout': True, 'improvement_required': False},
            'contrast_definition': 'Balanced minus original dropout at fixed model, loss switches, publication policy and expert budget.',
            'intervention_scope': 'Changes whole-source availability and therefore source occurrences and auxiliary eligibility; these counts are audited rather than matched.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count-per-query', type=int, default=512)
    args = parser.parse_args(argv); result = compare_manifest(args.manifest, args.count_per_query)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix+'.tmp'); temporary.write_text(json.dumps(result, indent=2)+'\n'); temporary.replace(args.out)
    print(json.dumps({'verified_pairs': result['verified_pair_count'], 'count_per_query': args.count_per_query, 'output': str(args.out)}))


if __name__ == '__main__': main()
