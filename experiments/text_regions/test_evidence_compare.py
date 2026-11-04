"""Fast saved-result fixtures for policy decomposition and integrity failures."""
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from .evidence_compare import (CONTROLS, QUERIES, VISIBLE, compare_manifest, main,
                               source_coverage)


def digest(value): return hashlib.sha256(str(value).encode()).hexdigest()


def score(predictions, answers):
    count, hits = Counter(answers), Counter(answer for prediction, answer in zip(predictions, answers) if prediction == answer)
    correct = sum(hits.values()); n = len(answers)
    return {'episodes': n, 'exact_correct': correct, 'exact_accuracy': correct/n if n else None,
            'balanced_accuracy': sum(hits[value]/count[value] for value in count)/len(count) if count else None,
            'by_answer': {chr(value): {'episodes': count[value], 'correct': hits[value]} for value in b'ABCD'}}


class Fixture:
    def __init__(self, directory):
        self.root = Path(directory); self.pairs = []

    def write(self, path, value):
        path = self.root/path; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2)+'\n')

    def add_pair(self, name, seed=1, views=True, include_go=True, gg_correct_classes=3):
        pair = {'name': name}
        for arm, policy in (('original', False), ('gated', True)):
            directory = Path(name)/arm
            parameters = {'total_unique_parameters': 20, 'parameters_including_auxiliary_heads': 40,
                          'expert_calls_per_round_per_external_event': 2}
            protocol = {'steps': 2, 'batch': 2, 'segment': 16, 'gap': 0, 'query_every': 4,
                        'dropout': .25, 'seed': seed, 'threads': 1, 'lr': .002, 'rounds': 1,
                        'view_objective': views, 'utility': False, 'baseline_all_variants': True,
                        'evidence_gated_ports': policy, 'initial_tensor_sha256': digest(('initial', seed)),
                        'base_initializer': {'path': 'initializer.pt', 'sha256': digest('base')},
                        'learner_config': {'model': {'seed': seed, 'rounds': 1, 'evidence_gated_ports': policy},
                                           'view_objective': {'enabled': views}, 'baseline_enabled': True},
                        'parameters': parameters,
                        'stream': {'batch_size': 2, 'gap': 0, 'query_every': 4, 'seed': seed, 'dropout': .25, 'query_type': None}}
            summary = {'step': 2, 'initial_tensor_sha256': protocol['initial_tensor_sha256'],
                       'final_tensor_sha256': digest((name, arm, 'trained')), 'parameters': parameters,
                       'training': {'segments': 2, 'external_events': 34, 'source_occurrences': 85,
                                    'expert_executions': 68, 'optimizer_steps': 2,
                                    'by_query': {'shared': {'count': 1}, 'left': {'count': 1}, 'right': {'count': 0}}}}
            self.write(directory/'protocol.json', protocol); self.write(directory/'summary.json', summary)
            evaluations = {}
            policies = ('original', 'gated') if arm == 'original' or include_go else ('gated',)
            for inference in policies:
                classes = {'original': {'original': 1, 'gated': 2}, 'gated': {'original': 0, 'gated': gg_correct_classes}}[arm][inference]
                eval_directory = directory/'evaluations'/inference
                self.evaluation(eval_directory, protocol, summary, inference == 'gated', classes)
                evaluations[inference] = eval_directory.as_posix()
            pair[arm] = {'run_directory': directory.as_posix(), 'evaluations': evaluations}
        self.pairs.append(pair)
        return pair

    def evaluation(self, directory, protocol, summary, policy, correct_classes):
        training_protocol = {key: value for key, value in protocol.items() if key not in ('learner_config', 'parameters', 'stream')}
        report = {'checkpoint_step': 2, 'model_tensor_sha256': summary['final_tensor_sha256'],
                  'protocol': training_protocol, 'inference_policy': {'evidence_gated_ports': policy}, 'evaluations': {}}
        records = []
        for cohort in ('test', 'independent'):
            cohort_result = {'split': cohort, 'cohort': cohort, 'seed': protocol['seed'], 'gap': 0, 'query_every': 4,
                             'count_per_query': 64, 'factor_combinations': 16 if cohort == 'test' else 64,
                             'answer_vocabulary': 'ABCD', 'scoring': 'full byte argmax', 'controls': {}}
            keys = [[query, f'{cohort}-{protocol["seed"]}-{query}-{index}'] for query in QUERIES for index in range(64)]
            answers = [b'ABCD'[index % 4] for query in QUERIES for index in range(64)]
            predictions = [answer if index % 4 < correct_classes else ord('Z') for index, answer in enumerate(answers)]
            for control in (('all',) if cohort == 'test' else CONTROLS):
                row = score(predictions, answers); row['by_query'] = {}
                for i, query in enumerate(QUERIES):
                    qrow = score(predictions[i*64:(i+1)*64], answers[i*64:(i+1)*64])
                    qrow.update(information_present=query in VISIBLE[control],
                                conditional_split_prior_chance=1. if query in VISIBLE[control] else .25,
                                answer_nats_per_byte=1.)
                    row['by_query'][query] = qrow
                for name, present in (('supported', True), ('absent_information', False)):
                    selected = [i for i, key in enumerate(keys) if (key[0] in VISIBLE[control]) == present]
                    row[name] = score([predictions[i] for i in selected], [answers[i] for i in selected])
                row.update(external_events=192*17, expert_executions=192*17*2,
                           episode_fingerprint=hashlib.sha256(json.dumps(keys).encode()).hexdigest())
                cohort_result['controls'][control] = row
                records.append({'cohort': cohort, 'control': control, 'predictions': predictions,
                                'answers': answers, 'episode_keys': keys})
            report['evaluations'][cohort] = cohort_result
        self.write(directory/'results.json', report)
        (self.root/directory/'predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))

    def manifest(self):
        path = self.root/'manifest.json'
        path.write_text(json.dumps({'schema_version': 1, 'pairs': self.pairs}))
        return path

    def change(self, relative, mutate):
        path = self.root/relative; data = json.loads(path.read_text()); mutate(data); self.write(relative, data)


class EvidenceComparisonChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(self.temp.name)
        self.fixture.add_pair('seed1')

    def compare(self): return compare_manifest(self.fixture.manifest(), count_per_query=64, include_coverage=False)

    def test_four_cells_decompose_effects_and_preserve_relative_file_manifest(self):
        result = self.compare(); pair = result['pairs'][0]
        metric = 'independent/b_only/shared'
        self.assertEqual(pair['contrasts']['inference_policy_effect_on_original_training'][metric],
                         {'episodes': 64, 'accuracy_change': .25, 'newly_correct': 16, 'newly_wrong': 0})
        self.assertEqual(pair['contrasts']['training_effect_at_gated_inference'][metric]['accuracy_change'], .25)
        self.assertEqual(pair['contrasts']['combined_change'][metric]['accuracy_change'], .5)
        self.assertEqual(pair['training_inference_interaction'][metric], .5)
        self.assertEqual(pair['evaluations']['gated_training_gated_inference']['minimum_supported_accuracy'], .75)
        absent = pair['evaluations']['gated_training_gated_inference']['cohorts']['independent']['b_only']['absent_information']
        self.assertEqual(absent['episodes'], 128)
        self.assertEqual(result['episodes_per_control'], 192)
        self.assertEqual(len(result['file_manifest']), 13)
        self.assertNotIn(self.temp.name, json.dumps(result))
        self.assertTrue(all(not Path(row['path']).is_absolute() and len(row['sha256']) == 64 for row in result['file_manifest']))

    def test_seed_variation_does_not_require_improvement_or_pool_auxiliary_groups(self):
        self.fixture.add_pair('seed2', seed=2, gg_correct_classes=1)
        self.fixture.add_pair('control1', seed=1, views=False, gg_correct_classes=2)
        result = self.compare(); groups = result['groups']; metric = 'test/all/overall'
        self.assertEqual(set(groups), {'views_on_utility_off', 'views_off_utility_off'})
        effect = groups['views_on_utility_off']['contrasts']['training_effect_at_gated_inference'][metric]
        self.assertEqual(effect, {'n_seeds': 2, 'mean': 0., 'min': -.25, 'max': .25, 'per_seed': {'1': .25, '2': -.25}})
        self.assertEqual(groups['views_off_utility_off']['n_seed_pairs'], 1)
        self.assertFalse(result['integrity']['improvement_required'])

    def test_optional_fourth_cell_and_legacy_missing_false_policy(self):
        self.fixture.pairs = []
        pair = self.fixture.add_pair('legacy', include_go=False)
        def remove_policy(protocol):
            protocol.pop('evidence_gated_ports')
            if 'learner_config' in protocol: protocol['learner_config']['model'].pop('evidence_gated_ports')
        self.fixture.change('legacy/original/protocol.json', remove_policy)
        for inference in ('original', 'gated'):
            self.fixture.change(f'legacy/original/evaluations/{inference}/results.json', lambda result: remove_policy(result['protocol']))
        self.fixture.change('legacy/original/evaluations/original/results.json', lambda result: result.pop('inference_policy'))
        result = self.compare()['pairs'][0]
        self.assertIsNone(result['training_inference_interaction'])
        self.assertEqual(len(result['contrasts']), 3)
        self.assertNotIn('gated_training_original_inference', result['evaluations'])

    def test_protocol_initialization_parameters_and_training_cost_mismatches_rejected(self):
        cases = (
            ('seed1/gated/protocol.json', lambda p: p.update(dropout=.5), 'protocols differ'),
            ('seed1/gated/protocol.json', lambda p: p.update(initial_tensor_sha256=digest('wrong')), 'initial tensor hash'),
            ('seed1/gated/summary.json', lambda s: s['parameters'].update(total_unique_parameters=21), 'parameter summaries'),
            ('seed1/gated/summary.json', lambda s: s['training'].update(expert_executions=69), 'execution counts'),
        )
        for path, mutation, message in cases:
            with self.subTest(message=message):
                original = (self.fixture.root/path).read_text()
                self.fixture.change(path, mutation)
                with self.assertRaisesRegex(ValueError, message): self.compare()
                (self.fixture.root/path).write_text(original)

    def test_legacy_baseline_flags_accept_explicit_false_reruns_but_preserve_true(self):
        flags = {'baseline_detach_features': False, 'baseline_observed_only': False}
        self.fixture.change('seed1/gated/protocol.json', lambda protocol: protocol.update(flags))
        self.fixture.change('seed1/gated/protocol.json', lambda protocol: protocol['learner_config'].update(flags))
        for inference in ('original', 'gated'):
            self.fixture.change(f'seed1/original/evaluations/{inference}/results.json',
                                lambda result: result['protocol'].update(flags))
        self.assertEqual(self.compare()['verified_pair_count'], 1)
        path = 'seed1/gated/protocol.json'
        self.fixture.change(path, lambda protocol: protocol['learner_config'].update(baseline_observed_only=True))
        with self.assertRaisesRegex(ValueError, 'learner configurations differ'): self.compare()

    def test_evaluation_fingerprint_policy_budget_and_reported_counts_rejected(self):
        path = 'seed1/original/evaluations/gated/results.json'
        cases = (
            (lambda r: r.update(model_tensor_sha256=digest('wrong')), 'tensor fingerprint'),
            (lambda r: r['inference_policy'].update(evidence_gated_ports=False), 'inference policy'),
            (lambda r: r['evaluations']['independent']['controls']['all'].update(expert_executions=1), 'execution budget'),
            (lambda r: r['evaluations']['test']['controls']['all'].update(exact_correct=1), 'prediction counts'),
            (lambda r: r['evaluations']['test']['controls']['all'].update(episode_fingerprint=digest('wrong')), 'episode fingerprint'),
            (lambda r: r['evaluations']['independent']['controls'].pop('b_only'), 'incomplete independent controls'),
        )
        for mutation, message in cases:
            with self.subTest(message=message):
                original = (self.fixture.root/path).read_text(); self.fixture.change(path, mutation)
                with self.assertRaisesRegex(ValueError, message): self.compare()
                (self.fixture.root/path).write_text(original)

    def test_pairing_failure_is_detected_even_with_recomputed_valid_fingerprints(self):
        directory = self.fixture.root/'seed1/original/evaluations/gated'
        records = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
        result = json.loads((directory/'results.json').read_text())
        for record in records:
            if record['cohort'] != 'independent': continue
            record['episode_keys'][0][1] += '-changed'
            result['evaluations']['independent']['controls'][record['control']]['episode_fingerprint'] = hashlib.sha256(json.dumps(record['episode_keys']).encode()).hexdigest()
        (directory/'predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
        (directory/'results.json').write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, 'paired case IDs'): self.compare()

    def test_wrong_requested_case_count_and_duplicate_seed_rejected(self):
        with self.assertRaisesRegex(ValueError, 'evaluation count'):
            compare_manifest(self.fixture.manifest(), count_per_query=512, include_coverage=False)
        duplicate = deepcopy(self.fixture.pairs[0]); duplicate['name'] = 'duplicate'
        self.fixture.pairs.append(duplicate)
        with self.assertRaisesRegex(ValueError, 'duplicate seed pair'): self.compare()

    def test_cli_writes_verified_summary(self):
        output = self.fixture.root/'result.json'
        with redirect_stdout(io.StringIO()):
            main(['--manifest', str(self.fixture.manifest()), '--out', str(output), '--count-per-query', '64', '--skip-coverage'])
        self.assertEqual(json.loads(output.read_text())['verified_pair_count'], 1)
        self.assertFalse(output.with_suffix('.json.tmp').exists())

    def test_source_coverage_reconstructs_retained_sources_and_exact_exposure(self):
        from .view_data import ViewStream
        spec = {'batch_size': 6, 'gap': 0, 'query_every': 4, 'seed': 73, 'dropout': 1., 'query_type': None}
        stream = ViewStream(**spec); chunks = [stream.next_chunk(16), stream.next_chunk(16)]
        counts = {'segments': 2, 'external_events': sum(chunk.targets.numel() for chunk in chunks),
                  'source_occurrences': sum(int(mask.sum()) for chunk in chunks for mask in chunk.availability.values())}
        report = source_coverage({'stream': spec, 'segment': 16}, counts)
        self.assertEqual(report['episodes_started'], 6); self.assertEqual(report['episodes_completed'], 6)
        self.assertEqual(report['unsupported_supervised_query_events'], 0)
        for query, subsets in report['query_by_retained_sources'].items():
            self.assertEqual(sum(row['episodes'] for row in subsets.values()), 2)
            self.assertTrue(all(len(subset) < 3 for subset in subsets))
            if query == 'left': self.assertTrue(all('A' in subset for subset in subsets))
            if query == 'right': self.assertTrue(all('C' in subset for subset in subsets))
        with self.assertRaisesRegex(ValueError, 'recorded exposure'):
            source_coverage({'stream': spec, 'segment': 16}, {**counts, 'external_events': counts['external_events']+1})


if __name__ == '__main__': unittest.main()
