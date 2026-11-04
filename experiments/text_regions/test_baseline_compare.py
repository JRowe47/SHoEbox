"""Fast saved-result tests for the isolated baseline gradient comparison."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from .baseline_compare import DETACH, OBSERVED, compare_manifest, main
from .test_evidence_compare import Fixture, digest


class BaselineFixture(Fixture):
    def add_baseline_pair(self, name, seed=1, publication='gated', views=True,
                          original_classes=1, detached_classes=3, legacy=False):
        super().add_pair(name, seed=seed, views=views)
        self.pairs.pop()
        pair = {'name': name, 'publication_policy': publication}
        for arm, source, flag, classes in (('original', 'original', False, original_classes),
                                           ('detached', 'gated', True, detached_classes)):
            source_directory = self.root/name/source
            directory = Path(name)/arm
            if source != arm: source_directory.rename(self.root/directory)
            protocol = json.loads((self.root/directory/'protocol.json').read_text())
            summary = json.loads((self.root/directory/'summary.json').read_text())
            protocol['evidence_gated_ports'] = publication == 'gated'
            protocol['learner_config']['model']['evidence_gated_ports'] = publication == 'gated'
            protocol['learner_config']['baseline_weight'] = .1
            protocol['learner_config']['utility'] = {'enabled': protocol['utility']}
            protocol['learner_config']['view_objective'] = {'cfg': {
                term+'_enabled': views for term in ('agreement', 'prediction', 'variance', 'covariance', 'private', 'separation')}}
            if not legacy or flag:
                protocol[DETACH] = flag; protocol['learner_config'][DETACH] = flag
            self.write(directory/'protocol.json', protocol)
            evaluation = directory/'evaluation'
            self.evaluation(evaluation, protocol, summary, publication == 'gated', classes)
            pair[arm] = {'run_directory': directory.as_posix(), 'evaluation_directory': evaluation.as_posix()}
        self.pairs.append(pair)
        return pair


class BaselineComparisonChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.fixture = BaselineFixture(self.temp.name)
        self.fixture.add_baseline_pair('seed1')

    def compare(self, include_coverage=False):
        return compare_manifest(self.fixture.manifest(), count_per_query=64, include_coverage=include_coverage)

    def test_paired_private_shared_and_support_scores_preserve_raw_hashes(self):
        result = self.compare(); pair = result['pairs'][0]
        self.assertEqual(pair['detached_minus_original']['independent/b_only/shared'],
                         {'episodes': 64, 'accuracy_change': .5, 'newly_correct': 32, 'newly_wrong': 0})
        self.assertEqual(pair['detached_minus_original']['independent/b_only/private'],
                         {'episodes': 128, 'accuracy_change': .5, 'newly_correct': 64, 'newly_wrong': 0})
        detached = pair['evaluations']['detached']['cohorts']['independent']
        self.assertEqual(detached['b_only']['absent_private']['episodes'], 128)
        self.assertEqual(detached['a_only']['supported_private']['episodes'], 64)
        self.assertIsNone(detached['b_only']['supported_private']['exact_accuracy'])
        self.assertTrue(pair['matched']['same_training_and_inference_publication_policy'])
        self.assertEqual(len(result['file_manifest']), 9)
        for entry in result['file_manifest']:
            path = self.fixture.root/entry['path']
            self.assertEqual(entry['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(entry['size_bytes'], path.stat().st_size)
            self.assertFalse(Path(entry['path']).is_absolute())
        self.assertNotIn(self.temp.name, json.dumps(result))

    def test_legacy_missing_flags_and_explicit_false_rerun_are_compatible(self):
        self.fixture.add_baseline_pair('legacy', seed=2, legacy=True)
        self.fixture.change('legacy/original/evaluation/results.json', lambda result: result['protocol'].update({DETACH: False}))
        result = self.compare()
        self.assertEqual(result['verified_pair_count'], 2)
        self.assertTrue(result['integrity']['legacy_missing_baseline_detach_normalized_false'])
        original = json.loads((self.fixture.root/'legacy/original/protocol.json').read_text())
        self.assertNotIn(DETACH, original)
        self.assertNotIn(DETACH, original['learner_config'])

    def test_seed_variation_keeps_publication_and_auxiliary_groups_separate(self):
        self.fixture.add_baseline_pair('seed2', seed=2, detached_classes=0)
        self.fixture.add_baseline_pair('seed3', seed=3, detached_classes=1)
        self.fixture.add_baseline_pair('original_seed1', publication='original', detached_classes=4)
        self.fixture.add_baseline_pair('no_views', views=False, detached_classes=2)
        result = self.compare(); groups = result['groups']; metric = 'test/all/private'
        self.assertEqual(set(groups), {'publication_gated_views_on_utility_off',
                                      'publication_original_views_on_utility_off',
                                      'publication_gated_views_off_utility_off'})
        effect = groups['publication_gated_views_on_utility_off']['detached_minus_original'][metric]
        self.assertEqual(effect, {'n_seeds': 3, 'mean': 1/12, 'min': -.25, 'max': .5,
                                  'per_seed': {'1': .5, '2': -.25, '3': 0.}})
        self.assertEqual(groups['publication_original_views_on_utility_off']['n_seed_pairs'], 1)
        self.assertFalse(groups['publication_gated_views_on_utility_off']['utility_mediation_possible'])
        self.assertFalse(result['integrity']['improvement_required'])

    def test_only_detach_difference_is_permitted(self):
        cases = (
            ('seed1/detached/protocol.json', lambda p: p.update(dropout=.5), 'protocols differ'),
            ('seed1/detached/protocol.json', lambda p: p['learner_config'].update(baseline_weight=.2), 'learner configurations'),
            ('seed1/detached/protocol.json', lambda p: p['stream'].update(seed=2), 'source streams'),
            ('seed1/detached/protocol.json', lambda p: p.update(initial_tensor_sha256=digest('wrong')), 'initial tensor hash'),
            ('seed1/detached/summary.json', lambda s: s['parameters'].update(total_unique_parameters=21), 'parameter summaries'),
            ('seed1/detached/summary.json', lambda s: s['training'].update(source_occurrences=86), 'exposure/execution counts'),
            ('seed1/detached/summary.json', lambda s: s['training'].update(expert_executions=69), 'training expert budget'),
            ('seed1/detached/summary.json', lambda s: s['training']['by_query']['left'].update(count=0), 'query exposure'),
        )
        for path, mutation, message in cases:
            with self.subTest(message=message):
                original = (self.fixture.root/path).read_text(); self.fixture.change(path, mutation)
                with self.assertRaisesRegex(ValueError, message): self.compare()
                (self.fixture.root/path).write_text(original)

    def test_active_baseline_and_consistent_boolean_detach_flags_are_required(self):
        path = 'seed1/detached/protocol.json'
        cases = (
            (lambda p: p.update({DETACH: False}), 'wrong baseline detach'),
            (lambda p: p.update({DETACH: 1}), 'wrong baseline detach'),
            (lambda p: p['learner_config'].update({DETACH: False}), 'detach flags differ'),
            (lambda p: p['learner_config'].update(baseline_enabled=False), 'baseline head'),
            (lambda p: p['learner_config'].update(baseline_weight=0.), 'positive finite weight'),
            (lambda p: p['learner_config'].update(baseline_weight=float('nan')), 'positive finite weight'),
        )
        for mutation, message in cases:
            with self.subTest(message=message):
                original = (self.fixture.root/path).read_text(); self.fixture.change(path, mutation)
                with self.assertRaisesRegex(ValueError, message): self.compare()
                (self.fixture.root/path).write_text(original)

    def test_evaluation_policy_detach_fingerprints_counts_and_budgets_are_verified(self):
        path = 'seed1/detached/evaluation/results.json'
        cases = (
            (lambda r: r['inference_policy'].update(evidence_gated_ports=False), 'inference policy'),
            (lambda r: r['protocol'].update({DETACH: False}), 'training protocol differs'),
            (lambda r: r.update(model_tensor_sha256=digest('wrong')), 'tensor fingerprint'),
            (lambda r: r['evaluations']['test']['controls']['all'].update(exact_correct=1), 'prediction counts'),
            (lambda r: r['evaluations']['test']['controls']['all'].update(episode_fingerprint=digest('wrong')), 'episode fingerprint'),
            (lambda r: r['evaluations']['independent']['controls']['all'].update(expert_executions=1), 'execution budget'),
            (lambda r: r['evaluations']['independent']['controls'].pop('b_only'), 'incomplete independent controls'),
        )
        for mutation, message in cases:
            with self.subTest(message=message):
                original = (self.fixture.root/path).read_text(); self.fixture.change(path, mutation)
                with self.assertRaisesRegex(ValueError, message): self.compare()
                (self.fixture.root/path).write_text(original)

    def test_auxiliary_labels_must_match_actual_learner_flags(self):
        for term in ('utility', 'agreement', 'prediction', 'variance', 'covariance', 'private', 'separation'):
            with self.subTest(term=term):
                originals = {}
                for arm in ('original', 'detached'):
                    path = f'seed1/{arm}/protocol.json'
                    originals[path] = (self.fixture.root/path).read_text()
                    if term == 'utility':
                        self.fixture.change(path, lambda p: p['learner_config']['utility'].update(enabled=True))
                    else:
                        self.fixture.change(path, lambda p: p['learner_config']['view_objective']['cfg'].update({term+'_enabled': False}))
                with self.assertRaisesRegex(ValueError, 'protocol/config .* flags differ'): self.compare()
                for path, text in originals.items(): (self.fixture.root/path).write_text(text)

    def test_changed_case_identifiers_cannot_hide_behind_valid_fingerprints(self):
        directory = self.fixture.root/'seed1/detached/evaluation'
        records = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
        report = json.loads((directory/'results.json').read_text())
        for record in records:
            if record['cohort'] != 'independent': continue
            record['episode_keys'][0][1] += '-changed'
            report['evaluations']['independent']['controls'][record['control']]['episode_fingerprint'] = hashlib.sha256(json.dumps(record['episode_keys']).encode()).hexdigest()
        (directory/'predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
        (directory/'results.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'paired case IDs'): self.compare()

    def test_manifest_policy_duplicate_seeds_and_count_mismatches_rejected(self):
        with self.assertRaisesRegex(ValueError, 'evaluation count'):
            compare_manifest(self.fixture.manifest(), include_coverage=False)
        self.fixture.pairs[0]['publication_policy'] = 'original'
        with self.assertRaisesRegex(ValueError, 'wrong training policy'): self.compare()
        self.fixture.pairs[0]['publication_policy'] = 'gated'
        duplicate = deepcopy(self.fixture.pairs[0]); duplicate['name'] = 'duplicate'
        self.fixture.pairs.append(duplicate)
        with self.assertRaisesRegex(ValueError, 'duplicate seed pair'): self.compare()

    def test_detach_contrast_forbids_observability_differences(self):
        path = 'seed1/detached/protocol.json'
        self.fixture.change(path, lambda protocol: protocol.update({OBSERVED: True}))
        self.fixture.change(path, lambda protocol: protocol['learner_config'].update({OBSERVED: True}))
        with self.assertRaisesRegex(ValueError, 'training protocols differ'): self.compare()

    def test_deterministic_stream_reconstructs_matched_source_and_query_exposure(self):
        from .view_data import ViewStream
        protocol = json.loads((self.fixture.root/'seed1/original/protocol.json').read_text())
        stream = ViewStream(**protocol['stream']); chunks = [stream.next_chunk(16), stream.next_chunk(16)]
        events = sum(chunk.targets.numel() for chunk in chunks)
        occurrences = sum(int(mask.sum()) for chunk in chunks for mask in chunk.availability.values())
        finals = {query: sum(int(chunk.final[row].sum()) for chunk in chunks
                             for row, q in enumerate(chunk.query_types) if q == query)
                  for query in ('shared', 'left', 'right')}
        for arm in ('original', 'detached'):
            self.fixture.change(f'seed1/{arm}/summary.json', lambda summary: summary['training'].update(
                external_events=events, source_occurrences=occurrences, expert_executions=events*2,
                by_query={query: {'count': count} for query, count in finals.items()}))
        pair = self.compare(include_coverage=True)['pairs'][0]
        self.assertEqual(pair['source_coverage']['source_occurrences'], occurrences)
        self.assertEqual(pair['source_coverage']['episodes_completed'], sum(finals.values()))
        self.assertEqual(pair['source_coverage']['unsupported_supervised_query_events'], 0)
        self.assertEqual(pair['source_coverage']['supervised_query_events'], sum(int((c.targets != -100).sum()) for c in chunks))

    def test_cli_writes_portable_summary_atomically(self):
        output = self.fixture.root/'comparison.json'
        with redirect_stdout(io.StringIO()):
            main(['--manifest', str(self.fixture.manifest()), '--out', str(output), '--count-per-query', '64', '--skip-coverage'])
        result = json.loads(output.read_text())
        self.assertEqual(result['verified_pair_count'], 1)
        self.assertEqual(result['comparison'], 'baseline_feature_gradient_ablation')
        self.assertFalse(output.with_suffix('.json.tmp').exists())


if __name__ == '__main__': unittest.main()
