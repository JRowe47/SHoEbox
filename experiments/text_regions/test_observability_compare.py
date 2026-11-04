"""Saved-result and causal-stream checks for source-aware baseline supervision."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
import shutil
import unittest

from .baseline_compare import DETACH, OBSERVED, count_baseline_supervision
from .observability_compare import compare_manifest, main
from .test_baseline_compare import BaselineFixture
from .test_evidence_compare import digest


class ObservabilityFixture(BaselineFixture):
    def add_observed_pair(self, name, seed=1, publication='gated', context=True, legacy=True, classes=2):
        from .view_data import ViewStream
        pair = super().add_baseline_pair(name, seed=seed, publication=publication, legacy=legacy)
        self.pairs.pop()
        shutil.copytree(self.root/name/'original', self.root/name/'observed')
        result = {'name': name, 'publication_policy': publication}
        for arm in ('original', 'observed', 'detached'):
            directory = self.root/name/arm
            protocol = json.loads((directory/'protocol.json').read_text())
            protocol.update(batch=3, dropout=0.)
            protocol['stream'].update(batch_size=3, dropout=0.)
            protocol['learner_config']['baseline_view'] = 'A'
            if arm == 'observed' or not legacy:
                protocol[OBSERVED] = arm == 'observed'
                protocol['learner_config'][OBSERVED] = arm == 'observed'
            if arm == 'observed':
                protocol[DETACH] = protocol['learner_config'][DETACH] = False
            stream = ViewStream(**protocol['stream']); chunks = [stream.next_chunk(16), stream.next_chunk(16)]
            summary = json.loads((directory/'summary.json').read_text())
            events = sum(chunk.targets.numel() for chunk in chunks)
            summary['training'].update(external_events=events, expert_executions=events*2,
                source_occurrences=sum(int(mask.sum()) for chunk in chunks for mask in chunk.availability.values()),
                by_query={q: {'count': 1} for q in ('shared', 'left', 'right')})
            if arm == 'observed':
                summary['final_tensor_sha256'] = digest((name, 'observed'))
                # Three all-source episodes, each with query events 14,15,16.
                # Source A has shared and left facts and never has the right fact.
                summary['training']['baseline_supervision'] = {
                    'eligible': 9, 'observed': 6, 'used': 6, 'suppressed': 3,
                    'by_query': {'shared': {'eligible': 3, 'observed': 3, 'used': 3, 'suppressed': 0},
                                 'left': {'eligible': 3, 'observed': 3, 'used': 3, 'suppressed': 0},
                                 'right': {'eligible': 3, 'observed': 0, 'used': 0, 'suppressed': 3}},
                    'recorded_segments': 2, 'unrecorded_segments': 0}
            self.write(directory/'protocol.json', protocol); self.write(directory/'summary.json', summary)
            self.evaluation(directory/'evaluation', protocol, summary, publication == 'gated',
                            {'original': 1, 'observed': classes, 'detached': 3}[arm])
            if arm != 'detached' or context:
                result[arm] = {'run_directory': directory.relative_to(self.root).as_posix(),
                               'evaluation_directory': (directory/'evaluation').relative_to(self.root).as_posix()}
        self.pairs.append(result)
        return result


class ObservabilityComparisonChecks(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.fixture = ObservabilityFixture(self.temp.name)
        self.fixture.add_observed_pair('seed1')

    def compare(self, coverage=False):
        return compare_manifest(self.fixture.manifest(), count_per_query=64, include_coverage=coverage)

    def test_primary_mask_contrast_and_optional_detach_context_are_separate(self):
        result = self.compare(); pair = result['pairs'][0]; metric = 'independent/all/private'
        self.assertEqual(pair['observed_minus_original'][metric]['accuracy_change'], .25)
        self.assertEqual(pair['detached_minus_original'][metric]['accuracy_change'], .5)
        self.assertFalse(pair['baseline']['observed_detach_features'])
        audits = pair['baseline_supervision_audit']
        self.assertEqual(audits['observed']['recording'], 'complete_verified')
        self.assertEqual(audits['original']['recording'], 'legacy_reconstructed_only')
        self.assertEqual(audits['detached']['recording'], 'legacy_reconstructed_only')
        self.assertEqual(audits['observed']['expected']['suppressed'], 3)
        self.assertEqual(audits['original']['expected']['used'], 9)
        self.assertEqual(audits['observed']['expected']['by_query']['right']['used'], 0)
        self.assertFalse(result['integrity']['source_coverage_reconstructed'])
        self.assertTrue(result['integrity']['baseline_supervision_counts_causally_reconstructed'])
        self.assertEqual(len(result['file_manifest']), 13)
        for entry in result['file_manifest']:
            self.assertEqual(entry['sha256'], hashlib.sha256((self.fixture.root/entry['path']).read_bytes()).hexdigest())

    def test_legacy_flags_normalize_in_evaluation_without_changing_raw_bytes(self):
        path = 'seed1/original/evaluation/results.json'
        self.fixture.change(path, lambda result: result['protocol'].update({DETACH: False, OBSERVED: False}))
        original = (self.fixture.root/'seed1/original/protocol.json').read_bytes()
        result = self.compare()
        self.assertEqual((self.fixture.root/'seed1/original/protocol.json').read_bytes(), original)
        self.assertNotIn(OBSERVED, json.loads(original))
        self.assertEqual(result['verified_pair_count'], 1)

    def test_required_observed_audit_cannot_be_missing_partial_or_forged(self):
        path = 'seed1/observed/summary.json'
        cases = (
            (lambda s: s['training'].pop('baseline_supervision'), 'missing recorded'),
            (lambda s: s['training']['baseline_supervision'].update(recorded_segments=1, unrecorded_segments=1), 'incomplete baseline'),
            (lambda s: s['training']['baseline_supervision'].update(eligible=8), 'total/eligible'),
            (lambda s: s['training']['baseline_supervision'].update(observed=9), 'total/observed'),
            (lambda s: s['training']['baseline_supervision'].update(used=9, suppressed=0), 'total/used'),
            (lambda s: s['training']['baseline_supervision']['by_query']['right'].update(observed=3), 'right/observed'),
            (lambda s: s['training']['baseline_supervision']['by_query'].pop('left'), 'query audit'),
        )
        for mutation, message in cases:
            with self.subTest(message=message):
                text = (self.fixture.root/path).read_text(); self.fixture.change(path, mutation)
                with self.assertRaisesRegex(ValueError, message): self.compare()
                (self.fixture.root/path).write_text(text)

    def test_recorded_prior_audit_is_checked_when_present(self):
        observed = json.loads((self.fixture.root/'seed1/observed/summary.json').read_text())['training']['baseline_supervision']
        prior = deepcopy(observed)
        for row in [prior, *prior['by_query'].values()]: row['used'], row['suppressed'] = row['eligible'], 0
        path = 'seed1/original/summary.json'
        self.fixture.change(path, lambda s: s['training'].update(baseline_supervision=prior))
        self.assertEqual(self.compare()['pairs'][0]['baseline_supervision_audit']['original']['recording'], 'complete_verified')
        self.fixture.change(path, lambda s: s['training']['baseline_supervision']['by_query']['right'].update(used=0))
        with self.assertRaisesRegex(ValueError, 'right/used'): self.compare()

    def test_mask_is_the_only_primary_change_and_configuration_labels_agree(self):
        path = 'seed1/observed/protocol.json'
        cases = (
            (lambda p: p.update({DETACH: True}), 'wrong baseline detach'),
            (lambda p: p.update({OBSERVED: False}), 'wrong baseline observed-only'),
            (lambda p: p['learner_config'].update({OBSERVED: False}), 'observed-only flags differ'),
            (lambda p: p.update(dropout=.5), 'training protocols differ'),
            (lambda p: p['learner_config']['utility'].update(enabled=True), 'utility flags differ'),
            (lambda p: p['learner_config'].update(baseline_weight=.2), 'learner configurations differ'),
        )
        for mutation, message in cases:
            with self.subTest(message=message):
                text = (self.fixture.root/path).read_text(); self.fixture.change(path, mutation)
                with self.assertRaisesRegex(ValueError, message): self.compare()
                (self.fixture.root/path).write_text(text)

    def test_detached_context_is_optional_and_requires_mask_disabled(self):
        path = 'seed1/detached/protocol.json'
        self.fixture.change(path, lambda p: p.update({OBSERVED: True}))
        self.fixture.change(path, lambda p: p['learner_config'].update({OBSERVED: True}))
        with self.assertRaisesRegex(ValueError, 'wrong baseline observed-only'): self.compare()
        self.fixture.pairs[0].pop('detached')
        pair = self.compare()['pairs'][0]
        self.assertNotIn('detached', pair['evaluations'])
        self.assertNotIn('detached_minus_original', pair)

    def test_evaluation_mask_protocol_and_case_identity_must_match(self):
        path = 'seed1/observed/evaluation/results.json'
        text = (self.fixture.root/path).read_text()
        self.fixture.change(path, lambda result: result['protocol'].update({OBSERVED: False}))
        with self.assertRaisesRegex(ValueError, 'training protocol differs'): self.compare()
        (self.fixture.root/path).write_text(text)
        directory = self.fixture.root/'seed1/observed/evaluation'
        records = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
        report = json.loads(text)
        for record in records:
            if record['cohort'] != 'independent': continue
            record['episode_keys'][0][1] += '-changed'
            report['evaluations']['independent']['controls'][record['control']]['episode_fingerprint'] = hashlib.sha256(json.dumps(record['episode_keys']).encode()).hexdigest()
        (directory/'predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
        (directory/'results.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'paired case IDs'): self.compare()

    def test_policy_and_seed_aggregates_include_only_available_context_seeds(self):
        self.fixture.add_observed_pair('seed2', seed=2, context=False, classes=0)
        self.fixture.add_observed_pair('seed3', seed=3, context=False, classes=1)
        self.fixture.add_observed_pair('original_seed1', publication='original', context=False, classes=4)
        result = self.compare(); group = result['groups']['publication_gated_views_on_utility_off']
        self.assertEqual(group['n_seed_pairs'], 3)
        self.assertEqual(group['observed_minus_original']['test/all/private']['mean'], 0.)
        self.assertEqual(group['evaluations']['detached']['test/all/private']['n_seeds'], 1)
        self.assertEqual(result['groups']['publication_original_views_on_utility_off']['n_seed_pairs'], 1)

    def test_causal_audit_respects_fact_identity_prefix_reset_and_chunk_boundaries(self):
        from .view_data import make_view_batch
        batch = make_view_batch('train', 0, 3, seed=7, gap=8)
        targets = batch.targets.clone().fill_(-100); resets = batch.reset.clone()
        fact = int(batch.fact_positions['A']['shared'][0])
        targets[0, 1] = batch.answers[0]
        targets[0, fact] = batch.answers[0]
        targets[:, -1] = batch.answers
        # Equal byte values from A do not make a right-private factor observed.
        targets[2, -1] = batch.factors['left'][2]
        resets[1, -1] = True
        batch = replace(batch, targets=targets, reset=resets)
        whole = count_baseline_supervision([batch])
        chunked = count_baseline_supervision([batch.slice(0, 8), batch.slice(8, 16), batch.slice(16, batch.length)])
        self.assertEqual(whole['eligible'], 5); self.assertEqual(whole['observed'], 2)
        self.assertEqual(whole['suppressed'], 3)
        self.assertEqual(whole['by_query']['left']['observed'], 0)
        self.assertEqual(whole['by_query']['right']['observed'], 0)
        self.assertEqual({k:v for k,v in whole.items() if k != 'recorded_segments'},
                         {k:v for k,v in chunked.items() if k != 'recorded_segments'})

    def test_cli_validates_audit_even_when_general_coverage_is_skipped(self):
        output = self.fixture.root/'comparison.json'
        with redirect_stdout(io.StringIO()):
            main(['--manifest', str(self.fixture.manifest()), '--out', str(output), '--count-per-query', '64', '--skip-coverage'])
        result = json.loads(output.read_text())
        self.assertEqual(result['comparison'], 'baseline_observability_ablation')
        self.assertTrue(result['integrity']['baseline_supervision_counts_causally_reconstructed'])


if __name__ == '__main__': unittest.main()
