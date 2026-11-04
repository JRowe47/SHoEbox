"""Fast saved-result and source-audit fixtures for whole-source scheduling."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from .baseline_compare import DETACH, OBSERVED
from .source_schedule_compare import compare_manifest, count_source_exposure, main, replay_schedule
from .test_baseline_compare import BaselineFixture
from .test_evidence_compare import digest


def recorded_exposure(chunks):
    """Use the training logger, independently of comparator reconstruction."""
    from .view_train import accumulate_source_exposure, empty_source_exposure, source_exposure
    totals = {'segments': 0, 'source_exposure': empty_source_exposure()}
    for chunk in chunks:
        accumulate_source_exposure(totals, source_exposure(chunk))
        totals['segments'] += 1
    return totals['source_exposure']


def recorded_baseline(chunks):
    """Derive attached-head logging counts from the dataset's causal mask."""
    from .view_train import accumulate_baseline_supervision, empty_baseline_supervision
    totals = {'segments': 0, 'baseline_supervision': empty_baseline_supervision()}
    for chunk in chunks:
        eligible = chunk.targets.ne(-100) & chunk.availability['A']
        observed = eligible & chunk.baseline_observed_mask('A')
        counts = empty_baseline_supervision(); counts['recorded_segments'] = 1
        counts.update(eligible=int(eligible.sum()), observed=int(observed.sum()), used=int(eligible.sum()))
        for row, query in enumerate(chunk.query_types):
            values = counts['by_query'][query]
            values['eligible'] += int(eligible[row].sum())
            values['observed'] += int(observed[row].sum())
            values['used'] += int(eligible[row].sum())
        accumulate_baseline_supervision(totals, counts); totals['segments'] += 1
    return totals['baseline_supervision']


class SourceScheduleFixture(BaselineFixture):
    def add_schedule_pair(self, name, seed=1, publication='gated', classes=3, legacy=True, views=True):
        from .view_data import ViewStream
        super().add_baseline_pair(name, seed=seed, publication=publication, legacy=legacy, views=views)
        self.pairs.pop()
        (self.root/name/'detached').rename(self.root/name/'balanced')
        pair = {'name': name, 'publication_policy': publication}
        for arm in ('original', 'balanced'):
            directory = self.root/name/arm
            protocol = json.loads((directory/'protocol.json').read_text())
            summary = json.loads((directory/'summary.json').read_text())
            dropout = .25 if arm == 'original' else 0.
            protocol.update(batch=12, dropout=dropout)
            protocol['stream'].update(batch_size=12, dropout=dropout)
            protocol['learner_config']['baseline_view'] = 'A'
            if arm == 'balanced' or not legacy:
                schedule = 'balanced' if arm == 'balanced' else 'dropout'
                protocol['source_schedule'] = protocol['stream']['source_schedule'] = schedule
                for flag in (DETACH, OBSERVED):
                    protocol[flag] = protocol['learner_config'][flag] = False
            stream = ViewStream(**protocol['stream'])
            chunks = [stream.next_chunk(16), stream.next_chunk(16)]
            counts = recorded_exposure(chunks)
            events = sum(chunk.targets.numel() for chunk in chunks)
            summary['training'].update(external_events=events, expert_executions=events*2,
                source_occurrences=sum(counts['source_occurrences'].values()),
                by_query={query: {'count': row['final_queries']} for query, row in counts['by_query'].items()})
            summary['final_tensor_sha256'] = digest((name, arm, 'trained'))
            if arm == 'balanced' or not legacy:
                summary['training']['source_exposure'] = counts
                summary['training']['baseline_supervision'] = recorded_baseline(chunks)
            self.write(directory/'protocol.json', protocol); self.write(directory/'summary.json', summary)
            self.evaluation(directory/'evaluation', protocol, summary, publication == 'gated',
                            1 if arm == 'original' else classes)
            pair[arm] = {'run_directory': directory.relative_to(self.root).as_posix(),
                         'evaluation_directory': (directory/'evaluation').relative_to(self.root).as_posix()}
        self.pairs.append(pair)
        return pair


class SourceScheduleComparisonChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.fixture = SourceScheduleFixture(self.temp.name)
        self.fixture.add_schedule_pair('seed1')

    def compare(self): return compare_manifest(self.fixture.manifest(), count_per_query=64)

    def test_schedule_contrast_preserves_cases_compute_and_raw_hashes(self):
        result = self.compare(); pair = result['pairs'][0]
        self.assertEqual(pair['balanced_minus_original']['independent/all/private'],
                         {'episodes': 128, 'accuracy_change': .5, 'newly_correct': 64, 'newly_wrong': 0})
        exposure = pair['source_exposure_audit']
        self.assertEqual(exposure['balanced']['recording'], 'complete_verified')
        self.assertEqual(exposure['original']['recording'], 'legacy_reconstructed_only')
        self.assertEqual(exposure['original']['training_case_sha256'], exposure['balanced']['training_case_sha256'])
        self.assertTrue(exposure['balanced']['balanced_selection_verified'])
        self.assertNotEqual(exposure['original']['expected']['source_occurrences'],
                            exposure['balanced']['expected']['source_occurrences'])
        self.assertEqual(exposure['original']['expected']['final_queries'], exposure['balanced']['expected']['final_queries'])
        self.assertEqual(pair['baseline_supervision_audit']['balanced']['recording'], 'complete_verified')
        self.assertEqual(len(result['file_manifest']), 9)
        for entry in result['file_manifest']:
            self.assertFalse(Path(entry['path']).is_absolute())
            self.assertEqual(entry['sha256'], hashlib.sha256((self.fixture.root/entry['path']).read_bytes()).hexdigest())
        self.assertNotIn(self.temp.name, json.dumps(result))

    def test_legacy_schedule_and_flags_normalize_without_mutating_raw_bytes(self):
        protocol = self.fixture.root/'seed1/original/protocol.json'
        before = protocol.read_bytes()
        self.fixture.change('seed1/original/evaluation/results.json', lambda result: result['protocol'].update(
            source_schedule='dropout', **{DETACH: False, OBSERVED: False}))
        self.assertEqual(self.compare()['verified_pair_count'], 1)
        self.assertEqual(protocol.read_bytes(), before)
        self.assertNotIn('source_schedule', json.loads(before))
        self.assertNotIn('source_schedule', json.loads(before)['stream'])

    def test_required_source_audit_rejects_missing_partial_and_forged_counts(self):
        path = 'seed1/balanced/summary.json'
        cases = (
            lambda s: s['training'].pop('source_exposure'),
            lambda s: s['training']['source_exposure'].update(recorded_segments=1, unrecorded_segments=1),
            lambda s: s['training']['source_exposure'].update(episodes_started=1),
            lambda s: s['training']['source_exposure']['source_occurrences'].update(A=1),
            lambda s: s['training']['source_exposure']['private_eligible'].update(A=999),
            lambda s: s['training']['source_exposure']['shared_available_samples'].update(C=999),
            lambda s: s['training']['source_exposure']['shared_pair_eligible'].update(A_C=999),
            lambda s: s['training']['source_exposure']['by_query']['shared']['subsets']['ABC'].update(final_queries=999),
        )
        for index, mutation in enumerate(cases):
            with self.subTest(case=index):
                before = (self.fixture.root/path).read_text(); self.fixture.change(path, mutation)
                with self.assertRaises(ValueError): self.compare()
                (self.fixture.root/path).write_text(before)

    def test_prior_recorded_source_counts_and_balanced_baseline_counts_are_checked(self):
        self.fixture.add_schedule_pair('explicit', seed=2, legacy=False)
        self.assertEqual(self.compare()['pairs'][1]['source_exposure_audit']['original']['recording'], 'complete_verified')
        path = 'explicit/original/summary.json'
        before = (self.fixture.root/path).read_text()
        self.fixture.change(path, lambda s: s['training']['source_exposure']['private_eligible'].update(C=999))
        with self.assertRaises(ValueError): self.compare()
        (self.fixture.root/path).write_text(before)
        path = 'seed1/balanced/summary.json'
        self.fixture.change(path, lambda s: s['training'].pop('baseline_supervision'))
        with self.assertRaisesRegex(ValueError, 'missing recorded baseline'): self.compare()

    def test_only_declared_schedule_and_dropout_changes_are_allowed(self):
        cases = (
            ('seed1/balanced/protocol.json', lambda p: p.update(source_schedule='dropout')),
            ('seed1/balanced/protocol.json', lambda p: p.update(dropout=.25)),
            ('seed1/original/protocol.json', lambda p: p.update(dropout=.5)),
            ('seed1/balanced/protocol.json', lambda p: p.update(lr=.003)),
            ('seed1/balanced/protocol.json', lambda p: p.update({DETACH: True})),
            ('seed1/balanced/protocol.json', lambda p: p.update({OBSERVED: True})),
            ('seed1/balanced/protocol.json', lambda p: p['learner_config'].update(baseline_weight=.2)),
            ('seed1/balanced/protocol.json', lambda p: p['stream'].update(seed=23)),
            ('seed1/balanced/protocol.json', lambda p: p.update(initial_tensor_sha256=digest('wrong'))),
            ('seed1/balanced/summary.json', lambda s: s['training'].update(expert_executions=1)),
            ('seed1/balanced/summary.json', lambda s: s['training'].update(optimizer_steps=1)),
            ('seed1/balanced/summary.json', lambda s: s['training'].update(source_occurrences=1)),
        )
        for path, mutation in cases:
            with self.subTest(path=path, mutation=mutation):
                before = (self.fixture.root/path).read_text(); self.fixture.change(path, mutation)
                with self.assertRaises(ValueError): self.compare()
                (self.fixture.root/path).write_text(before)

    def test_evaluation_policy_counts_and_paired_case_identifiers_are_checked(self):
        path = 'seed1/balanced/evaluation/results.json'
        before = (self.fixture.root/path).read_text()
        for mutation in (
                lambda r: r['inference_policy'].update(evidence_gated_ports=False),
                lambda r: r['evaluations']['test']['controls']['all'].update(exact_correct=1),
                lambda r: r['evaluations']['independent']['controls']['all'].update(expert_executions=1)):
            self.fixture.change(path, mutation)
            with self.assertRaises(ValueError): self.compare()
            (self.fixture.root/path).write_text(before)
        directory = self.fixture.root/'seed1/balanced/evaluation'
        records = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
        report = json.loads(before)
        for record in records:
            if record['cohort'] != 'independent': continue
            record['episode_keys'][0][1] += '-changed'
            report['evaluations']['independent']['controls'][record['control']]['episode_fingerprint'] = hashlib.sha256(json.dumps(record['episode_keys']).encode()).hexdigest()
        (directory/'predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
        (directory/'results.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'paired case IDs'): self.compare()

    def test_coherently_forged_pair_budgets_still_fail_stream_reconstruction(self):
        for budget in ('external', 'optimizer'):
            with self.subTest(budget=budget):
                originals = {}
                for arm in ('original', 'balanced'):
                    path = f'seed1/{arm}/summary.json'
                    originals[path] = (self.fixture.root/path).read_text()
                    summary = json.loads(originals[path]); counts = summary['training']
                    if budget == 'external':
                        counts['external_events'] *= 2
                        counts['expert_executions'] *= 2
                    else:
                        self.assertGreater(counts['optimizer_steps'], 1)
                        counts['optimizer_steps'] = 1
                    self.fixture.write(path, summary)
                message = 'external event budget' if budget == 'external' else 'optimizer step budget'
                try:
                    with self.assertRaisesRegex(ValueError, message): self.compare()
                finally:
                    for path, original in originals.items(): (self.fixture.root/path).write_text(original)

    def test_policy_aggregates_and_matched_seed_interaction_stay_separate(self):
        self.fixture.add_schedule_pair('gated2', seed=2, classes=0)
        self.fixture.add_schedule_pair('original1', publication='original', classes=2)
        self.fixture.add_schedule_pair('original2', seed=2, publication='original', classes=3)
        result = self.compare(); metric = 'test/all/private'
        gated = result['groups']['publication_gated_views_on_utility_off']
        original = result['groups']['publication_original_views_on_utility_off']
        self.assertEqual(gated['n_seed_pairs'], 2); self.assertEqual(original['n_seed_pairs'], 2)
        self.assertEqual(gated['balanced_minus_original'][metric]['mean'], .125)
        self.assertEqual(original['balanced_minus_original'][metric]['mean'], .375)
        interaction = result['policy_interactions']['views_on_utility_off']
        self.assertEqual(interaction['n_seed_pairs'], 2)
        self.assertEqual(interaction['balanced_effect_gated_minus_original'][metric],
                         {'n_seeds': 2, 'mean': -.25, 'min': -.75, 'max': .25,
                          'per_seed': {'1': .25, '2': -.75}})

    def test_policy_interaction_requires_the_same_initial_tensors_across_policy_pairs(self):
        self.fixture.add_schedule_pair('original1', publication='original')
        changed = digest('different but coherently recorded initializer')
        for arm in ('original', 'balanced'):
            self.fixture.change(f'original1/{arm}/protocol.json', lambda p: p.update(initial_tensor_sha256=changed))
            self.fixture.change(f'original1/{arm}/summary.json', lambda s: s.update(initial_tensor_sha256=changed))
            self.fixture.change(f'original1/{arm}/evaluation/results.json', lambda r: r['protocol'].update(initial_tensor_sha256=changed))
        with self.assertRaisesRegex(ValueError, 'policy interaction initial tensors'): self.compare()

    def test_source_counter_matches_independent_logger_and_case_hash_ignores_chunk_boundaries(self):
        from .view_data import ViewStream
        protocol = json.loads((self.fixture.root/'seed1/balanced/protocol.json').read_text())
        stream = ViewStream(**protocol['stream']); full = stream.batch
        chunks = [stream.next_chunk(16), stream.next_chunk(16)]
        expected = recorded_exposure(chunks)
        counts, fingerprint = count_source_exposure(chunks)
        self.assertEqual(counts, expected)
        self.assertEqual(len(fingerprint), 64)
        whole, whole_fingerprint = count_source_exposure([full])
        self.assertEqual(fingerprint, whole_fingerprint)
        for key in ('episodes_started', 'final_queries', 'supported_final_queries', 'source_occurrences'):
            self.assertEqual(counts[key], whole[key])
        summary = json.loads((self.fixture.root/'seed1/balanced/summary.json').read_text())
        replay = replay_schedule(protocol, summary['training'])
        self.assertEqual(replay['source_exposure'], expected)
        self.assertEqual(replay['training_case_sha256'], fingerprint)
        self.assertTrue(replay['balanced_selection_verified'])
        self.assertEqual(set(replay['allocation']), {'shared', 'left', 'right'})
        for query, allocation in replay['allocation'].items():
            self.assertEqual(allocation['episodes_started'], 4)
            self.assertEqual(len(allocation['factor_tuple_by_subset']), 4)
            self.assertEqual((allocation['tuple_subset_min'], allocation['tuple_subset_max']), (0, 1))
            self.assertTrue(all(len(set(counts.values())) == 1 for counts in allocation['answer_by_subset'].values()))
            self.assertEqual(sum(sum(counts.values()) for counts in allocation['answer_by_subset'].values()), 4)

    def test_wrong_requested_budget_duplicate_seed_and_cli_audit_bypass_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'evaluation count'):
            compare_manifest(self.fixture.manifest())
        duplicate = deepcopy(self.fixture.pairs[0]); duplicate['name'] = 'duplicate'
        self.fixture.pairs.append(duplicate)
        with self.assertRaisesRegex(ValueError, 'duplicate seed'): self.compare()
        self.fixture.pairs.pop()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--manifest', str(self.fixture.manifest()), '--out', str(self.fixture.root/'out.json'), '--skip-audit'])

    def test_cli_writes_verified_portable_result(self):
        output = self.fixture.root/'comparison.json'
        with redirect_stdout(io.StringIO()):
            main(['--manifest', str(self.fixture.manifest()), '--out', str(output), '--count-per-query', '64'])
        result = json.loads(output.read_text())
        self.assertEqual(result['verified_pair_count'], 1)
        self.assertEqual(result['count_per_query'], 64)
        self.assertFalse(output.with_suffix('.json.tmp').exists())


if __name__ == '__main__': unittest.main()
