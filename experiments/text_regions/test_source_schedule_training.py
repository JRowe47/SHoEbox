"""Source scheduling preserves execution and exact training continuation."""
from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest

import torch

from ..hat_minkron.train import optimizer_for
from .config import GraphConfig
from .view_data import QUERY_TYPES, VIEWS, ViewStream, make_view_batch
from .view_train import (accumulate_baseline_supervision, accumulate_source_exposure,
                         build_learner, initial_totals, load_view_checkpoint,
                         save_view_checkpoint, source_exposure, train_step)


torch.set_num_threads(2)


def equal_tree(test, left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            equal_tree(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for first, second in zip(left, right):
            equal_tree(test, first, second)
    else:
        test.assertEqual(left, right)


def accumulate_step(totals, metrics):
    for key in ('external_events', 'source_occurrences', 'expert_executions', 'optimizer_steps'):
        totals[key] += metrics[key]
    accumulate_baseline_supervision(totals, metrics['baseline_supervision'])
    accumulate_source_exposure(totals, metrics['source_exposure'])
    totals['segments'] += 1


class SourceScheduleTrainingChecks(unittest.TestCase):
    def learner(self):
        torch.manual_seed(83)
        return build_learner(GraphConfig.tiny(seed=83, evidence_gated_ports=True),
                             view_enabled=True, utility_enabled=False)

    def test_exposure_tracks_episode_boundaries_and_causal_private_eligibility(self):
        batch = make_view_batch('train', 0, 96, seed=83, gap=8, source_schedule='balanced')
        prefix = source_exposure(batch.slice(0, 7))
        self.assertEqual(prefix['private_eligible'], {'A': 0, 'C': 0})
        self.assertEqual(prefix['episodes_started'], 96)
        self.assertEqual(prefix['final_queries'], 0)
        totals = initial_totals()
        for start, stop in ((0, 16), (16, batch.length)):
            chunk = batch.slice(start, stop)
            counts = source_exposure(chunk)
            self.assertEqual(counts['recorded_segments'], 1)
            self.assertEqual(counts['episodes_started'], 96 if start == 0 else 0)
            self.assertEqual(counts['final_queries'], 96 if stop == batch.length else 0)
            self.assertEqual(counts['supported_final_queries'], counts['final_queries'])
            for view in VIEWS:
                self.assertEqual(counts['source_occurrences'][view], int(chunk.availability[view].sum()))
            for view in ('A', 'C'):
                retained = int(batch.retained_views[:, VIEWS.index(view)].sum())
                self.assertEqual(counts['private_eligible'][view], retained)
                self.assertEqual(counts['shared_available_samples'][view], retained)
            self.assertEqual(counts['shared_pair_eligible']['A_C'],
                             int((batch.retained_views[:, 0] & batch.retained_views[:, 2]).sum()))
            accumulate_source_exposure(totals, counts)
            totals['segments'] += 1
        audit = totals['source_exposure']
        self.assertEqual((audit['recorded_segments'], audit['unrecorded_segments']), (2, 0))
        self.assertEqual((audit['episodes_started'], audit['final_queries']), (96, 96))
        for query in QUERY_TYPES:
            row = audit['by_query'][query]
            self.assertEqual((row['episodes_started'], row['final_queries']), (32, 32))
            self.assertEqual(sum(item['episodes_started'] for item in row['subsets'].values()), 32)
            self.assertEqual(sum(item['final_queries'] for item in row['subsets'].values()), 32)
        withheld = make_view_batch('independent', 0, 3, seed=83, control='no_private')
        self.assertEqual(source_exposure(withheld)['private_eligible'], {'A': 0, 'C': 0})

    def test_real_steps_audit_actual_objective_inputs_without_changing_baseline_flags(self):
        learner = self.learner()
        self.assertFalse(learner.baseline_detach_features)
        self.assertFalse(learner.baseline_observed_only)
        self.assertFalse(learner.utility.enabled)
        self.assertTrue(all(value for key, value in vars(learner.view_objective.cfg).items()
                            if key.endswith('_enabled')))
        stream = ViewStream(batch_size=3, seed=83, gap=0, source_schedule='balanced')
        optimizer = optimizer_for(learner, .002)
        totals, state = initial_totals(), None
        paired, private = [], {'A': [], 'C': []}
        handles = [learner.view_objective.register_forward_hook(
            lambda module, args, output: paired.append(output['paired_samples']))]
        for view in ('A', 'C'):
            handles.append(learner.view_objective.private_readout[view].register_forward_pre_hook(
                lambda module, args, view=view: private[view].append(len(args[0]))))
        try:
            for _ in range(2):
                paired.clear()
                for values in private.values(): values.clear()
                state, metrics, logits = train_step(learner, optimizer, stream.next_chunk(16), state)
                self.assertTrue(bool(torch.isfinite(logits).all()))
                self.assertEqual(metrics['optimizer_steps'], 1)
                exposure = metrics['source_exposure']
                self.assertEqual(paired, [exposure['shared_pair_eligible']['A_C']])
                for view in ('A', 'C'):
                    self.assertEqual(sum(private[view]), exposure['private_eligible'][view])
                self.assertEqual(sum(exposure['source_occurrences'].values()), metrics['source_occurrences'])
                self.assertEqual(metrics['baseline_supervision']['used'], metrics['baseline_supervision']['eligible'])
                self.assertEqual(metrics['baseline_supervision']['suppressed'], 0)
                self.assertEqual(metrics['utility_updates'], 0)
                accumulate_step(totals, metrics)
        finally:
            for handle in handles: handle.remove()
        self.assertEqual(totals['optimizer_steps'], 2)
        self.assertEqual(totals['external_events'], 3*17)
        self.assertEqual(totals['expert_executions'], 3*17*84)
        self.assertEqual(totals['source_exposure']['episodes_started'], 3)
        self.assertEqual(totals['source_exposure']['supported_final_queries'], 3)

    @torch.no_grad()
    def test_schedule_has_no_forward_channel_and_absence_keeps_expert_budget(self):
        candidates = make_view_batch('train', 0, 28, seed=83, gap=0,
                                     query_type='shared', source_schedule='balanced')
        b_only = (~candidates.retained_views[:, 0] & candidates.retained_views[:, 1]
                  & ~candidates.retained_views[:, 2]).nonzero().flatten()
        start = int(b_only[0])
        scheduled = make_view_batch('train', start, 2, seed=83, gap=0,
                                    query_type='shared', source_schedule='balanced')
        ordinary = make_view_batch('train', start, 2, seed=83, gap=0, query_type='shared')
        equal_tree(self, scheduled.inputs, ordinary.inputs)
        equal_tree(self, scheduled.targets, ordinary.targets)
        self.assertEqual(scheduled.identities, ordinary.identities)
        manual = {view: available & scheduled.retained_views[:, index, None]
                  for index, (view, available) in enumerate(ordinary.availability.items())}
        equal_tree(self, scheduled.availability, manual)
        learner = self.learner().eval()
        first = learner(scheduled.inputs, reset=scheduled.reset, availability=scheduled.availability)
        second = learner(ordinary.inputs, reset=ordinary.reset, availability=manual)
        full = learner(ordinary.inputs, reset=ordinary.reset, availability=ordinary.availability)
        equal_tree(self, first['logits'], second['logits'])
        equal_tree(self, first['state'].to_dict(), second['state'].to_dict())
        self.assertEqual(first['expert_executions'], full['expert_executions'])
        self.assertEqual(first['external_events'], full['external_events'])
        self.assertLess(first['source_occurrences'], full['source_occurrences'])

    def test_balanced_and_legacy_checkpoints_restore_exact_next_step(self):
        for schedule, legacy in (('balanced', False), ('dropout', True)):
            with self.subTest(schedule=schedule):
                learner = self.learner()
                random.seed(83)
                options = dict(batch_size=3, gap=0, seed=83, source_schedule=schedule,
                               dropout=0. if schedule == 'balanced' else .25)
                stream = ViewStream(**options)
                optimizer = optimizer_for(learner, .002)
                state, metrics, _ = train_step(learner, optimizer, stream.next_chunk(16))
                totals = initial_totals(); accumulate_step(totals, metrics)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/'resume.pt'
                    save_view_checkpoint(path, learner, optimizer, state, stream, 1, {'lr': .002}, totals)
                    if legacy:
                        saved = torch.load(path, weights_only=True)
                        saved['protocol'].pop('source_schedule')
                        saved['stream']['spec'].pop('source_schedule')
                        saved['totals'].pop('source_exposure')
                        torch.save(saved, path)
                    rng = torch.rand(3), [random.random() for _ in range(3)]
                    expected_batch = stream.next_chunk(16)
                    expected_state, expected_metrics, expected_logits = train_step(learner, optimizer, expected_batch, state)
                    resumed_stream = ViewStream(**options)
                    resumed, resumed_optimizer, resumed_state, saved = load_view_checkpoint(path, resumed_stream)
                    self.assertEqual(saved['protocol']['source_schedule'], schedule)
                    self.assertFalse(resumed.baseline_detach_features)
                    self.assertFalse(resumed.baseline_observed_only)
                    equal_tree(self, rng, (torch.rand(3), [random.random() for _ in range(3)]))
                    actual_batch = resumed_stream.next_chunk(16)
                    equal_tree(self, vars(expected_batch), vars(actual_batch))
                    actual_state, actual_metrics, actual_logits = train_step(
                        resumed, resumed_optimizer, actual_batch, resumed_state)
                    equal_tree(self, expected_logits, actual_logits)
                    equal_tree(self, expected_state.to_dict(), actual_state.to_dict())
                    equal_tree(self, learner.state_dict(), resumed.state_dict())
                    equal_tree(self, optimizer.state_dict(), resumed_optimizer.state_dict())
                    for key in ('loss', 'ce', 'gradient_norm', 'auxiliary', 'baseline_supervision', 'source_exposure',
                                'external_events', 'source_occurrences', 'expert_executions', 'optimizer_steps'):
                        equal_tree(self, expected_metrics[key], actual_metrics[key])
                    exposure = saved['totals']['source_exposure']
                    self.assertEqual((exposure['recorded_segments'], exposure['unrecorded_segments']),
                                     (0, 1) if legacy else (1, 0))
                    accumulate_step(saved['totals'], actual_metrics)
                    self.assertEqual(exposure['recorded_segments']+exposure['unrecorded_segments'], 2)
                    equal_tree(self, vars(stream.next_chunk(16)), vars(resumed_stream.next_chunk(16)))
                    mismatched = deepcopy(torch.load(path, weights_only=True))
                    mismatched['protocol']['source_schedule'] = 'dropout' if schedule == 'balanced' else 'balanced'
                    torch.save(mismatched, path)
                    with self.assertRaises(ValueError): load_view_checkpoint(path, restore_rng=False)
                    with self.assertRaises(ValueError):
                        save_view_checkpoint(path, resumed, resumed_optimizer, actual_state, resumed_stream, 2,
                                             {'lr': .002, 'source_schedule': mismatched['protocol']['source_schedule']},
                                             saved['totals'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
