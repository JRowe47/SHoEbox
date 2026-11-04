"""Matched ablations, label isolation, and exact view-training continuation."""
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import torch
from ..hat_minkron.train import optimizer_for
from .config import GraphConfig
from .model import RegionLanguageModel
from .view_data import ViewStream, make_view_batch
from .view_train import (accumulate_baseline_supervision, build_learner, initial_totals, load_view_checkpoint,
                         save_view_checkpoint, tensor_fingerprint, train_step)

torch.set_num_threads(2)


class ViewTrainingChecks(unittest.TestCase):
    def setUp(self):
        self.cfg = GraphConfig.tiny(seed=829)

    def learner(self, views=True, utility=True, detach=False, observed=False):
        torch.manual_seed(829)
        return build_learner(self.cfg, views, utility, baseline_detach_features=detach,
                             baseline_observed_only=observed)

    def assert_tree_equal(self, first, second):
        if torch.is_tensor(first):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_tree_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_four_switches_have_identical_initial_tensors_parameters_and_rng(self):
        reference, rng = None, None
        fingerprints, counts = set(), set()
        for views, utility in ((False, False), (True, False), (False, True), (True, True)):
            learner = self.learner(views, utility)
            with patch.object(torch.Tensor, 'numpy', side_effect=RuntimeError('NumPy is unavailable')):
                fingerprints.add(tensor_fingerprint(learner))
            counts.add(sum(p.numel() for p in learner.parameters()))
            self.assertTrue(learner.baseline_enabled)
            self.assertEqual(learner.baseline_weight, .1)
            self.assertEqual(learner.utility.enabled, utility)
            self.assertEqual(learner.view_objective.cfg.private_enabled, views)
            if reference is None:
                reference = {key: value.clone() for key, value in learner.state_dict().items()}
                rng = torch.get_rng_state().clone()
            else:
                self.assert_tree_equal(reference, learner.state_dict())
                torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(len(fingerprints), 1)
        self.assertEqual(len(counts), 1)

    def test_common_pretrained_graph_initializer_is_preserved_in_every_variant(self):
        torch.manual_seed(100)
        graph = RegionLanguageModel(self.cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bare.pt'
            torch.save({'config': self.cfg.to_dict(), 'weights': graph.state_dict()}, path)
            fingerprints = set()
            for views, utility in ((False, False), (True, False), (False, True), (True, True)):
                torch.manual_seed(829)
                learner = build_learner(self.cfg, views, utility, path)
                self.assert_tree_equal(graph.state_dict(), learner.model.state_dict())
                fingerprints.add(tensor_fingerprint(learner))
            self.assertEqual(len(fingerprints), 1)

    def test_baseline_switches_preserve_initial_tensors_rng_and_parameter_count(self):
        attached = self.learner()
        rng = torch.get_rng_state().clone()
        self.assertFalse(attached.baseline_detach_features)
        self.assertFalse(attached.baseline_observed_only)
        for detach, observed in ((True, False), (False, True), (True, True)):
            altered = self.learner(detach=detach, observed=observed)
            self.assertEqual(altered.baseline_detach_features, detach)
            self.assertEqual(altered.baseline_observed_only, observed)
            self.assert_tree_equal(attached.state_dict(), altered.state_dict())
            self.assertEqual(sum(p.numel() for p in attached.parameters()), sum(p.numel() for p in altered.parameters()))
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_first_private_step_is_finite_baseline_trains_in_every_arm_and_compute_matches(self):
        batch = make_view_batch('train', 0, 2, seed=829, gap=0, dropout=.5).slice(0, 16)
        budgets = set()
        for views, utility in ((False, False), (True, False), (False, True), (True, True)):
            learner = self.learner(views, utility)
            optimizer = optimizer_for(learner, .002)
            before = learner.baseline.weight.detach().clone()
            state, metrics, logits = train_step(learner, optimizer, batch)
            self.assertTrue(bool(torch.isfinite(logits).all()))
            self.assertLess(metrics['gradient_norm'], 1e6)
            self.assertGreater(metrics['gradient_norm'], 0)
            self.assertEqual(metrics['optimizer_steps'], 1)
            self.assertFalse(torch.equal(before, learner.baseline.weight))
            self.assertGreater(float(learner.baseline.weight.grad.abs().sum()), 0)
            self.assertTrue(all(value.grad_fn is None for value in state.tensors()))
            budgets.add((metrics['external_events'], metrics['source_occurrences'], metrics['expert_executions']))
            self.assertEqual(metrics['utility_updates'], int(utility))
            if views:
                self.assertGreater(float(learner.view_objective.private_readout['A'].weight.grad.abs().sum()), 0)
                self.assertGreater(metrics['auxiliary']['view_private'], 0)
        self.assertEqual(len(budgets), 1)

    def test_target_and_private_label_changes_do_not_enter_predictions_or_carried_state(self):
        learner = self.learner()
        batch = make_view_batch('train', 0, 2, seed=829, gap=0).slice(0, 16)
        private = batch.pooled_private_targets()
        altered_private = {name: torch.where(values == -100, values, 65+(values-65+1)%4)
                           for name, values in private.items()}
        altered_targets = torch.where(batch.targets == -100, batch.targets, 65+(batch.targets-65+1)%4)
        first = learner(batch.inputs, batch.targets, reset=batch.reset,
                        availability=batch.availability, private_targets=private)
        second = learner(batch.inputs, altered_targets, reset=batch.reset,
                         availability=batch.availability, private_targets=altered_private)
        for key in ('logits', 'baseline_logits'):
            self.assert_tree_equal(first[key], second[key])
        self.assert_tree_equal(first['state'].to_dict(), second['state'].to_dict())
        self.assertEqual(int(learner.utility.feedback_updates), 0)
        self.assertNotEqual(float(first['view_objective']['private'].detach()),
                            float(second['view_objective']['private'].detach()))

    def test_checkpoint_restores_exact_next_optimizer_step_stream_state_and_rng(self):
        for detach, observed in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(baseline_detach_features=detach, baseline_observed_only=observed):
                self._assert_exact_resume(detach, observed)

    def _assert_exact_resume(self, detach, observed):
        learner = self.learner(detach=detach, observed=observed)
        random.seed(112)
        stream = ViewStream(batch_size=2, gap=1, seed=829, dropout=.5)
        optimizer = optimizer_for(learner, .002)
        state, first_metrics, _ = train_step(learner, optimizer, stream.next_chunk(16))
        totals = initial_totals()
        accumulate_baseline_supervision(totals, first_metrics['baseline_supervision']);totals['segments'] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'resume.pt'
            save_view_checkpoint(path, learner, optimizer, state, stream, 1, {'lr': .002}, totals)
            expected_rng = torch.rand(4), [random.random() for _ in range(4)]
            expected_batch = stream.next_chunk(16)
            expected_state, expected_metrics, expected_logits = train_step(learner, optimizer, expected_batch, state)
            restored_stream = ViewStream(batch_size=2, gap=1, seed=829, dropout=.5)
            resumed, resumed_optimizer, restored_state, saved = load_view_checkpoint(path, restored_stream)
            self.assertEqual(resumed.baseline_detach_features, detach)
            self.assertEqual(saved['protocol']['baseline_detach_features'], detach)
            self.assertEqual(resumed.baseline_observed_only, observed)
            self.assertEqual(saved['protocol']['baseline_observed_only'], observed)
            self.assert_tree_equal(expected_rng, (torch.rand(4), [random.random() for _ in range(4)]))
            self.assertEqual(saved['totals'], totals)
            self.assertEqual(saved['step'], 1)
            actual_batch = restored_stream.next_chunk(16)
            self.assert_tree_equal(expected_batch.inputs, actual_batch.inputs)
            self.assert_tree_equal(expected_batch.availability, actual_batch.availability)
            self.assert_tree_equal(expected_batch.targets, actual_batch.targets)
            actual_state, actual_metrics, actual_logits = train_step(resumed, resumed_optimizer, actual_batch, restored_state)
            self.assert_tree_equal(expected_logits, actual_logits)
            self.assert_tree_equal(expected_state.to_dict(), actual_state.to_dict())
            self.assert_tree_equal(learner.state_dict(), resumed.state_dict())
            self.assert_tree_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
            for key in ('loss', 'ce', 'gradient_norm', 'auxiliary', 'utility_updates', 'baseline_supervision'):
                self.assert_tree_equal(expected_metrics[key], actual_metrics[key])
            self.assertEqual(stream.state_dict(), restored_stream.state_dict())

    def test_legacy_checkpoint_normalizes_flags_and_marks_unrecorded_audit(self):
        learner = self.learner()
        optimizer = optimizer_for(learner, .002)
        stream = ViewStream(batch_size=2, gap=1, seed=829)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'legacy.pt'
            save_view_checkpoint(path, learner, optimizer, None, stream, 0, {'lr': .002}, initial_totals())
            legacy = torch.load(path, weights_only=True)
            legacy['learner_config'].pop('baseline_detach_features')
            legacy['protocol'].pop('baseline_detach_features')
            legacy['learner_config'].pop('baseline_observed_only')
            legacy['protocol'].pop('baseline_observed_only')
            legacy['totals'].pop('baseline_supervision'); legacy['totals']['segments'] = 7
            torch.save(legacy, path)
            restored, _, _, saved = load_view_checkpoint(path, restore_rng=False)
            self.assertFalse(restored.baseline_detach_features)
            self.assertFalse(saved['protocol']['baseline_detach_features'])
            self.assertFalse(restored.baseline_observed_only)
            self.assertFalse(saved['protocol']['baseline_observed_only'])
            self.assertEqual(saved['totals']['baseline_supervision']['unrecorded_segments'], 7)
            self.assertEqual(saved['totals']['baseline_supervision']['recorded_segments'], 0)
            self.assert_tree_equal(learner.state_dict(), restored.state_dict())
            for key in ('baseline_detach_features', 'baseline_observed_only'):
                legacy['protocol'][key] = True
                torch.save(legacy, path)
                with self.assertRaises(ValueError): load_view_checkpoint(path, restore_rng=False)
                with self.assertRaises(ValueError):
                    save_view_checkpoint(path, learner, optimizer, None, stream, 0,
                                         {'lr': .002, key: True}, initial_totals())
                legacy['protocol'].pop(key)

    def test_baseline_audit_counts_every_supervised_event_per_query_and_accumulates(self):
        batch = make_view_batch('train', 0, 3, seed=829, gap=4, query_every=2).slice(14, 21)
        for observed in (False, True):
            learner = self.learner(views=False, utility=False, observed=observed)
            optimizer = optimizer_for(learner, .002)
            _, metrics, _ = train_step(learner, optimizer, batch)
            audit = metrics['baseline_supervision']
            for row, query in enumerate(batch.query_types):
                eligible = int(((batch.targets[row] != -100) & batch.availability['A'][row]).sum())
                known = eligible if query in ('shared', 'left') else 0
                used = known if observed else eligible
                self.assertEqual(audit['by_query'][query],
                                 {'eligible': eligible, 'observed': known, 'used': used, 'suppressed': eligible-used})
            for key in ('eligible', 'observed', 'used', 'suppressed'):
                self.assertEqual(audit[key], sum(values[key] for values in audit['by_query'].values()))
            self.assertGreater(audit['eligible'], sum(values['count'] for values in metrics['by_query'].values()))
            self.assertGreater(audit['eligible'], audit['observed'])
            totals = initial_totals()
            for _ in range(2):
                accumulate_baseline_supervision(totals, audit); totals['segments'] += 1
            self.assertEqual(totals['baseline_supervision']['recorded_segments'], 2)
            self.assertEqual(totals['baseline_supervision']['unrecorded_segments'], 0)
            for key in ('eligible', 'observed', 'used', 'suppressed'):
                self.assertEqual(totals['baseline_supervision'][key], 2*audit[key])
            legacy = {'segments': 7}
            accumulate_baseline_supervision(legacy, audit)
            self.assertEqual(legacy['baseline_supervision']['recorded_segments'], 1)
            self.assertEqual(legacy['baseline_supervision']['unrecorded_segments'], 7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
