"""Balanced cohort scoring, source boundaries, and mutation-free diagnostics."""
import itertools
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from .config import GraphConfig
from .model import RegionLanguageModel
from .objectives import SourceView, PairedViewObjective, ViewObjectiveConfig
from .view_eval import (CONTROLS, VALUES, _run_batch, _source_metrics, accuracy_summary, conditional_chance,
                        evaluate_views, evaluation_mode, fit_linear_readout, fit_source_readouts,
                        load_source_readouts, SourceReadouts)

torch.set_num_threads(2)


class ConstantModel(nn.Module):
    """A known-chance predictor; signature rejects accidental target injection."""
    def __init__(self, change_cost=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.child = nn.Dropout()
        self.register_buffer('utility_memory', torch.tensor([1., -2.]))
        self.change_cost = change_cost

    def forward(self, inputs, *, state=None, reset=None, availability=None, return_view_activity=False):
        values = next(iter(inputs.values()))
        b, t = values.shape
        logits = torch.zeros(b, t, 256, device=values.device)
        logits[..., ord('A')] = 1
        extra = int(not bool(availability['C'].any())) if self.change_cost else 0
        return {'logits': logits, 'state': None, 'external_events': b*t,
                'expert_executions': b*t*7+extra,
                'view_activity': {key: {'payload': value.float()[..., None], 'available': availability[key],
                                        'provenance': 'independent_ingress'}
                                  for key, value in inputs.items()} if return_view_activity else {}}


class BalancedMetricChecks(unittest.TestCase):
    def test_accuracy_uses_full_byte_predictions_and_class_balance(self):
        result = accuracy_summary([65, 65, 65, 65, 90], [65, 65, 66, 67, 68])
        self.assertEqual(result['exact_accuracy'], .4)
        self.assertEqual(result['balanced_accuracy'], .25)
        self.assertEqual(result['majority_class_chance'], .4)
        self.assertEqual(result['by_answer']['D']['correct'], 0)
        self.assertIsNone(accuracy_summary([], [])['exact_accuracy'])

    def test_conditional_chance_exposes_holdout_correlations(self):
        full = list(itertools.product(range(4), repeat=3))
        heldout = [row for row in full if sum(row)%4 == 3]
        self.assertEqual(conditional_chance(full, 'right', 'a_only'), .25)
        self.assertEqual(conditional_chance(heldout, 'right', 'a_only'), 1.)
        self.assertEqual(conditional_chance(full, 'left', 'no_private'), .25)
        self.assertEqual(conditional_chance(full, 'shared', 'b_only'), 1.)
        self.assertEqual(conditional_chance(heldout, 'left', 'reset_query'), .25)

    def test_balanced_controls_have_same_episodes_experts_and_correct_support(self):
        model = ConstantModel().train()
        model.child.eval()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        result = evaluate_views(model, count_per_query=64, batch_size=19, segment=7, gap=1)
        self.assertEqual(result['cohort'], 'independent_information_controls')
        self.assertEqual(tuple(result['controls']), CONTROLS)
        self.assertEqual(len({row['expert_executions'] for row in result['controls'].values()}), 1)
        self.assertEqual(len({row['episode_fingerprint'] for row in result['controls'].values()}), 1)
        expected_supported = {'all':192, 'a_only':128, 'b_only':64, 'c_only':128, 'no_private':64, 'reset_query':0}
        for control, row in result['controls'].items():
            self.assertEqual(row['episodes'], 192)
            self.assertEqual(row['exact_accuracy'], .25)
            self.assertEqual(row['macro_query_accuracy'], .25)
            self.assertEqual(row['supported']['episodes'], expected_supported[control])
            self.assertEqual(row['absent_information']['episodes'], 192-expected_supported[control])
            if control != 'all':
                self.assertEqual(row['absent_information']['conditional_split_prior_chance'], .25)
            for score in row['by_query'].values():
                self.assertTrue(all(item['episodes'] == 16 for item in score['by_answer'].values()))
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_heldout_combinations_are_separate_and_controls_restricted(self):
        result = evaluate_views(ConstantModel(), split='test', count_per_query=64, gap=0)
        self.assertEqual(result['cohort'], 'heldout_factor_combinations')
        self.assertEqual(result['factor_combinations'], 16)
        self.assertEqual(tuple(result['controls']), ('all',))
        with self.assertRaises(ValueError):
            evaluate_views(ConstantModel(), split='test', controls=('a_only',), count_per_query=64)
        with self.assertRaises(ValueError):
            evaluate_views(ConstantModel(), count_per_query=65)
        with self.assertRaises(ValueError):
            evaluate_views(ConstantModel(), controls=('all', 'carried'), count_per_query=64)
        with self.assertRaises(ValueError):
            evaluate_views(ConstantModel(), split='train', count_per_query=64)
        with self.assertRaises(ValueError):
            evaluate_views(ConstantModel(), controls=(), count_per_query=64)

    def test_changed_expert_cost_is_rejected_and_modes_restore_on_error(self):
        model = ConstantModel(change_cost=True).train()
        with self.assertRaises(ValueError):
            evaluate_views(model, count_per_query=64, gap=0, controls=('all', 'a_only'))
        self.assertTrue(model.training)
        with self.assertRaises(RuntimeError):
            with evaluation_mode(model):
                raise RuntimeError('test failure')
        self.assertTrue(model.training)


class SourceDiagnosticChecks(unittest.TestCase):
    def test_source_readout_safe_loader_preserves_exact_predictions_and_fit_identity(self):
        features = torch.eye(4).repeat(8, 1)
        head = fit_linear_readout(features, torch.tensor(VALUES).repeat(8))
        original = SourceReadouts({'ingress/A/left': head}, 'train',
                                  tuple(f'train-{index}' for index in range(32)), .01)
        payload = {'source_readout_schema': 1, **asdict(original)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source_readouts.pt'
            torch.save(payload, path)
            restored = load_source_readouts(path)
            self.assertEqual(restored.fit_episode_ids, original.fit_episode_ids)
            self.assertEqual(restored.fit_split, 'train')
            self.assertEqual(restored.ridge, original.ridge)
            torch.testing.assert_close(restored.heads['ingress/A/left'].weight, head.weight, rtol=0, atol=0)
            torch.testing.assert_close(restored.heads['ingress/A/left'].predict(features), head.predict(features), rtol=0, atol=0)
            for invalid in ({**payload, 'source_readout_schema': 2},
                            {**payload, 'fit_split': 'test'},
                            {**payload, 'fit_episode_ids': ('duplicate',)*32}):
                torch.save(invalid, path)
                with self.assertRaises(ValueError):
                    load_source_readouts(path)

    def test_ridge_readout_fits_detached_features_without_rng_or_input_mutation(self):
        features = torch.eye(4).repeat(8, 1).requires_grad_()
        labels = torch.tensor(VALUES).repeat(8)
        before = features.detach().clone()
        rng = torch.get_rng_state().clone()
        head = fit_linear_readout(features, labels)
        self.assertEqual(head.predict(torch.eye(4)).tolist(), list(VALUES))
        self.assertFalse(head.weight.requires_grad)
        self.assertIsNone(features.grad)
        torch.testing.assert_close(features, before, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        with self.assertRaises(ValueError):
            fit_linear_readout(features, labels, ridge=0)
        with self.assertRaises(ValueError):
            fit_source_readouts(ConstantModel(), split='test', count=32)

    def test_boundary_capture_matches_independent_pooling_and_chunking(self):
        from .view_data import make_view_batch
        torch.manual_seed(84)
        model = RegionLanguageModel(GraphConfig.tiny()).eval()
        batch = make_view_batch('test', 0, 2, gap=0, query_type='left')
        with evaluation_mode(model):
            whole = _run_batch(model, batch, batch.length, True)
            chunked = _run_batch(model, batch, 5, True)
            prefix = batch.slice(0, batch.description_end)
            direct = model(prefix.inputs, reset=prefix.reset, availability=prefix.availability,
                           return_view_activity=True)
            for source in ('A', 'B', 'C'):
                expected = SourceView.from_activity(source, direct['view_activity']).pool('last')
                torch.testing.assert_close(whole[3][source], expected.payload, rtol=0, atol=0)
                torch.testing.assert_close(chunked[3][source], expected.payload, rtol=0, atol=0)
                self.assertTrue(torch.equal(chunked[4][source], expected.available))
            torch.testing.assert_close(whole[0], chunked[0], rtol=0, atol=0)
            self.assertEqual(whole[1], chunked[1])

    def test_frozen_train_only_source_fit_and_heldout_scores_are_separate(self):
        model = ConstantModel().train()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        readouts = fit_source_readouts(model, count=32, batch_size=11, segment=7, gap=0)
        self.assertEqual(readouts.fit_split, 'train')
        self.assertEqual(set(readouts.heads), {'ingress/A/shared', 'ingress/A/left', 'ingress/B/shared',
                                              'ingress/C/shared', 'ingress/C/right'})
        result = evaluate_views(model, split='test', count_per_query=64, batch_size=21,
                                segment=7, gap=0, source_readouts=readouts)
        metrics = result['source_readouts']
        self.assertEqual(metrics['fit_split'], 'train')
        self.assertEqual(metrics['evaluation_control'], 'all')
        for name, row in metrics['frozen_linear'].items():
            self.assertEqual(row['episodes'], 192)
            self.assertEqual(row['fitted_examples'], 32)
            self.assertEqual(readouts.heads[name].fitted_examples, 32)
        self.assertTrue(model.training)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_private_head_metadata_exposes_prequery_timing_difference(self):
        model = ConstantModel()
        model.view_objective = PairedViewObjective(1, views=('A', 'C'), cfg=ViewObjectiveConfig(private_enabled=True))
        features = {f'private/{source}': [torch.zeros(4, 16)] for source in ('A', 'C')}
        labels = {factor: [torch.tensor(VALUES)] for factor in ('shared', 'left', 'right')}
        masks = {source: [torch.ones(4, dtype=torch.bool)] for source in ('A', 'C')}
        with evaluation_mode(model):
            metrics = _source_metrics(model, features, labels, masks, None)
        for row in metrics['trained_private_heads'].values():
            self.assertEqual(row['evaluation_timing']['description_end_exclusive'], 14)
            self.assertEqual(row['evaluation_timing']['latest_possible_event_index'], 13)
            self.assertFalse(row['evaluation_timing']['query_markers_observed'])
            self.assertFalse(row['timing_matches_training_pool'])
            self.assertIn('event 15', row['timing_note'])


if __name__ == '__main__':
    unittest.main()
