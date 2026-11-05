"""Causal local targets, graph isolation and exact phase continuation."""
import contextlib
from dataclasses import replace
import io
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from ..hat_minkron.train import optimizer_for
from .composition_train import build_model
from .config import GraphConfig
from .episode_data import make_batch, observed_memory_targets
from .fact_supervision import attach_heads, forward_sources, retention_loss, view_targets
from .integrated_train import make_streams, save_integrated, load_integrated, stream_loss
from .regional_train import main
from .test_integrated import progress
from .test_probe import assert_tree_equal
from .view_data import make_view_batch, QUERY_TYPES
from scripts.train_integrated_balanced import balanced_update, clip_task_gradients

torch.set_num_threads(1)


def model_fixture():
    torch.manual_seed(830)
    model = build_model(GraphConfig.tiny(readout_mode='concat', rounds=2,
                                        recurrent_feedback=True), .5, 1.)
    optimizer = optimizer_for(model, .0005)
    return model, optimizer


class FactSupervisionChecks(unittest.TestCase):
    def test_gradient_clipping_handles_finite_overflow_without_changing_direction(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([3e20, 4e20])
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertFalse(torch.isfinite(parameter.grad.norm()))
        norm = clip_task_gradients([parameter])
        self.assertTrue(math.isfinite(float(norm)))
        self.assertAlmostEqual(float(norm)/5e20, 1., places=6)
        torch.testing.assert_close(parameter.grad, torch.tensor([.6, .8]))
        self.assertAlmostEqual(float(parameter.grad.norm()), 1., places=6)

    def test_gradient_clipping_preserves_ordinary_path_and_rejects_nonfinite(self):
        for values in ([.03, .04], [3., 4.], [0., 0.]):
            actual = torch.nn.Parameter(torch.zeros(2))
            expected = torch.nn.Parameter(torch.zeros(2))
            actual.grad = torch.tensor(values); expected.grad = actual.grad.clone()
            norm = clip_task_gradients([actual])
            reference = torch.nn.utils.clip_grad_norm_([expected], 1., error_if_nonfinite=True)
            self.assertTrue(torch.equal(norm, reference))
            self.assertTrue(torch.equal(actual.grad, expected.grad))
        for value in (float('nan'), float('inf'), -float('inf')):
            parameter = torch.nn.Parameter(torch.zeros(2))
            parameter.grad = torch.tensor([value, 1.])
            before = parameter.grad.clone()
            with self.assertRaises(RuntimeError):
                clip_task_gradients([parameter])
            torch.testing.assert_close(parameter.grad, before, equal_nan=True, rtol=0, atol=0)

    def test_correction_targets_follow_observed_writes_and_resets(self):
        for control in ('carried', 'withhold_update', 'reset_query'):
            batch = make_batch('correction', 'train', 0, 4, gap=16, control=control)
            labels = batch.retained_targets
            self.assertTrue(labels[:, :8].eq(-100).all())
            self.assertTrue(torch.equal(labels[:, 8], batch.old_answers))
            expected = (batch.old_answers if control == 'withhold_update' else
                        torch.full_like(batch.answers, -100) if control == 'reset_query' else batch.answers)
            self.assertTrue(torch.equal(labels[:, -1], expected))
            self.assertTrue(torch.equal(batch.slice(16, 32).retained_targets, labels[:, 16:32]))
        values = torch.tensor([list(b'=A....=C..!')])
        reset = torch.zeros_like(values, dtype=torch.bool); reset[:, 4] = True
        labels = observed_memory_targets(values, reset)
        self.assertEqual(labels.tolist(), [[-100, 65, 65, 65, -100, -100, -100, 67, 67, 67, 67]])

    def test_fact_masks_exclude_hidden_future_private_and_reset_values(self):
        model, _ = model_fixture()
        sources = forward_sources(model.cfg)
        self.assertEqual(sources['P'], {'A', 'B'})
        self.assertEqual(sources['Q'], {'B', 'C'})
        for control in ('all', 'a_only', 'b_only', 'c_only', 'no_private', 'reset_query'):
            batch = make_view_batch('independent', 0, 64, control=control)
            for name, reachable in sources.items():
                labels = view_targets(batch, reachable)
                self.assertTrue(labels[:, :7].eq(-100).all())
                self.assertTrue(torch.equal(view_targets(batch.slice(16, batch.length), reachable), labels[:, 16:]))
                if name == 'B': self.assertTrue(labels[:, :, 1:].eq(-100).all())
                if control == 'b_only': self.assertTrue(labels[:, :, 1:].eq(-100).all())
                if control == 'no_private': self.assertTrue(labels[:, :, 1:].eq(-100).all())
                if control == 'reset_query': self.assertTrue(labels[:, -1].eq(-100).all())
            # Padding at a source must retain already observed facts.
            shared = view_targets(batch, {'B'})[:, :, 0]
            if control == 'b_only': self.assertTrue(shared[:, 13].ne(-100).all())

    def test_capture_and_loss_labels_do_not_change_inference(self):
        model, optimizer = model_fixture()
        before_rng = torch.get_rng_state().clone()
        attach_heads(model, optimizer)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        batch = make_view_batch('train', 0, 4, gap=0)
        _, ordinary = stream_loss(model, batch, None)
        loss, captured = stream_loss(model, batch, None, 1.)
        changed = replace(batch, factors={k: 65+(v-64)%4 for k,v in batch.factors.items()})
        altered_loss, altered = stream_loss(model, changed, None, 1.)
        for result in (captured, altered):
            self.assertTrue(torch.equal(ordinary['logits'], result['logits']))
            assert_tree_equal(self, ordinary['state'].to_dict(), result['state'].to_dict())
            self.assertEqual(ordinary['expert_executions'], result['expert_executions'])
        self.assertNotEqual(float(loss.detach()), float(altered_loss.detach()))
        loss.backward()
        self.assertGreater(float(model.fact_retention.weight.grad.norm()), 0.)
        self.assertGreater(sum(float(p.grad.norm()) for p in model.nodes['A'].parameters() if p.grad is not None), 0.)
        for name in ('source_reconstruction', 'regional_readout', 'memory_retention', 'fact_retention'):
            delattr(model, name)
        _, exported = stream_loss(model, batch, None)
        self.assertTrue(torch.equal(ordinary['logits'], exported['logits']))
        assert_tree_equal(self, ordinary['state'].to_dict(), exported['state'].to_dict())

    def test_empty_targets_do_not_supervise_unobserved_facts(self):
        model, _ = model_fixture(); attach_heads(model)
        batch = make_view_batch('train', 0, 4).slice(0, 7)
        loss, result = stream_loss(model, batch, None, 1.)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(result['retention']['local_labels'], 0)
        self.assertEqual(result['retention']['local_nats'], 0.)
        with self.assertRaises(ValueError): model(batch.inputs, return_node_activity=('missing',))

    def parent(self, path):
        model, optimizer = model_fixture()
        protocol = dict(stream_batch=4, composition_batch=4, segment=16, memory_gap=16,
                        view_gap=8, seed=507, lr=.0005, threads=1, validation_count=64,
                        task_weights={'composition': 1., 'correction': 1., 'views': 1.})
        streams = make_streams(protocol); rng = random.Random(93)
        states, _ = balanced_update(model, optimizer, rng, streams, {k: None for k in streams}, protocol)
        save_integrated(path, model, optimizer, rng, streams, states, 1, protocol, [], {})

    def run_cli(self, parent, out, extras=()):
        args = ['regional', '--parent', str(parent), '--out', str(out), '--steps', '2',
                '--validate-every', '1', '--save-every', '1', *extras]
        with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()), \
             patch('experiments.text_regions.regional_train.evaluate_capabilities', return_value=progress()):
            main()

    def test_resume_preserves_auxiliary_heads_moments_rng_and_streams(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); parent=root/'parent.pt'; self.parent(parent)
            self.run_cli(parent, root/'full')
            self.run_cli(parent, root/'split', ('--stop-after', '1'))
            self.run_cli(parent, root/'split', ('--resume',))
            a=torch.load(root/'full/last.pt', weights_only=True)
            b=torch.load(root/'split/last.pt', weights_only=True)
            for key in ('weights', 'optimizer', 'rng', 'torch_rng', 'states', 'streams'):
                assert_tree_equal(self, a[key], b[key])
            original=torch.load(parent, weights_only=True)
            initial=torch.load(root/'full/initial.pt', weights_only=True)
            for key in ('rng', 'torch_rng', 'states', 'streams'):
                assert_tree_equal(self, original[key], initial[key])
            for key, value in original['weights'].items():
                self.assertTrue(torch.equal(value, initial['weights'][key]))
            model, optimizer, _, _, _=load_integrated(root/'full/last.pt')
            self.assertIn(model.fact_retention.weight, optimizer.state)
            self.assertIn(model.memory_retention.weight, optimizer.state)
            with self.assertRaises(ValueError):
                wrong=dict(a['protocol']); wrong['fact_supervision']=dict(wrong['fact_supervision'], weight=0.)
                load_integrated(root/'full/last.pt', wrong)

    def test_matched_arms_have_identical_examples_costs_initial_losses(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d)/'parent.pt'; self.parent(parent)
            records=[]
            for weight in (0., 1.):
                model, optimizer, rng, (streams, states), saved=load_integrated(parent)
                attach_heads(model, optimizer)
                protocol=dict(saved['protocol'], fact_supervision={'weight': weight})
                states, row=balanced_update(model, optimizer, rng, streams, states, protocol)
                records.append((row, {k:v.to_dict() for k,v in states.items()}, rng.getstate(),
                                {k:v.state_dict() for k,v in streams.items()}))
            for key in ('expert_executions', 'external_events', 'source_occurrences',
                        'composition_nats', 'correction_nats', 'views_nats'):
                self.assertEqual(records[0][0][key], records[1][0][key])
            for i in (1,2,3): assert_tree_equal(self, records[0][i], records[1][i])


if __name__ == '__main__': unittest.main()
