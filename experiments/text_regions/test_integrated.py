"""Joint learning: one decoder, causal labels, preserved state and exact resume."""
import contextlib
import copy
from dataclasses import replace
import io
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from .composition_train import build_model
from .config import GraphConfig
from .integrated_eval import evaluate_views, selection_key
from .integrated_train import main, stream_loss, joint_update
from . import test_composition_continue as continuation_tests
from .test_probe import assert_tree_equal
from .view_data import make_view_batch

torch.set_num_threads(1)


def progress(minimum=.5, preserved=True, qualified=False):
    return {'gate_failures': [] if qualified else ['left'],
            'composition_preserved': preserved, 'minimum_accuracy': minimum,
            'macro_accuracy': minimum, 'macro_nats': 1.-minimum,
            'scores': {'composition': .95, 'left': minimum}}


class IntegrationChecks(unittest.TestCase):
    def test_labels_never_change_logits_or_state(self):
        torch.manual_seed(801)
        model = build_model(GraphConfig.tiny(readout_mode='concat', rounds=2,
                                            recurrent_feedback=True), .5, 1.)
        batch = make_view_batch('train', 0, 4, gap=0)
        changed = replace(batch, targets=torch.where(batch.targets.ne(-100),
                                                     65+(batch.targets-64)%4, -100))
        _, a = stream_loss(model, batch, None)
        _, b = stream_loss(model, changed, None)
        assert_tree_equal(self, a['state'].to_dict(), b['state'].to_dict())
        self.assertTrue(torch.equal(a['logits'], b['logits']))
        self.assertFalse(torch.equal(a['ce'], b['ce']))

    def test_selection_requires_composition_and_does_not_hide_a_weak_task(self):
        self.assertGreater(selection_key(progress(.5, True)), selection_key(progress(.9, False)))
        self.assertGreater(selection_key(progress(.7)), selection_key(progress(.6)))
        self.assertGreater(selection_key(progress(.9, True, True)), selection_key(progress(.95)))

    def run_cli(self, parent, out, extras=(), scores=None):
        args = ['integrated', '--parent', str(parent), '--out', str(out), '--steps', '2',
                '--stream-batch', '4', '--composition-batch', '4', '--validate-every', '1',
                '--save-every', '1', *extras]
        with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
            with patch('experiments.text_regions.integrated_train.evaluate_capabilities',
                       side_effect=scores or (lambda *a, **kw: progress())):
                main()

    def make_parent(self, path):
        # Reuse the exercised composition parent format; set inference defaults
        # to the two-round execution already used by its training objective.
        continuation_tests.ContinuationChecks().make_parent(path)
        data = torch.load(path, weights_only=True)
        data['config'].update(rounds=2, recurrent_feedback=True)
        torch.save(data, path)

    def test_joint_cli_exact_resume_including_both_runtime_states(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); parent = root/'parent.pt'; self.make_parent(parent)
            self.run_cli(parent, root/'full')
            self.run_cli(parent, root/'split', ('--stop-after', '1'))
            self.run_cli(parent, root/'split', ('--resume',))
            a = torch.load(root/'full/last.pt', weights_only=True)
            b = torch.load(root/'split/last.pt', weights_only=True)
            for key in ('weights', 'optimizer', 'rng', 'torch_rng', 'states', 'streams'):
                assert_tree_equal(self, a[key], b[key])
            for key in a['totals']:
                if key != 'training_seconds': self.assertEqual(a['totals'][key], b['totals'][key])
            self.assertNotEqual(a['states']['correction']['input_mode'], a['states']['views']['input_mode'])
            initial = torch.load(root/'full/initial.pt', weights_only=True)
            old = torch.load(parent, weights_only=True)
            for key in ('weights', 'optimizer', 'rng', 'torch_rng'):
                assert_tree_equal(self, old[key], initial[key])
            self.assertEqual(a['totals']['composition_episodes'], 8)
            self.assertEqual(a['totals']['views_final_queries'], 4)
            self.assertEqual(a['totals']['correction_final_queries'], 0)

    def test_regressed_model_does_not_replace_preserved_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); parent = root/'parent.pt'; self.make_parent(parent)
            self.run_cli(parent, root/'run', scores=[progress(.3), progress(.5), progress(.9, False)])
            selected = torch.load(root/'run/best.pt', weights_only=True)
            self.assertEqual(selected['step'], 1)
            with self.assertRaises(ValueError):
                from .integrated_train import load_integrated
                bad = copy.deepcopy(selected['protocol']); bad['lr'] *= 2
                load_integrated(root/'run/best.pt', bad)

    def test_view_evaluation_masks_preserve_work_and_use_256_logits(self):
        torch.manual_seed(802)
        model = build_model(GraphConfig.tiny(readout_mode='concat', rounds=2,
                                            recurrent_feedback=True), .5, 1.)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        results = evaluate_views(model, 'independent', 4, 4, gap=0,
                                  controls=('all', 'b_only', 'reset_query'))
        self.assertTrue(model.training)
        for query in ('shared', 'left', 'right'):
            rows = [results[control][query] for control in results]
            self.assertEqual(len({r['expert_executions'] for r in rows}), 1)
            self.assertEqual(len({tuple(r['episode_ids']) for r in rows}), 1)
            self.assertEqual(rows[0]['episodes'], 4)
            self.assertEqual(rows[0]['majority_class_chance'], .25)
        assert_tree_equal(self, before, model.state_dict())



class BalancedIntegrationChecks(unittest.TestCase):
    def fixture(self, root):
        parent = root/'parent.pt'; IntegrationChecks().make_parent(parent)
        protocol = dict(stream_batch=4, composition_batch=4, segment=16, memory_gap=16,
                        view_gap=8, seed=507, lr=.0005, threads=1, validation_count=64,
                        validate_every=1, save_every=1,
                        task_weights={'composition': 1., 'correction': 1., 'views': 1.},
                        parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest())
        path = root/'baseline.json'; path.write_text(json.dumps(protocol))
        return parent, path, protocol

    def test_gradient_cap_preserves_examples_costs_losses_and_forward_states(self):
        from scripts.train_integrated_balanced import balanced_update
        from .integrated_train import make_streams, joint_update
        from .composition_train import load_checkpoint
        with tempfile.TemporaryDirectory() as d:
            parent, _, protocol = self.fixture(Path(d))
            records = []
            for update in (joint_update, balanced_update):
                model, opt, rng, _ = load_checkpoint(parent)
                streams = make_streams(protocol)
                states, row = update(model, opt, rng, streams, {k: None for k in streams}, protocol)
                records.append((row, {k: v.to_dict() for k, v in states.items()}))
            for key in ('expert_executions', 'source_occurrences', 'external_events',
                        'composition_nats', 'correction_nats', 'views_nats'):
                self.assertEqual(records[0][0][key], records[1][0][key])
            assert_tree_equal(self, records[0][1], records[1][1])
            self.assertLessEqual(records[1][0]['gradient_norm'], 1.000001)
            self.assertTrue(all(records[1][0][name+'_gradient_norm'] > 0
                                for name in ('composition', 'correction', 'views')))

    def test_long_view_stream_preserves_reference_values_and_gradients(self):
        from .integrated_train import make_streams
        from .composition_train import load_checkpoint
        with tempfile.TemporaryDirectory() as d:
            parent, _, protocol = self.fixture(Path(d))
            chunk = make_streams(protocol)['views'].next_chunk(protocol['segment'])
            for dtype in (torch.float32, torch.float64):
                with self.subTest(dtype=dtype):
                    snapshots = []
                    for backend in ('reference', 'optimized'):
                        model, _, _, _ = load_checkpoint(parent)
                        model.to(dtype).set_execution_backend(backend)
                        loss, result = stream_loss(model, chunk, None)
                        loss.backward()
                        snapshots.append((float(loss.detach()), result['logits'].detach(),
                            result['state'].detach().to_dict(),
                            {name: None if p.grad is None else p.grad.detach().double().clone()
                             for name, p in model.named_parameters()}))
                    reference, actual = snapshots
                    self.assertEqual(reference[0], actual[0])
                    torch.testing.assert_close(reference[1], actual[1], rtol=0, atol=0)
                    assert_tree_equal(self, reference[2], actual[2])
                    self.assertEqual(reference[3].keys(), actual[3].keys())
                    error2 = norm2 = 0.
                    for name, expected in reference[3].items():
                        gradient = actual[3][name]
                        if expected is None:
                            self.assertIsNone(gradient, name)
                            continue
                        self.assertIsNotNone(gradient, name)
                        self.assertTrue(bool(torch.isfinite(expected).all()), name)
                        self.assertTrue(bool(torch.isfinite(gradient).all()), name)
                        error2 += float((gradient-expected).square().sum())
                        norm2 += float(expected.square().sum())
                    limit = 1e-10 if dtype == torch.float64 else 5e-6
                    self.assertLessEqual((error2/max(norm2, 1e-30))**.5, limit)

    def test_balanced_cli_resume_preserves_weights_moments_rng_and_both_streams(self):
        from scripts.train_integrated_balanced import main as balanced_main
        def run(parent, protocol, out, extra=()):
            args = ['balanced', '--parent', str(parent), '--baseline-protocol', str(protocol),
                    '--out', str(out), '--steps', '2', *extra]
            with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
                with patch('scripts.train_integrated_balanced.evaluate_capabilities',
                           side_effect=lambda *a, **kw: progress()): balanced_main()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); parent, protocol, _ = self.fixture(root)
            run(parent, protocol, root/'full')
            run(parent, protocol, root/'split', ('--stop-after', '1'))
            run(parent, protocol, root/'split', ('--resume',))
            a = torch.load(root/'full/last.pt', weights_only=True)
            b = torch.load(root/'split/last.pt', weights_only=True)
            for key in ('weights', 'optimizer', 'rng', 'torch_rng', 'states', 'streams'):
                assert_tree_equal(self, a[key], b[key])
            self.assertIn('../scripts/train_integrated_balanced.py', a['protocol']['source_hashes'])
            self.assertIn('Clip each', a['protocol']['gradient_policy'])
            initial = torch.load(root/'full/initial.pt', weights_only=True)
            old = torch.load(parent, weights_only=True)
            for key in ('weights', 'optimizer', 'rng', 'torch_rng'):
                assert_tree_equal(self, old[key], initial[key])


class JointGradientClippingChecks(unittest.TestCase):
    @contextlib.contextmanager
    def update_with_gradient(self, gradient):
        # Isolate the final joint clip from expensive data/model execution.
        # Inject after a finite loss backward so NaN/Inf tests reach the clip.
        model = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        model.weight.register_hook(lambda _: gradient.reshape_as(model.weight).clone())
        optimizer = torch.optim.SGD(model.parameters(), lr=.25)
        batch = SimpleNamespace(size=1, answers=torch.zeros(1, dtype=torch.long))
        counts = dict(expert_executions=1, source_occurrences=1, external_events=1)

        def loss(*args, **kwargs):
            value = model.weight.sum()
            return value, value.detach(), counts, dict(logits=torch.zeros(1, 1, 1), **counts)

        protocol = dict(lr=.25, composition_batch=1, task_weights={'composition': 1.})
        with patch('experiments.text_regions.integrated_train.training_batch', return_value=batch), \
             patch('experiments.text_regions.integrated_train.training_loss', side_effect=loss), \
             patch.object(optimizer, 'step', wraps=optimizer.step) as step:
            yield lambda: joint_update(model, optimizer, None, {}, {}, protocol), model.weight, step

    def test_joint_clip_handles_finite_fp32_norm_overflow(self):
        gradient = torch.tensor([3e20, 4e20])
        self.assertFalse(bool(torch.isfinite(gradient.norm())))
        with self.update_with_gradient(gradient) as (update, parameter, step):
            _, row = update()
            self.assertAlmostEqual(row['gradient_norm']/5e20, 1., places=6)
            torch.testing.assert_close(parameter.grad, torch.tensor([[.6, .8]]))
            torch.testing.assert_close(parameter, torch.tensor([[-.15, -.2]]))
            self.assertLessEqual(float(parameter.grad.double().norm()), 1.000001)
            step.assert_called_once()

    def test_joint_clip_preserves_ordinary_clipping_exactly(self):
        for values in ([.03, .04], [3., 4.], [0., 0.]):
            with self.subTest(values=values):
                reference = torch.nn.Parameter(torch.zeros(1, 2))
                reference.grad = torch.tensor([values])
                norm = torch.nn.utils.clip_grad_norm_([reference], 1., error_if_nonfinite=True)
                with self.update_with_gradient(torch.tensor(values)) as (update, parameter, step):
                    _, row = update()
                    self.assertEqual(row['gradient_norm'], float(norm))
                    self.assertTrue(torch.equal(parameter.grad, reference.grad))
                    self.assertTrue(torch.equal(parameter.detach(), -.25*reference.grad))
                    step.assert_called_once()

    def test_joint_clip_rejects_nonfinite_before_optimizer_step(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                gradient = torch.tensor([value, 1.])
                with self.update_with_gradient(gradient) as (update, parameter, step):
                    with self.assertRaises(RuntimeError):
                        update()
                    step.assert_not_called()
                    self.assertTrue(torch.equal(parameter.detach(), torch.zeros_like(parameter)))
                    torch.testing.assert_close(parameter.grad, gradient[None],
                                               equal_nan=True, rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
