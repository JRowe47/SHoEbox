"""Recall capacity, causal ordering, trainable controls and exact continuation."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F

from .ports import tensor_list
from .task_memory import TaskMemoryModel, TaskMemoryState, TaskMemoryConfig
from .task_memory_train import initialize_weights
from .task_payload import make_eligibility
from .task_query_audit import select_state, concatenate_states, query_forks
from .task_recall import RecallWorkspace
from .test_task_retention_integration import model_config

torch.set_num_threads(1)


class RecallChecks(unittest.TestCase):
    def same(self, left, right):
        a, b = tensor_list(left), tensor_list(right)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def model(self, **options):
        torch.manual_seed(52)
        return TaskMemoryModel(model_config(recall_slots=2, payload_rank=4, **options))

    @torch.no_grad()
    def test_seven_slots_choose_only_valid_items_or_null_and_block_exactly(self):
        torch.manual_seed(73)
        module = RecallWorkspace(8, 4, 3, 7, 2)
        memory = torch.randn(3, 9, 12)
        keys = F.normalize(torch.randn(3, 9, 3), dim=-1)
        query = torch.randn(3, 3)
        valid = torch.zeros(3, 9, dtype=torch.bool)
        valid[1, 3] = True; valid[2] = True
        feedback = torch.randn(3, 8)
        result = module.select(query, keys, memory, valid, torch.zeros(3, 1), feedback)
        weights = result['weights']
        self.assertEqual(tuple(weights.shape), (3, 7, 10))
        self.assertTrue(bool(((weights == 0) | (weights == 1)).all()))
        self.same(weights.sum(-1), torch.ones(3, 7))
        self.assertEqual(float((weights[..., :-1] * ~valid[:, None]).sum()), 0.)
        self.assertEqual(float(weights[0, :, -1].sum()), 7.)
        self.assertTrue(bool((weights[..., :-1].sum((1, 2)) <= 7).all()))
        blocked = module.select(query, keys, memory, valid, torch.zeros(3, 1), feedback, blocked=True)
        self.assertEqual(int(blocked['contents'].count_nonzero()), 0)
        self.same(result['weights'], blocked['weights'])
        soft = module.select(query, keys, memory, valid, torch.zeros(3, 1), feedback, hard=False)
        torch.testing.assert_close(soft['weights'].sum(-1), torch.ones(3, 7))
        self.assertEqual(float((soft['weights'][..., :-1] * ~valid[:, None]).sum()), 0.)

    def test_hard_selection_has_analytic_softmax_null_derivative(self):
        module = RecallWorkspace(2, 1, 2, 1, 2).double()
        memory = torch.tensor([[[2., 4., 1.], [5., 3., 2.]]], dtype=torch.double, requires_grad=True)
        result = module.select(torch.tensor([[1., 0.]], dtype=torch.double),
            torch.tensor([[[1., 0.], [0., 1.]]], dtype=torch.double), memory,
            torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.double),
            torch.zeros(1, 2, dtype=torch.double))
        result['contents'].sum().backward()
        p = result['probabilities'].detach()[0, 0]
        expected = -p[-1] * (p[:-1] * memory.detach().sum(-1)[0]).sum()
        torch.testing.assert_close(module.slot_null.grad[0], expected, rtol=1e-12, atol=1e-12)
        self.same(memory.grad, result['weights'].detach()[..., :-1].transpose(1, 2).expand_as(memory))

    def test_distinct_slot_maps_preserve_role_that_pooled_control_erases(self):
        module = RecallWorkspace(2, 1, 2, 2, 2)
        with torch.no_grad():
            module.payload_merge.weight.copy_(torch.cat((torch.eye(2), -torch.eye(2)), -1))
        payload = torch.tensor([[[1., 2.], [3., 5.]]])
        semantic = torch.ones(1, 2, 1)
        separate, _ = module.compose(payload, semantic)
        swapped, _ = module.compose(payload.flip(1), semantic)
        pooled, _ = module.compose(payload, semantic, 'pooled')
        self.same(separate, torch.tensor([[-2., -3.]]))
        self.same(swapped, -separate)
        self.same(pooled, torch.zeros_like(pooled))

    def test_common_initialization_legacy_state_and_parent_migration(self):
        torch.manual_seed(52)
        legacy = TaskMemoryModel(model_config(payload_rank=4))
        rng = torch.get_rng_state().clone()
        model = self.model()
        self.same(torch.get_rng_state(), rng)
        for key, value in legacy.state_dict().items():
            self.same(value, model.state_dict()[key])
        old_config = legacy.cfg.to_dict()
        for name in ('recall_slots', 'recall_layout', 'recall_feedback', 'recall_hard'):
            old_config.pop(name)
        self.assertEqual(TaskMemoryConfig.from_dict(old_config).recall_slots, 0)
        state = legacy.initial_state(2).to_dict(); state.pop('recall_feedback')
        self.assertIsNone(TaskMemoryState.from_dict(state).recall_feedback)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'parent.pt'
            torch.save(dict(weights=legacy.state_dict(), step=600), path)
            initialized = initialize_weights(model, path)
            self.assertTrue(all(n.startswith('recall.') for n in initialized['new_parameters']))
        result = model(torch.tensor([[65], [66]]), state=TaskMemoryState.from_dict(state))
        self.assertEqual(tuple(result['state'].recall_feedback.shape), (2, 32))

    def test_read_before_write_feedback_gradients_and_delayed_credit(self):
        model = self.model()
        x = torch.tensor([list(b'A x=1;B x=2;A x?'), list(b'A x=2;B x=3;A x?')])
        result = model(x)
        self.assertTrue(bool(result['recall_weights'][:, 0, :, -1].eq(1).all()))
        self.assertTrue(bool(result['memory_valid'][:, 0].any(-1).all()))
        result['logits'].square().sum().backward()
        for name in ('slot_queries', 'slot_null', 'coverage_logit', 'query_feedback.weight',
                     'null_feedback.weight', 'usage_feedback.weight', 'payload_merge.weight',
                     'semantic_merge.weight'):
            gradient = dict(model.recall.named_parameters())[name].grad
            self.assertTrue(bool(torch.isfinite(gradient).all()), name)
            self.assertGreater(float(gradient.norm()), 1e-10, name)
        credit = make_eligibility(model, 2).advance_chunk(result)
        receipt = credit.capture(result['state'].memory_tensor())
        state = result['state'].detach().with_memory_tensor(receipt.memory)
        model.zero_grad(set_to_none=True)
        final = model(x[:, -3:], state=state, write_override='hold')
        final['logits'].square().sum().backward()
        self.assertGreater(float(receipt.memory.grad.norm()), 0.)
        self.assertGreater(receipt.apply(model.write_policy), 0.)

    @torch.no_grad()
    def test_feedback_roundtrip_row_reset_and_query_fork_alignment(self):
        model = self.model()
        x = torch.tensor([list(b'A x=1;B x=2;'), list(b'A x=2;B x=3;')])
        result = model(x)
        state = result['state'].detach()
        self.same(state, TaskMemoryState.from_dict(state.to_dict()))
        reset = state.reset(torch.tensor([True, False]))
        self.same(reset.recall_feedback[0], torch.zeros(32))
        self.same(reset.recall_feedback[1], state.recall_feedback[1])
        rebuilt = concatenate_states([select_state(state, [0]), select_state(state, [1])])
        self.same(state, rebuilt)
        query = torch.tensor([list(b'A x?'), list(b'B x?')])
        direct = model(query, state=state)
        config = model.cfg
        forks = query_forks(model, query, state, donors=[1, 0], age_threshold=2, include_recall=True)
        self.assertIs(model.cfg, config)
        self.assertIn('recall_pooled', forks)
        self.assertIn('recall_no_feedback', forks)
        self.same(direct['logits'], forks['intact']['result']['logits'])
        self.same(state, result['state'])
        self.assertIn('recall_weights', forks['intact']['trace'][0][0])
        self.same(direct['logits'], model(query, state=state, capture=False)['logits'])

    @torch.no_grad()
    def test_feedback_ablation_and_seven_slot_model_are_operational(self):
        model = self.model()
        with torch.no_grad():
            model.recall.query_feedback.weight.normal_(std=.4)
            model.recall.null_feedback.weight.normal_(std=.4)
            model.recall.usage_feedback.weight.normal_(std=.4)
        inputs = torch.tensor([list(b'A x=1;B x=2;'), list(b'A x=2;B x=3;')])
        prefix = model(inputs)['state']
        active = model(inputs, state=prefix)
        model.cfg = replace(model.cfg, recall_feedback=False)
        disabled = model(inputs, state=prefix)
        self.assertFalse(torch.equal(active['logits'], disabled['logits']))
        seven = TaskMemoryModel(model_config(recall_slots=7, owners=9))
        result = seven(inputs)
        self.assertEqual(tuple(result['recall_weights'].shape), (2, 12, 7, 10))
        self.assertEqual(result['expert_executions'], 2*12*10*3*2*2)
        self.assertTrue(bool(torch.isfinite(result['logits']).all()))

    def test_exact_resume_with_feedback_holds_and_stage_transition(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            common = [sys.executable, '-m', 'experiments.text_regions.task_memory_train',
                '--stream-preset', 'retention', '--curriculum', 'acquisition',
                '--acquire-steps', '2', '--bridge-steps', '2', '--truncate-every', '8',
                '--write-mode', 'competitive', '--memory-aware-write', '--bootstrap-empty',
                '--payload-rank', '4', '--payload-credit', '--state-commit', 'publication',
                '--recall-slots', '2', '--width', '32', '--batch', '2', '--context', '16',
                '--eval-count', '2', '--eval-every', '2', '--seed', '9', '--data-seed', '42']
            for name, steps, extra in [('full', 5, []), ('split', 3, []),
                ('resumed', 5, ['--resume', str(root/'split/last.pt')])]:
                run = subprocess.run(common + ['--out', str(root/name), '--steps', str(steps)] + extra,
                    capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
            full, resumed = [torch.load(root/name/'last.pt', weights_only=False) for name in ('full', 'resumed')]
            for key in ('weights', 'optimizer', 'state', 'credit', 'rng'):
                self.same(full[key], resumed[key])
            self.assertEqual(full['stream'], resumed['stream'])
            self.assertEqual(full['totals']['stages'], resumed['totals']['stages'])
            run = subprocess.run(common + ['--out', str(root/'resumed'), '--steps', '6',
                '--resume', str(root/'resumed/last.pt'), '--no-recall-feedback'], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('Resume configuration mismatch', run.stderr)


if __name__ == '__main__':
    unittest.main()
