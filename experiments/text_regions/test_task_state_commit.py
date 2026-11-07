"""Owner-state protection, live surrogate gradients and exact continuation."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import subprocess
import sys
import tempfile
import unittest

import torch

from .ports import tensor_list
from .state import NodeState
from .task_memory import TaskMemoryConfig, TaskMemoryModel
from .task_state_commit import commit_producer_state
from .test_task_retention_integration import model_config

torch.set_num_threads(1)


class StateCommitChecks(unittest.TestCase):
    def model(self, mode='publication'):
        torch.manual_seed(73)
        return TaskMemoryModel(model_config(state_commit=mode, payload_rank=4))

    def same(self, a, b):
        left, right = tensor_list(a), tensor_list(b)
        self.assertEqual(len(left), len(right))
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    @torch.no_grad()
    def test_legacy_config_initialization_and_all_commit_equivalence(self):
        old = self.model('always')
        rng = torch.get_rng_state().clone()
        held = self.model()
        self.same(rng, torch.get_rng_state())
        self.same(old.state_dict(), held.state_dict())
        legacy = old.cfg.to_dict(); legacy.pop('state_commit')
        self.assertEqual(TaskMemoryConfig.from_dict(legacy).state_commit, 'always')
        with self.assertRaises(ValueError):
            TaskMemoryModel(replace(old.cfg, state_commit='invalid'))
        x = torch.tensor([list(b'A x=2;A x?'), list(b'A x=3;A x?')])
        a, b = old(x, write_override='always'), held(x, write_override='always')
        self.same(a['logits'], b['logits'])
        self.same(a['state'], b['state'])
        self.assertEqual(a['expert_executions'], b['expert_executions'])
        self.assertEqual(a['producer_candidates'], b['producer_candidates'])
        self.assertEqual(a['producer_commits'], b['producer_commits'])

    @torch.no_grad()
    def test_hold_protects_every_content_tensor_but_context_and_time_advance(self):
        model = self.model()
        prefix = model(torch.tensor([[65, 66, 67], [68, 69, 70]]), write_override='always')['state'].detach()
        result = model(torch.tensor([list(b'1234;A x?'), list(b'4321;B x?')]),
                       state=prefix, write_override='hold')
        current = result['state']
        self.same(prefix.producer.contextualizer, current.producer.contextualizer)
        self.same(prefix.producer.blocks, current.producer.blocks)
        self.same(prefix.publications, current.publications)
        self.same(current.events_seen, prefix.events_seen + 9)
        self.same(current.producer.clock, prefix.producer.clock + 9)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
            tensor_list(prefix.ingress), tensor_list(current.ingress))))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
            tensor_list(prefix.composer), tensor_list(current.composer))))
        self.assertFalse(bool(result['state_actions'].any()))
        self.assertEqual(result['producer_commits'], 0)
        self.assertEqual(result['producer_candidates'], 2 * 9 * model.cfg.owners)

    @torch.no_grad()
    def test_selected_owner_commits_and_reads_do_not_grant_write_permission(self):
        model, baseline = self.model(), self.model('always')
        with torch.no_grad():
            for item in (model, baseline):
                item.write_policy.zero_(); item.write_policy[1, -1] = 2
        initial = model(torch.tensor([[65], [66]]), write_override='always')['state'].detach()
        x = torch.tensor([[67], [68]])
        a, b = model(x, state=initial, control='fresh'), baseline(x, state=initial, control='fresh')
        self.same(a['logits'], b['logits'])  # The commit affects only later events.
        mask = a['state_actions'][:, 0].flatten()
        self.same(a['state_actions'], a['write_actions'])
        self.assertEqual(mask.tolist(), [False, True, False] * 2)
        for before, candidate, after in zip(tensor_list(initial.producer)[:-1],
                                            tensor_list(b['state'].producer)[:-1],
                                            tensor_list(a['state'].producer)[:-1]):
            self.same(after[mask], candidate[mask])
            self.same(after[~mask], before[~mask])
        self.assertTrue(bool(a['read_valid'].all()))
        self.assertEqual(a['producer_commits'], 2)

    @torch.no_grad()
    def test_reset_clears_only_requested_rows_even_during_hold(self):
        model = self.model()
        prefix = model(torch.tensor([[65, 66], [67, 68]]), write_override='always')['state'].detach()
        result = model(torch.tensor([[69], [70]]), state=prefix,
                       reset=torch.tensor([True, False]), write_override='hold')['state']
        for a, b in zip(tensor_list(prefix.producer)[:-1], tensor_list(result.producer)[:-1]):
            self.assertEqual(int(b[:3].count_nonzero()), 0)
            self.same(a[3:], b[3:])
        self.assertEqual(result.events_seen.tolist(), [1, 3])
        self.assertEqual(result.producer.clock.tolist(), [1, 1, 1, 3, 3, 3])
        self.assertEqual(result.epochs.tolist(), [1, 0])

    def test_surrogate_matches_independent_softmax_derivative_and_content_gradients(self):
        torch.manual_seed(19)
        old = torch.randn(6, 2, 3, dtype=torch.float64, requires_grad=True)
        new = torch.randn_like(old, requires_grad=True)
        coefficient = torch.randn_like(old)
        logits = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        probs = logits.softmax(-1)[:, :3]
        actions = torch.tensor([[False, True, False], [True, False, False]])
        eligible = torch.tensor([[True, True, True], [False, False, False]])
        previous = NodeState((old,), (), torch.zeros(6, dtype=torch.long))
        candidate = NodeState((new,), (), torch.ones(6, dtype=torch.long))
        actual = commit_producer_state(previous, candidate, actions, probs, eligible)
        loss = (actual.contextualizer[0] * coefficient).sum()
        actual_grad = torch.autograd.grad(loss, (logits, old, new))
        gain = ((new.detach() - old.detach()) * coefficient).sum((1, 2)).reshape(2, 3) * eligible
        all_gain = torch.cat((gain, torch.zeros(2, 1, dtype=torch.float64)), -1)
        p = logits.detach().softmax(-1)
        expected = p * (all_gain - (p * all_gain).sum(-1, keepdim=True))
        torch.testing.assert_close(actual_grad[0], expected, rtol=1e-12, atol=1e-12)
        self.same(actual_grad[1], coefficient * (~actions).reshape(6, 1, 1))
        self.same(actual_grad[2], coefficient * actions.reshape(6, 1, 1))
        self.same(actual.contextualizer[0], torch.where(actions.reshape(6, 1, 1), new, old))

    def test_live_model_state_path_trains_the_admission_policy(self):
        model = self.model()
        with torch.no_grad():
            state = model(torch.tensor([[65], [66]]), write_override='always')['state']
            model.write_policy.zero_(); model.write_policy[:, -1] = -2
        gradients, states = [], []
        for enabled in (True, False):
            model.zero_grad(set_to_none=True)
            def conditional_commit(previous, candidate, actions, probabilities, eligible):
                return commit_producer_state(previous, candidate, actions,
                                             probabilities if enabled else None, eligible)
            with patch('experiments.text_regions.task_memory.commit_producer_state', conditional_commit):
                result = model(torch.tensor([[67, 68], [69, 70]]), state=state)
            self.assertFalse(bool(result['write_actions'].any()))
            contents = (*result['state'].producer.contextualizer, *result['state'].producer.blocks)
            sum(value.square().sum() for value in contents).backward()
            gradients.append(None if model.write_policy.grad is None else model.write_policy.grad.clone())
            states.append(result['state'].detach())
        self.same(states[0], states[1])
        self.assertGreater(float(gradients[0].norm()), 1e-8)
        self.assertTrue(gradients[1] is None or not bool(gradients[1].count_nonzero()))

    def test_resume_with_holds_payload_credit_and_stage_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = [sys.executable, '-m', 'experiments.text_regions.task_memory_train',
                      '--stream-preset', 'retention', '--curriculum', 'acquisition',
                      '--acquire-steps', '2', '--bridge-steps', '2', '--truncate-every', '8',
                      '--write-mode', 'competitive', '--memory-aware-write', '--bootstrap-empty',
                      '--payload-rank', '4', '--payload-credit', '--state-commit', 'publication',
                      '--width', '32', '--batch', '2', '--context', '16', '--eval-count', '2',
                      '--eval-every', '2', '--seed', '9', '--data-seed', '42']
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
            totals = full['totals']['stages']
            self.assertTrue(all(0 < v['producer_commits'] <= v['producer_candidates'] / 3 for v in totals.values()))
            invalid = [x if x != 'publication' else 'always' for x in common]
            run = subprocess.run(invalid + ['--out', str(root/'resumed'), '--steps', '6',
                                 '--resume', str(root/'resumed/last.pt')], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('Resume configuration mismatch', run.stderr)


if __name__ == '__main__':
    unittest.main()
