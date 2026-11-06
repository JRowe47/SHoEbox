"""Cross-owner credit for a categorical writer and an unparameterized null."""
import unittest
import torch
from .task_credit import CompetitiveWriteEligibility


class CompetitiveCreditChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(931)
        self.features = torch.zeros(7, 2, 3, 2, dtype=torch.float64)
        self.features[0, :, 1, 1] = 2.
        self.features[1, :, 0, 0] = 2.
        self.features[3, :, 0, 0] = 1.2
        self.features[4, :, 2, 0] = -2.
        self.features[6, :, 0, 0] = 1.5
        self.candidates = torch.randn(7, 2, 3, 4, dtype=torch.float64)
        self.start = torch.randn(2, 3, 4, dtype=torch.float64)
        self.policy = torch.tensor([[1., 0., -.2], [0., 1., -.2], [-1., 0., -.2]], dtype=torch.float64)

    def sequence(self, split=None):
        policy = self.policy.clone().requires_grad_()
        memory = self.start.clone()
        versions = torch.zeros(2, 3, dtype=torch.long)
        trace = CompetitiveWriteEligibility.zeros(2, ('a', 'b', 'c'), 4, 2, dtype=torch.float64, decay=1.)
        receipt = None
        decisions = []
        for t, (features, candidate) in enumerate(zip(self.features, self.candidates)):
            if t == split:
                receipt = trace.capture(memory)
                memory = receipt.memory
            logits = (features*policy[None, :, :-1]).sum(-1)+policy[None, :, -1]
            all_probabilities = torch.cat((logits, torch.zeros(2, 1, dtype=logits.dtype)), -1).softmax(-1)
            p = all_probabilities[:, :3]
            chosen = all_probabilities.argmax(-1)
            action = chosen[:, None].eq(torch.arange(3)[None])
            old = memory
            memory = torch.where(action[..., None], candidate, old)+(p-p.detach())[..., None]*(candidate-old)
            versions = versions+action.long()
            trace = trace.advance(old, candidate, features, p, action, versions)
            decisions.append(chosen.tolist())
        loss = (memory[:, 1].square().sum(-1)+.17*memory[:, 1].sum(-1)).mean()
        loss.backward()
        if receipt is not None:
            before = policy.grad.clone()
            self.assertGreater(receipt.apply(policy), 0.)
            self.assertGreater(float((policy.grad-before).abs().sum()), 0.)
        return policy.grad, trace, receipt, decisions

    def test_full_and_detached_gradients_include_competing_owners(self):
        full, _, _, decisions = self.sequence()
        self.assertEqual([x[0] for x in decisions], [1, 0, 3, 0, 2, 3, 0])
        # Only publication b is read, but the categorical decision trains every
        # head competing with b, including owners whose publications are unread.
        self.assertTrue(bool(full.abs().sum(-1).gt(0).all()))
        for split in (1, 3, 5):
            delayed, trace, _, _ = self.sequence(split)
            torch.testing.assert_close(full, delayed, rtol=1e-12, atol=1e-12)
            self.assertTrue(all(not x.requires_grad and x.grad_fn is None for x in trace.tensors()))

    def test_single_use_receipt_and_disabled_credit(self):
        _, trace, receipt, _ = self.sequence(3)
        with self.assertRaises(RuntimeError):
            receipt.apply(self.policy.clone().requires_grad_())
        receipt = trace.capture(self.start)
        receipt.memory.sum().backward()
        policy = self.policy.clone().requires_grad_()
        self.assertEqual(receipt.apply(policy, enabled=False), 0.)
        self.assertIsNone(policy.grad)

    def test_bootstrap_and_replacement_have_no_fresh_gate_credit(self):
        _, trace, _, _ = self.sequence()
        action = torch.tensor([[True, False, False], [False, False, True]])
        forced = trace.advance(self.start, self.candidates[0], self.features[0],
                               torch.full((2, 3), .2), action, trace.versions+action,
                               eligible=torch.zeros(2, 3, dtype=torch.bool))
        self.assertEqual(float(forced.sensitivity[0, 0].abs().sum()), 0.)
        self.assertEqual(float(forced.sensitivity[1, 2].abs().sum()), 0.)
        torch.testing.assert_close(forced.sensitivity[0, 1:], trace.sensitivity[0, 1:])
        torch.testing.assert_close(forced.sensitivity[1, :2], trace.sensitivity[1, :2])
        blank = CompetitiveWriteEligibility.zeros(2, ('a', 'b', 'c'), 4, 2, dtype=torch.float64)
        bootstrapped = blank.advance(self.start, self.candidates[0], self.features[0],
                                    torch.full((2, 3), .2), action, action.long(),
                                    eligible=torch.zeros(2, 3, dtype=torch.bool))
        self.assertEqual(float(bootstrapped.sensitivity.abs().sum()), 0.)

    def test_output_eligibility_does_not_erase_competing_policy_columns(self):
        policy = self.policy.clone().requires_grad_()
        features, candidate = self.features[0], self.candidates[0]
        logits = (features*policy[None, :, :-1]).sum(-1)+policy[None, :, -1]
        p = torch.cat((logits, torch.zeros(2, 1, dtype=logits.dtype)), -1).softmax(-1)[:, :3]
        action = torch.tensor([[False, True, False]]*2)
        eligible = action.clone()
        old = self.start
        memory = (torch.where(action[..., None], candidate, old)+
                  (p-p.detach())[..., None]*(candidate-old)*eligible[..., None])
        memory[:, 1].square().sum().backward()
        expected = policy.grad.clone()
        self.assertGreater(float(expected[0].abs().sum()), 0.)
        trace = CompetitiveWriteEligibility.zeros(2, ('a', 'b', 'c'), 4, 2, dtype=torch.float64, decay=1.)
        trace = trace.advance(old, candidate, features, p, action, action.long(), eligible=eligible)
        self.assertEqual(float(trace.sensitivity[:, [0, 2]].abs().sum()), 0.)
        receipt = trace.capture(memory)
        receipt.memory[:, 1].square().sum().backward()
        policy.grad = None
        receipt.apply(policy)
        torch.testing.assert_close(policy.grad, expected, rtol=1e-12, atol=1e-12)

    def test_decay_reset_checkpoint_and_bounded_shape(self):
        _, trace, _, _ = self.sequence()
        trace = CompetitiveWriteEligibility(trace.owners, trace.sensitivity, trace.versions, trace.age, .8)
        held = trace.advance(self.start, self.candidates[0], self.features[0], torch.full((2, 3), .2),
                             torch.zeros(2, 3, dtype=torch.bool), trace.versions,
                             eligible=torch.zeros(2, 3, dtype=torch.bool))
        torch.testing.assert_close(held.sensitivity, trace.sensitivity*.8)
        torch.testing.assert_close(held.age, trace.age+1)
        reset = held.reset(torch.tensor([True, False]))
        self.assertEqual(float(reset.sensitivity[0].abs().sum()), 0.)
        self.assertEqual(int(reset.versions[0].sum()), 0)
        restored = CompetitiveWriteEligibility.from_dict(reset.to_dict()).to('cpu')
        self.assertIsInstance(restored, CompetitiveWriteEligibility)
        for actual, expected in zip(restored.tensors(), reset.tensors()):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(restored.storage_bytes(), trace.storage_bytes())
        self.assertEqual(restored.sensitivity.shape, (2, 3, 4, 3, 3))
        with self.assertRaises(ValueError):
            CompetitiveWriteEligibility.from_dict({**reset.to_dict(), 'sensitivity': reset.sensitivity[:, :, :, 0]})

    def test_invalid_probability_and_version_identity_rejected(self):
        trace = CompetitiveWriteEligibility.zeros(2, ('a', 'b', 'c'), 4, 2, dtype=torch.float64)
        for probabilities, actions, versions in (
                (torch.full((2, 3), .4), torch.zeros(2, 3), trace.versions),
                (torch.full((2, 3), .2), torch.ones(2, 3), trace.versions+1),
                (torch.full((2, 3), .2), torch.zeros(2, 3), trace.versions+1)):
            with self.assertRaises(ValueError):
                trace.advance(self.start, self.candidates[0], self.features[0], probabilities, actions, versions)


if __name__ == '__main__':
    unittest.main()
