"""Conditional delayed gate derivatives, owner identity and bounded storage."""
import unittest
import torch
from .task_credit import WriteEligibility, matched_read_advantage


class TaskCreditChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(912)
        self.features = torch.randn(7, 2, 3, 2, dtype=torch.float64)
        self.features[0, :, 1] = torch.tensor([-1., 1.], dtype=torch.float64)
        self.features[1:, :, 1] = torch.tensor([1., -1.], dtype=torch.float64)
        self.candidates = torch.randn(7, 2, 3, 4, dtype=torch.float64)
        self.start = torch.randn(2, 3, 4, dtype=torch.float64)
        self.initial_policy = torch.tensor([[.4, -.3, .2], [-.2, .5, -.1], [.1, -.4, .3]], dtype=torch.float64)

    def run_sequence(self, split=None):
        policy = self.initial_policy.clone().requires_grad_()
        memory = self.start.clone()
        versions = torch.zeros(2, 3, dtype=torch.long)
        trace = WriteEligibility.zeros(2, ('a', 'b', 'c'), 4, 2, dtype=torch.float64, decay=1.)
        boundary = None
        for t, (features, candidate) in enumerate(zip(self.features, self.candidates)):
            if t == split:
                boundary = trace.capture(memory)
                memory = boundary.memory
            p = (features*policy[None, :, :-1]).sum(-1).add(policy[None, :, -1]).sigmoid()
            action = p.ge(.5).to(p.dtype)
            old = memory
            # Exact forward hold/replacement, ST derivative only for the gate.
            memory = torch.where(action[..., None].bool(), candidate, old)+(p-p.detach())[..., None]*(candidate-old)
            versions = versions+action.long()
            trace = trace.advance(old, candidate, features, p, action, versions)
        # Context selects only owner b: unselected write-policy heads receive zero.
        loss = (memory[:, 1].square().sum(-1)+.17*memory[:, 1].sum(-1)).mean()
        loss.backward()
        if boundary is not None:
            before = policy.grad.clone()
            boundary.apply(policy)
            self.assertGreater(float((policy.grad-before).abs().sum()), 0.)
        return policy.grad, trace, boundary

    def test_full_and_split_conditional_st_gradient_agree(self):
        full, _, _ = self.run_sequence()
        for split in (1, 3, 5):
            delayed, trace, _ = self.run_sequence(split)
            torch.testing.assert_close(full, delayed, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(delayed[[0, 2]], torch.zeros_like(delayed[[0, 2]]))
            self.assertTrue(all(not x.requires_grad and x.grad_fn is None for x in trace.tensors()))

    def test_receipt_is_single_use_and_disabled_path_leaves_gradient_unchanged(self):
        _, _, receipt = self.run_sequence(3)
        with self.assertRaises(RuntimeError):
            receipt.apply(self.initial_policy.clone().requires_grad_())
        trace = WriteEligibility.zeros(2, ('a', 'b', 'c'), 4, 2)
        receipt = trace.capture(self.start.float())
        receipt.memory.sum().backward()
        policy = self.initial_policy.float().clone().requires_grad_()
        self.assertEqual(receipt.apply(policy, enabled=False), 0.)
        self.assertIsNone(policy.grad)

    def test_forced_replacement_invalidates_prior_responsibility(self):
        _, trace, _ = self.run_sequence()
        shape = (2, 3)
        trace = trace.advance(self.start, self.candidates[0], self.features[0],
                              torch.full(shape, .5), torch.ones(shape), trace.versions+1,
                              eligible=torch.zeros(shape, dtype=torch.bool))
        self.assertEqual(float(trace.sensitivity.abs().sum()), 0.)
        self.assertEqual(int(trace.age.sum()), 0)

    def test_hold_decay_reset_and_serialization(self):
        _, trace, _ = self.run_sequence()
        trace = WriteEligibility(trace.owners, trace.sensitivity, trace.versions, trace.age, .8)
        held = trace.advance(self.start, self.candidates[0], self.features[0],
                             torch.full((2, 3), .5), torch.zeros(2, 3), trace.versions,
                             eligible=torch.zeros(2, 3, dtype=torch.bool))
        torch.testing.assert_close(held.sensitivity, trace.sensitivity*.8)
        torch.testing.assert_close(held.versions, trace.versions)
        torch.testing.assert_close(held.age, trace.age+1)
        reset = held.reset(torch.tensor([True, False]))
        self.assertEqual(float(reset.sensitivity[0].abs().sum()), 0.)
        self.assertEqual(int(reset.versions[0].sum()), 0)
        self.assertEqual(int(reset.age[0].sum()), 0)
        torch.testing.assert_close(reset.sensitivity[1], held.sensitivity[1])
        restored = WriteEligibility.from_dict(reset.to_dict())
        for expected, actual in zip(reset.tensors(), restored.tensors()):
            torch.testing.assert_close(expected, actual)
        self.assertEqual(restored.storage_bytes(), reset.storage_bytes())
        self.assertEqual(restored.owners, reset.owners)

    def test_mismatched_write_identity_rejected(self):
        trace = WriteEligibility.zeros(2, ('a', 'b', 'c'), 4, 2, dtype=torch.float64)
        with self.assertRaises(ValueError):
            trace.advance(self.start, self.candidates[0], self.features[0],
                          torch.full((2, 3), .5), torch.ones(2, 3), trace.versions)

    def test_matched_read_advantage_masks_unscored_positions(self):
        live = torch.tensor([[2., 9., 1.]], requires_grad=True)
        blocked = torch.tensor([[3., 1., 4.]], requires_grad=True)
        gain = matched_read_advantage(live, blocked, torch.tensor([[True, False, True]]))
        torch.testing.assert_close(gain, torch.tensor([[1., 0., 3.]]))
        self.assertFalse(gain.requires_grad)


if __name__ == '__main__':
    unittest.main()
