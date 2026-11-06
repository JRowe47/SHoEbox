"""Competitive ownership, conditional credit and exact training continuation."""
import copy
import io
import unittest
import torch
from .ports import tensor_list
from .task_memory import TaskMemoryConfig, TaskMemoryModel, TaskMemoryState
from .task_credit import WriteEligibility, CompetitiveWriteEligibility
from .task_stream import TaskByteStreams, TaskStreamConfig, make_documents
from .task_memory_train import objective


torch.set_num_threads(1)


def model_config(**changes):
    values = dict(width=32, owners=3, policy_features=4, experts_per_block=32,
                  shared_per_block=16, groups=4, address_dim=8, laces=16,
                  hat_symbols=16, write_mode='competitive',
                  memory_aware_write=True, bootstrap_empty=True)
    values.update(changes)
    return TaskMemoryConfig(**values)


class RetentionIntegrationChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(871)
        self.model = TaskMemoryModel(model_config())
        self.inputs = torch.tensor([list(b'A x=1;B x=3;'), list(b'A x=2;B x=0;')])

    def same(self, left, right):
        a, b = tensor_list(left), tensor_list(right)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def eligibility(self, model, batch=2, decay=.995):
        kind = CompetitiveWriteEligibility if model.cfg.write_mode == 'competitive' else WriteEligibility
        return kind.zeros(batch, model.owner_names, model.memory_dim,
                          model.cfg.policy_features, dtype=model.write_policy.dtype, decay=decay)

    def test_bootstrap_recruits_every_owner_without_policy_credit(self):
        for mode in ('competitive', 'independent'):
            model = TaskMemoryModel(model_config(write_mode=mode))
            result = model(self.inputs[:, :model.cfg.owners])
            self.assertTrue(bool(result['bootstrap_mask'].all()))
            self.assertTrue(bool(result['write_actions'].sum(-1).eq(1).all()))
            self.assertFalse(bool(result['write_eligible'].any()))
            self.assertTrue(all(bool(p.valid.all()) for p in result['state'].publications.values()))
            self.assertTrue(all(bool(p.version.eq(1).all()) for p in result['state'].publications.values()))
            credit = self.eligibility(model).advance_chunk(result)
            self.assertEqual(int(torch.count_nonzero(credit.sensitivity)), 0)
            self.same(credit.versions, torch.ones_like(credit.versions))
            # Explicit holds retain empty state and suppress recruitment.
            empty = model.initial_state(2)
            held = model(self.inputs[:, :3], state=empty, write_override='hold')
            self.same(held['state'].publications, empty.publications)
            self.assertFalse(bool(held['bootstrap_mask'].any()))

    def test_null_action_holds_all_valid_publications_and_reset_reboots_only_one_row(self):
        state = self.model(self.inputs[:, :3])['state'].detach()
        with torch.no_grad():
            self.model.write_policy[:, :-1].zero_()
            self.model.write_policy[:, -1].fill_(-2)
        held = self.model(self.inputs[:, 3:6], state=state)
        self.assertFalse(bool(held['write_actions'].any()))
        self.same(held['state'].publications, state.publications)
        reboot = self.model(self.inputs[:, :3], state=held['state'].detach(),
                            reset=torch.tensor([True, False]))
        self.assertTrue(bool(reboot['bootstrap_mask'][0].all()))
        self.assertFalse(bool(reboot['bootstrap_mask'][1].any()))
        self.assertTrue(bool(reboot['write_actions'][0].sum(-1).eq(1).all()))
        self.assertFalse(bool(reboot['write_actions'][1].any()))
        self.assertEqual(reboot['state'].epochs.tolist(), [1, 0])
        self.assertTrue(all(bool(p.version[0].eq(1)) for p in reboot['state'].publications.values()))

    def test_real_model_publication_gradient_matches_split_credit_with_conditioning(self):
        model = self.model.double()
        with torch.no_grad():
            # Exercise nonzero memory conditioning; its old-state input is an
            # explicit stop-gradient, matching the conditional trace contract.
            model.memory_policy_features.weight.normal_(std=.07)
        full = copy.deepcopy(model)
        x = self.inputs[:, :9]
        coefficient = torch.linspace(-.6, .8, 2*model.cfg.owners*model.memory_dim,
                                     dtype=torch.float64).reshape(2, model.cfg.owners, model.memory_dim)
        whole = full(x)
        (whole['state'].memory_tensor()*coefficient).sum().backward()
        expected = full.write_policy.grad.clone()
        first = model(x[:, :6])
        credit = self.eligibility(model, decay=1.).advance_chunk(first)
        boundary = credit.capture(first['state'].memory_tensor())
        split_state = first['state'].detach().with_memory_tensor(boundary.memory)
        final = model(x[:, 6:], state=split_state)
        self.same(whole['state'].memory_tensor(), final['state'].memory_tensor())
        (final['state'].memory_tensor()*coefficient).sum().backward()
        injected = boundary.apply(model.write_policy)
        self.assertGreater(injected, 0.)
        self.assertGreater(float(expected.abs().sum()), 0.)
        torch.testing.assert_close(model.write_policy.grad, expected, rtol=1e-9, atol=1e-10)
        # All cross-owner columns are necessary; competition is not diagonal.
        off_diagonal = ~torch.eye(model.cfg.owners, dtype=torch.bool)[None, :, None, :, None]
        self.assertGreater(float((credit.sensitivity*off_diagonal).abs().sum()), 0.)

    def test_competitive_checkpoint_continues_optimizer_stream_and_credit_exactly(self):
        docs = make_documents(TaskStreamConfig(), seed=85, count=6)
        streams = TaskByteStreams(docs, 2)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        state, trace = self.model.initial_state(2), self.eligibility(self.model)

        def step(model, optim, current, eligibility, stream):
            chunk = stream.next_chunk(9)
            optim.zero_grad(set_to_none=True)
            boundary = eligibility.capture(current.memory_tensor())
            result = model(chunk.values, state=current.with_memory_tensor(boundary.memory), reset=chunk.reset)
            self.assertTrue(bool(result['write_actions'].sum(-1).le(1).all()))
            loss, _, _ = objective(result, chunk)
            loss.backward()
            boundary.apply(model.write_policy)
            eligibility = eligibility.advance_chunk(result)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step()
            return result['state'].detach(), eligibility, loss.detach()

        state, trace, _ = step(self.model, optimizer, state, trace, streams)
        buffer = io.BytesIO()
        torch.save(dict(config=self.model.cfg.to_dict(), weights=self.model.state_dict(),
                        optimizer=optimizer.state_dict(), state=state.to_dict(), credit=trace.to_dict(),
                        stream=streams.state_dict(), rng=torch.get_rng_state()), buffer)
        expected_state, expected_trace, expected_loss = step(self.model, optimizer, state, trace, streams)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = TaskMemoryModel(TaskMemoryConfig.from_dict(saved['config']))
        restored.load_state_dict(saved['weights'])
        restored_optim = torch.optim.AdamW(restored.parameters(), lr=.001)
        restored_optim.load_state_dict(saved['optimizer'])
        restored_streams = TaskByteStreams(docs, 2)
        restored_streams.load_state_dict(saved['stream'])
        torch.set_rng_state(saved['rng'])
        actual_state, actual_trace, actual_loss = step(restored, restored_optim,
            TaskMemoryState.from_dict(saved['state']), CompetitiveWriteEligibility.from_dict(saved['credit']),
            restored_streams)
        self.same(actual_loss, expected_loss)
        self.same(actual_state, expected_state)
        self.same(actual_trace.to_dict(), expected_trace.to_dict())
        self.same(restored.state_dict(), self.model.state_dict())
        self.same(restored_optim.state_dict(), optimizer.state_dict())
        self.assertEqual(restored_streams.state_dict(), streams.state_dict())


if __name__ == '__main__':
    unittest.main()
