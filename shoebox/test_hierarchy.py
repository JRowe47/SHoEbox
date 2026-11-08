"""Causality, complete temporal recipes, task credit and continuation contracts."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import torch
from torch import nn
from .hierarchy_config import HierarchyConfig
from .hierarchy import HierarchyModel, future_span_loss
from .checkpoint import save, load


def tiny(**kw):
    return HierarchyConfig(message_width=8, local_width=16, mid_width=16, high_width=24,
            decoder_width=16, experts_per_layer=8, groups=2, active_groups=1,
            experts_per_group=2, expert_hidden=4, hat_symbols=8, hat_active_symbols=2,
            address_dim=4, laces=4, active_laces=2, kron_heads=1, kron_key=4, kron_value=4, **kw)


class HierarchyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)
        self.model = HierarchyModel(tiny()).set_execution_backend('optimized')
        self.x = torch.randint(256, (2, 33))

    def test_schedule_complete_recipes_and_bounded_pending(self):
        with torch.no_grad():
            out = self.model(self.x, trace=True)
        self.assertEqual({n: s.clock for n, s in out['state'].nodes.items()},
                         dict(A1=33, B1=33, L2=8, S2=8, L3=2, Y=33))
        self.assertEqual(len(out['state'].pending_local), 1)
        self.assertEqual(len(out['state'].pending_mid), 0)
        for r in out['trace']:
            frames = r['inputs']
            self.assertTrue(all(f['end'] <= r['event'] for f in frames))
            if r['region'] in ('L2', 'S2', 'L3'):
                sources = {'L2': ['A1'], 'S2': ['A1', 'B1'], 'L3': ['L2', 'S2']}[r['region']]
                self.assertEqual([f['source'] for f in frames], sources*4)
                self.assertEqual(len({(f['source'], f['occurrence']) for f in frames}), len(frames))
            if r['context']:
                self.assertLess(r['context']['end'], frames[0]['start'])
        self.assertEqual(out['state'].nodes['S2'].bytes_seen, 8*8)
        self.assertEqual(out['state'].nodes['L3'].bytes_seen, 2*8)

    def test_prefix_target_and_chunk_invariance(self):
        with torch.no_grad():
            whole = self.model(self.x)
            short = self.model(self.x[:, :19])
            altered = self.x.clone(); altered[:, 19:] = (altered[:, 19:]+71) % 256
            other = self.model(altered, targets=torch.zeros_like(altered))
            torch.testing.assert_close(whole['logits'][:, :19], short['logits'])
            torch.testing.assert_close(whole['logits'][:, :19], other['logits'][:, :19])
            parts, state = [], None
            for a, b in ((0, 3), (3, 11), (11, 16), (16, 33)):
                result = self.model(self.x[:, a:b], state=state)
                state = result['state'].detach(); parts.append(result['logits'])
            torch.testing.assert_close(whole['logits'], torch.cat(parts, 1))

    def test_cross_chunk_gradients_and_all_scales_task_credit(self):
        first = self.model(self.x[:, :3])
        second = self.model(self.x[:, 3:], targets=self.x[:, 3:].roll(-1, 1), state=first['state'])
        second['ce'].backward()
        for name in self.model.regions:
            grad = self.model.regions[name].hat.input.weight.grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(float(grad.abs().sum()), 0, name)
        self.assertGreater(float(self.model.regions['L3'].publish.weight.grad.abs().sum()), 0)
        self.assertEqual(second['span_targets'], dict(mid=60, high=36))
        detached = second['state'].detach()
        self.assertFalse(detached.latest['L3'].value.requires_grad)
        self.assertFalse(detached.pending_local[0][0].value.requires_grad)

    def test_future_targets_align_and_mask_tails(self):
        head = nn.Linear(3, 4*256)
        z = [torch.randn(2, 3), torch.randn(2, 3)]
        targets = torch.randint(256, (2, 10)); targets[0, 5] = -100
        loss, count = future_span_loss(head, z, [3, 7], targets, 4)
        expected = nn.functional.cross_entropy(head(torch.stack(z, 1)).reshape(-1, 256),
                    torch.stack((targets[:, 3:7], nn.functional.pad(targets[:, 7:10], (0, 1), value=-100)), 1).flatten())
        torch.testing.assert_close(loss, expected)
        self.assertEqual(count, 13)
        with torch.no_grad():
            target = torch.randint(256, (2, 48))
            a = self.model(self.x[:, :32], target[:, :32], future_targets=target)
            target[:, 32:] = (target[:, 32:]+12) % 256
            b = self.model(self.x[:, :32], target[:, :32], future_targets=target)
            torch.testing.assert_close(a['logits'], b['logits'])
            self.assertEqual(a['span_targets'], dict(mid=64, high=64))

    def test_swap_context_holds_local_states_fixed(self):
        with torch.no_grad():
            a = self.model(self.x)
            b = self.model(self.x, context_permutation=torch.tensor([1, 0]))
            self.assertGreater(float((a['logits'][:, 15:]-b['logits'][:, 15:]).abs().max()), 1e-6)
            for name in ('A1', 'B1', 'L2', 'S2', 'L3'):
                torch.testing.assert_close(a['state'].latest[name].value, b['state'].latest[name].value)

    def test_checkpoint_resume_between_boundaries(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        first = self.model(self.x[:, :7], targets=self.x[:, 1:8])
        first['loss'].backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pilot.pt'
            save(path, self.model, first['state'], optimizer, step=1)
            restored, state, _, record = load(path)
            restored.set_execution_backend('optimized')
            self.assertEqual(record['step'], 1)
            with torch.no_grad():
                a = self.model(self.x[:, 7:], state=first['state'].detach())
                b = restored(self.x[:, 7:], state=state)
            torch.testing.assert_close(a['logits'], b['logits'], rtol=0, atol=0)

    def test_fused_forward_backward_matches_portable(self):
        native = HierarchyModel(tiny()).set_execution_backend('cpu')
        native.load_state_dict(self.model.state_dict())
        a = self.model(self.x[:, :17], targets=self.x[:, 1:18])
        b = native(self.x[:, :17], targets=self.x[:, 1:18])
        torch.testing.assert_close(a['logits'], b['logits'], rtol=1e-4, atol=1e-5)
        a['loss'].backward(); b['loss'].backward()
        for (name, p), (_, q) in zip(self.model.named_parameters(), native.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=4e-3, atol=5e-5, msg=name)

    def test_control_has_one_region_and_no_auxiliary_heads(self):
        control = HierarchyModel(replace(tiny(), local_only=True)).set_execution_backend('optimized')
        result = control(self.x, targets=self.x.roll(-1, 1))
        self.assertEqual(tuple(control.regions), ('A1',))
        self.assertEqual(result['span_targets'], dict(mid=0, high=0))
        self.assertEqual(control.parameter_cost()['region_executions_per_byte'], 1.)
        self.assertAlmostEqual(self.model.parameter_cost()['region_executions_per_byte'], 3.5625)


if __name__ == '__main__':
    unittest.main()
