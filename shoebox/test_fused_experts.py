"""Indexed expert derivatives, dense optimizer semantics and graph lifetime."""
import copy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from .fused_experts import expert_forward
from .fused_kron import native_status, load_native


torch.set_num_threads(1)


def reference(x, up, down, ids, gates):
    hidden = F.gelu((up[ids]*x[..., None, None, :]).sum(-1))
    return ((hidden[..., None]*down[ids]).sum(-2)*gates[..., None]).sum(-2)/ids.shape[-1]**.5


def fused(x, up, down, ids, gates):
    return expert_forward(SimpleNamespace(up=up, down=down), x, ids, gates,
                          lambda module, name, compute: compute())


class FusedExpertChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        status = native_status()
        if not status['available']:
            if os.environ.get('SHOEBOX_REQUIRE_FUSED') == '1':
                raise RuntimeError(str(status))
            raise unittest.SkipTest(str(status))

    def inputs(self, dtype=torch.float64, k=3):
        x = torch.randn(2, 2, 7, dtype=dtype, requires_grad=True)
        up = (torch.randn(5, 3, 7, dtype=dtype)*.3).requires_grad_()
        down = (torch.randn_like(up)*.2).requires_grad_()
        gates = torch.randn(2, 2, k, dtype=dtype, requires_grad=True)
        ids = torch.tensor([0, 0, 3, 3, 0, 0, 1, 3, 1, 1, 1, 3]).reshape(2, 2, 3)[..., :k]
        return (x, up, down, gates), ids

    def test_values_and_all_gradients_with_duplicate_rows(self):
        torch.manual_seed(901)
        for dtype in (torch.float32, torch.float64):
            for k in (1, 3):
                args, ids = self.inputs(dtype, k)
                other = tuple(t.detach().clone().requires_grad_() for t in args)
                x, u, d, g = args
                a = reference(x, u, d, ids, g)
                b = fused(*other[:3], ids, other[3])
                tolerance = dict(atol=2e-6, rtol=2e-5) if dtype == torch.float32 else dict(atol=2e-12, rtol=2e-11)
                torch.testing.assert_close(a, b, **tolerance)
                grad = torch.randn_like(a)
                ga = torch.autograd.grad(a, args, grad)
                gb = torch.autograd.grad(b, other, grad)
                for left, right in zip(ga, gb):
                    self.assertFalse(right.is_sparse)
                    torch.testing.assert_close(left, right, **tolerance)
                self.assertTrue(all(t.grad is None for t in other))
                self.assertEqual(float(gb[1][2].abs().sum()), 0.)
                with torch.no_grad():
                    torch.testing.assert_close(fused(x, u, d, ids, g), b, atol=0, rtol=0)

    def test_finite_difference_gradcheck_and_noncontiguous_inputs(self):
        torch.manual_seed(902)
        args, ids = self.inputs()
        self.assertTrue(torch.autograd.gradcheck(lambda x, u, d, g: fused(x, u, d, ids, g),
                                                 args, fast_mode=True, atol=1e-7, rtol=1e-5))
        x, u, d, g = args
        x = x.transpose(0, 1)
        ids, g = ids.transpose(0, 1), g.transpose(0, 1)
        torch.testing.assert_close(fused(x, u, d, ids, g), reference(x, u, d, ids, g), atol=2e-12, rtol=2e-11)

    def test_scoped_accumulation_retained_graph_hooks_and_adamw(self):
        torch.manual_seed(903)
        args, ids = self.inputs()
        original = [torch.nn.Parameter(t.detach().clone()) for t in args[1:3]]
        native = [torch.nn.Parameter(t.detach().clone()) for t in args[1:3]]
        optimizers = [torch.optim.AdamW(ps, lr=.003, weight_decay=.1, foreach=False) for ps in (original, native)]
        seen = []
        native[0].register_hook(lambda grad: seen.append(grad.is_sparse))
        for step in range(3):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=step != 1)
            bank = SimpleNamespace(up=native[0], down=native[1])
            cache = {}
            def cached(module, name, compute):
                if name not in cache:
                    cache[name] = compute()
                return cache[name]
            x, g = args[0].detach(), args[3].detach()
            # Row 0 is active initially, then absent: its existing AdamW moments
            # must still decay/update exactly like the dense reference.
            active = ids if step == 0 else torch.ones_like(ids)
            a = sum(reference(x+i*.02, *original, active, g).square().sum() for i in range(11))
            b = sum(expert_forward(bank, x+i*.02, active, g, cached).square().sum() for i in range(11))
            self.assertEqual(len(cache), 2)
            expected = torch.autograd.grad(a, original, retain_graph=True)
            actual = torch.autograd.grad(b, native, retain_graph=True)
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-10)
            a.backward(retain_graph=True); b.backward(retain_graph=True)
            a.backward(); b.backward()
            for left, right in zip(original, native):
                torch.testing.assert_close(left.grad, right.grad, atol=1e-11, rtol=1e-10)
            for optimizer in optimizers:
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 1., foreach=True)
                optimizer.step()
            for left, right in zip(original, native):
                torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-10)
            for left, right in zip(optimizers[0].state.values(), optimizers[1].state.values()):
                for key in left:
                    torch.testing.assert_close(left[key], right[key], atol=1e-11, rtol=1e-10)
        self.assertTrue(seen)
        self.assertFalse(any(seen))

    def test_cortical_live_state_across_forward_calls(self):
        from . import CorticalModel
        from .test_wide import small_wide
        torch.manual_seed(905)
        original = CorticalModel(small_wide()).double().set_execution_backend('fused')
        native = copy.deepcopy(original)
        data = torch.tensor([[65, 66, 67, 68, 69, 70], [71, 72, 73, 74, 75, 76]])
        with patch.dict(os.environ, {'SHOEBOX_EXPERTS':'0'}):
            a = original(data[:, :3])
            aa = original(data[:, 3:], targets=data[:, :3], state=a['state'])
        b = native(data[:, :3])
        bb = native(data[:, 3:], targets=data[:, :3], state=b['state'])
        torch.testing.assert_close(aa['logits'], bb['logits'], atol=2e-10, rtol=2e-9)
        # An intermediate activation-only gradient pass must not write any .grad
        # or consume the bank gradient edges needed by the subsequent backward.
        first = (a['state'].nodes['A1'].blocks[0], b['state'].nodes['A1'].blocks[0])
        ga = torch.autograd.grad(aa['loss'], first[0], retain_graph=True)[0]
        gb = torch.autograd.grad(bb['loss'], first[1], retain_graph=True)[0]
        torch.testing.assert_close(ga, gb, atol=2e-9, rtol=2e-8)
        self.assertGreater(float(gb.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in native.parameters()))
        aa['loss'].backward(); bb['loss'].backward()
        for (name, left), (_, right) in zip(original.named_parameters(), native.named_parameters()):
            self.assertEqual(left.grad is None, right.grad is None, name)
            if left.grad is not None:
                self.assertFalse(right.grad.is_sparse)
                torch.testing.assert_close(left.grad, right.grad, atol=2e-8, rtol=2e-7, msg=name)

    def test_parallel_frozen_weights_fallback_and_invalid_ids(self):
        torch.manual_seed(904)
        args, ids = self.inputs()
        x, u, d, g = args
        x = x.repeat(12, 1, 1).detach().requires_grad_()
        ids, g = ids.repeat(12, 1, 1), g.repeat(12, 1, 1).detach().requires_grad_()
        u = u.detach()
        a = fused(x, u, d, ids, g)
        ga = torch.autograd.grad(a.square().sum(), (x, d, g))
        try:
            torch.set_num_threads(2)
            b = fused(x, u, d, ids, g)
            gb = torch.autograd.grad(b.square().sum(), (x, d, g))
        finally:
            torch.set_num_threads(1)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        for left, right in zip(ga, gb):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        for setting in ('SHOEBOX_EXPERTS', 'SHOEBOX_FUSED'):
            with patch.dict(os.environ, {setting:'0'}):
                self.assertIsNone(fused(x, u, d, ids, g))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            self.assertIsNone(fused(x, u, d, ids, g))
        with self.assertRaisesRegex(RuntimeError, 'out of range'):
            fused(x, u, d, torch.full_like(ids, -1), g)
        with self.assertRaisesRegex(RuntimeError, 'dimensions'):
            load_native().experts_forward(x.flatten(0, 1), u, d[:, :, :2].contiguous(),
                                          ids.flatten(0, 1), g.flatten(0, 1), True)


if __name__ == '__main__':
    unittest.main()
