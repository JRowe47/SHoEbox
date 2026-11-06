"""Independent derivatives, optimizer semantics, and generation invalidation."""
import copy
import unittest
import torch
from .config import GraphConfig
from .model import RegionLanguageModel
from . import runtime
from .runtime_autograd import DenseBankGradient, SelectedRows, SelectedExpert, KronStep
from ..hat_minkron.kron import compact_chunk
from .gradient_balance import gradient_moments, aggregate_statistics, statistics

torch.set_num_threads(1)


class FusionChecks(unittest.TestCase):
    def test_selected_expert_derivatives_and_large_split_banks(self):
        from ..hat_minkron.semantic_core import ExpertBank
        torch.manual_seed(921)
        inputs = tuple(torch.randn(shape, dtype=torch.float64, requires_grad=True)
                       for shape in ((2,3), (2,2,3), (2,2,3), (2,)))
        self.assertTrue(torch.autograd.gradcheck(SelectedExpert.apply, inputs, atol=1e-7, rtol=1e-5))
        self.assertTrue(torch.autograd.gradgradcheck(SelectedExpert.apply, inputs, atol=1e-7, rtol=1e-5))
        cfg = GraphConfig.tiny().hat_config().child_config()
        old = runtime.SplitExpertBank(cfg, ExpertBank(cfg, 512), 512).double()
        new = runtime.RuntimeSplitExpertBank(cfg, ExpertBank(cfg, 512), 512).double()
        new.load_state_dict(old.state_dict())
        for picks in ((0,0,1), (512,513,513), (0,513,0)):
            old.zero_grad(set_to_none=True);new.zero_grad(set_to_none=True)
            ids = torch.tensor(picks)[None,None].expand(2,2,-1)
            x = torch.randn(2,2,cfg.width,dtype=torch.float64,requires_grad=True)
            gates = torch.randn(2,2,3,dtype=torch.float64,requires_grad=True)
            y=x.detach().clone().requires_grad_();weights=gates.detach().clone().requires_grad_()
            expected=old(x,ids,gates)
            token=runtime._cache.set({})
            try:
                actual=new(y,ids,weights)
            finally:
                runtime._cache.reset(token)
            torch.testing.assert_close(expected,actual,rtol=0,atol=0)
            expected.square().sum().backward();actual.square().sum().backward()
            for a,b in ((x,y),(gates,weights),*zip(old.parameters(),new.parameters())):
                if a.grad is None:
                    self.assertIsNone(b.grad)
                else:
                    torch.testing.assert_close(a.grad,b.grad,rtol=1e-10,atol=1e-10)

    def test_compact_gradients_accumulate_once_and_keep_adamw_semantics(self):
        torch.manual_seed(81)
        left = torch.nn.Parameter(torch.randn(9, 3, 4, dtype=torch.float64))
        right = torch.nn.Parameter(left.detach().clone())
        optimizers = [torch.optim.AdamW([p], lr=.01, weight_decay=.1) for p in (left, right)]
        ids = [torch.tensor([1, 1, 3]), torch.tensor([3, 1, 4]), torch.tensor([8, 8, 4])]
        for step in range(3):
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            shared = DenseBankGradient.apply(right)
            expected = sum(left.index_select(0, picks).sin().sum() for picks in ids[:3-step])
            actual = sum(SelectedRows.apply(shared, picks).sin().sum() for picks in ids[:3-step])
            # Functional gradients and ordinary backward use the same path.
            gradient, = torch.autograd.grad(actual, right, retain_graph=True)
            expected.backward(); actual.backward()
            self.assertFalse(right.grad.is_sparse)
            torch.testing.assert_close(right.grad, left.grad, rtol=1e-14, atol=1e-14)
            torch.testing.assert_close(gradient, right.grad, rtol=0, atol=0)
            for opt in optimizers:
                opt.step()
            torch.testing.assert_close(left, right, rtol=1e-14, atol=1e-14)
            for key, value in optimizers[0].state[left].items():
                torch.testing.assert_close(value, optimizers[1].state[right][key], rtol=1e-14, atol=1e-14)

    def test_analytical_recurrence_finite_differences_and_clamped_norm(self):
        torch.manual_seed(31)
        for k, v in ((2, 2), (3, 2)):
            values = [torch.randn(1, 1, size, dtype=torch.float64, requires_grad=True) for size in (k, k, v, k)]
            values += [torch.rand(1, 1, dtype=torch.float64, requires_grad=True) for _ in range(2)]
            values += [torch.randn(1, 1, k, v, dtype=torch.float64, requires_grad=True)]
            self.assertTrue(torch.autograd.gradcheck(KronStep.apply, tuple(values), eps=1e-6, atol=1e-7, rtol=1e-5))
            self.assertTrue(torch.autograd.gradgradcheck(KronStep.apply, tuple(values), eps=1e-6, atol=1e-7, rtol=1e-5))
            for scale in (0., 1e-11, 1.):
                inputs = [x.detach().clone().requires_grad_() for x in values]
                inputs[2] = (inputs[2].detach()*scale).requires_grad_()
                other = [x.detach().clone().requires_grad_() for x in inputs]
                expected = compact_chunk(*(x[:, None] for x in inputs[:6]), state=inputs[6])
                actual = KronStep.apply(*other)
                torch.testing.assert_close(expected[0][:, 0], actual[0], rtol=1e-14, atol=1e-14)
                torch.testing.assert_close(expected[1], actual[1], rtol=1e-14, atol=1e-14)
                ga = torch.autograd.grad(sum(x.square().sum() for x in expected), inputs)
                gb = torch.autograd.grad(sum(x.square().sum() for x in actual), other)
                for a, b in zip(ga, gb):
                    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_generation_cache_reuses_and_invalidates_weights(self):
        torch.manual_seed(33)
        model = RegionLanguageModel(GraphConfig.tiny())
        x = torch.tensor([[65]])
        @torch.no_grad()
        @runtime.generation_scope
        def session(model):
            model(x)
            original = runtime._generation.get()[2]
            model(x)
            self.assertIs(original, runtime._generation.get()[2])
            # In-place updates and checkpoint loading each invalidate the cache.
            model.nodes.A.core.blocks[0].router.message.weight.add_(.01)
            changed = model(x)['logits']
            self.assertIsNot(original, runtime._generation.get()[2])
            model.set_execution_backend('reference')
            torch.testing.assert_close(changed, model(x)['logits'], rtol=2e-5, atol=1e-5)
            model.set_execution_backend('optimized')
            old_cache = runtime._generation.get()[2]
            model.load_state_dict(copy.deepcopy(model.state_dict()))
            model(x)
            self.assertIsNot(old_cache, runtime._generation.get()[2])
            old_cache = runtime._generation.get()[2]
            model.double()
            self.assertEqual(model(x)['logits'].dtype, torch.float64)
            self.assertIsNot(old_cache, runtime._generation.get()[2])
        session(model)
        self.assertIsNone(runtime._generation.get())
        self.assertIsNone(runtime._cache.get())

    def test_generation_exception_cleans_cache_and_restores_mode(self):
        model = RegionLanguageModel(GraphConfig.tiny())
        with self.assertRaises(ValueError):
            model.generate('R', count=2, rounds=0)
        self.assertTrue(model.training)
        self.assertIsNone(runtime._generation.get())
        self.assertIsNone(runtime._cache.get())

    def test_batched_stages_preserve_float64_state_routes_and_gradients(self):
        torch.manual_seed(53)
        model = RegionLanguageModel(GraphConfig.tiny(rounds=2, recurrent_feedback=True)).double()
        x = torch.tensor([[65, 66], [67, 68]])
        model.batch_independent = False
        a = model(x, x, trace=True)
        a['loss'].backward()
        gradients = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        model.batch_independent = True
        b = model(x, x, trace=True, reverse_independent=True)
        b['loss'].backward()
        for node in b['state'].nodes.values():
            for tensor in node.contextualizer:
                self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel()*tensor.element_size())
        torch.testing.assert_close(a['logits'], b['logits'], rtol=1e-11, atol=1e-11)
        for left, right in zip(a['state'].tensors(), b['state'].tensors()):
            torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-11)
        for name, p in model.named_parameters():
            if gradients[name] is None:
                self.assertIsNone(p.grad)
            else:
                torch.testing.assert_close(p.grad, gradients[name], rtol=1e-9, atol=1e-10, msg=name)
        left = {(r['node'], r['event'], r['round']):r['routing'] for r in a['trace']}
        for record in b['trace']:
            for key in ('groups', 'laces', 'observed_laces', 'experts'):
                torch.testing.assert_close(left[record['node'], record['event'], record['round']][key],
                                           record['routing'][key], rtol=0, atol=0)

    def test_gradient_moments_are_reused_across_partitions(self):
        a = [torch.tensor([1e30, 2.]), None, torch.tensor([.3, .4])]
        b = [-a[0], torch.tensor([2., 3.]), None]
        moments = gradient_moments(a, b)
        for indices in ([0, 1, 2], [0], [1, 2], [2, 0]):
            self.assertEqual(aggregate_statistics(moments, indices), statistics([a[i] for i in indices], [b[i] for i in indices]))


if __name__ == '__main__':
    unittest.main()
