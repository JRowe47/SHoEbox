"""Reference equivalence, gradients, masking and cache lifetime for execution."""
import copy
from dataclasses import replace
import unittest
import torch
from ..hat_minkron.events import ByteBatch
from ..hat_minkron.hat import HAT
from ..hat_minkron.kron import MinKron
from ..hat_minkron.semantic_core import ExpertBank
from .config import GraphConfig
from .graph import EdgeSubstitution
from .model import RegionLanguageModel
from .ports import tensor_list
from . import runtime

torch.set_num_threads(1)


class RuntimeChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(381)
        self.cfg = GraphConfig.tiny(rounds=2, recurrent_feedback=True)
        self.model = RegionLanguageModel(self.cfg)
        self.x = torch.tensor([[65, 66, 67], [68, 69, 70]])

    def same(self, a, b):
        left, right = tensor_list(a), tensor_list(b)
        self.assertEqual(len(left), len(right))
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def same_gradient(self, actual, expected, name='gradient'):
        self.assertTrue(bool(torch.isfinite(expected).all()), name)
        self.assertTrue(bool(torch.isfinite(actual).all()), name)
        if expected.dtype == torch.float64:
            rtol, atol, relative_limit = 1e-10, 1e-10, 1e-10
        else:
            # Projection reuse sums incoming derivatives before the shared
            # matrix multiply. FP32 cancellation can therefore move a small
            # coordinate by a few ulps of the tensor's larger coordinates.
            # Keep the original coordinate check, with a scale-aware rounding
            # floor, AND the unchanged per-parameter relative L2 bound.
            scale = float(expected.abs().max())
            rtol, atol = 1e-4, 2e-6 + 2*torch.finfo(expected.dtype).eps*scale
            # Scripted graphs can reorder accumulation in small, cancelling
            # decay derivatives. Keep the coordinate bound and use its relative
            # tolerance for each parameter; FP64 whole-graph checks stay 1e-10.
            relative_limit = rtol
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=name)
        relative = (actual.double()-expected.double()).norm()/expected.double().norm().clamp_min(1e-12)
        self.assertLessEqual(float(relative), relative_limit, name)

    def compare(self, inputs, **kwargs):
        self.model.set_execution_backend('reference')
        a = self.model(inputs, self.x, **kwargs)
        a['loss'].backward()
        grads = {n: None if p.grad is None else p.grad.clone() for n, p in self.model.named_parameters()}
        self.model.zero_grad(set_to_none=True)
        self.model.set_execution_backend('optimized')
        b = self.model(inputs, self.x, **kwargs)
        for key in ('logits', 'aux', 'hat_aux', 'state', 'trace', 'view_activity', 'edge_activity',
                    'round_activity', 'node_activity', 'publication_activity'):
            self.same(a[key], b[key])
        for key in ('expert_executions', 'external_events', 'source_occurrences', 'internal_rounds'):
            self.assertEqual(a[key], b[key])
        b['loss'].backward()
        for name, p in self.model.named_parameters():
            if grads[name] is None:
                self.assertIsNone(p.grad, name)
            else:
                self.assertIsNotNone(p.grad, name)
                self.same_gradient(p.grad, grads[name], name)
        self.assertIsNone(runtime._cache.get())

    def test_text_forward_state_routes_and_gradients(self):
        self.compare(self.x, trace=True, return_view_activity=True, return_edge_activity=True,
                     return_round_activity=True, return_node_activity=('A', 'P'))

    def test_text_forward_float64_gradients(self):
        self.model.double()
        self.test_text_forward_state_routes_and_gradients()

    def test_expert_kernel_duplicate_routes_and_unused_banks(self):
        cfg = self.cfg.hat_config().child_config()
        old = runtime.SplitExpertBank(cfg, ExpertBank(cfg, 3), 2).double()
        new = runtime.RuntimeSplitExpertBank(cfg, ExpertBank(cfg, 3), 2).double()
        new.load_state_dict(old.state_dict())
        for picks in ((0, 0, 1), (3, 4, 4), (0, 4, 0)):
            ids = torch.tensor(picks)[None, None].expand(2, 2, -1)
            x = torch.randn(2, 2, cfg.width, dtype=torch.float64, requires_grad=True)
            gates = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
            y, weights = x.detach().clone().requires_grad_(), gates.detach().clone().requires_grad_()
            old.zero_grad(set_to_none=True); new.zero_grad(set_to_none=True)
            expected = old(x, ids, gates)
            token = runtime._cache.set({})
            try:
                with torch.jit.optimized_execution(False):
                    actual = new(y, ids, weights)
            finally:
                runtime._cache.reset(token)
            self.same(expected, actual)
            expected.square().sum().backward(); actual.square().sum().backward()
            for a, b in [(x, y), (gates, weights), *zip(old.parameters(), new.parameters())]:
                if a.grad is None:
                    self.assertIsNone(b.grad)
                else:
                    self.same_gradient(b.grad, a.grad)

    def test_mixed_rows_resets_substitutions_and_inactive_nodes(self):
        active = torch.tensor([[True, False, True], [False, True, True]])
        reset = torch.tensor([[False, True, False], [False, False, True]])
        edge = 'A__U__forward'
        sub = EdgeSubstitution(torch.randn(2, 3, self.cfg.width), torch.randn(2, 3, 3, self.cfg.groups))
        self.compare(self.x, node_masks={'B': active, 'C': torch.zeros_like(active)},
                     availability={'B': active}, edge_masks={edge: active}, substitutions={edge: sub},
                     reset=reset, trace=True, edge_bias={edge: .1})

    def test_multiple_live_links_and_partial_arrivals(self):
        links = torch.tensor([[1, 2, 1], [2, 1, 2]])
        events = ByteBatch(self.x, links, torch.full_like(self.x, 7))
        self.compare(events, availability={'A': torch.tensor([[True, True, False], [True, False, True]])})

    def test_multiple_live_links_float64_gradients(self):
        # Check the entire graph at high precision, not only the recurrence.
        # This distinguishes FP32 cancellation from a wrong shared derivative.
        self.model.double()
        self.test_multiple_live_links_and_partial_arrivals()

    def test_gradient_comparison_rejects_wrong_derivatives(self):
        # The rounding floor must not hide missing small gradients, a spurious
        # gradient at zero, or a locally wrong coordinate in a larger tensor.
        for actual, expected in (([0.], [1e-8]), ([1e-8], [0.]),
                                 ([1., 1e-5], [1., 0.]), ([1.01], [1.]),
                                 ([float('nan')], [1.]), ([float('inf')], [1.])):
            with self.subTest(actual=actual, expected=expected):
                with self.assertRaises(AssertionError):
                    self.same_gradient(torch.tensor(actual), torch.tensor(expected))

    def test_constructor_keeps_original_weights_and_rng(self):
        cfg = self.cfg.hat_config()
        torch.manual_seed(33); old = HAT(cfg, safe_zero=True); old_rng = torch.get_rng_state()
        torch.manual_seed(33); new = runtime.RuntimeHAT(cfg, safe_zero=True)
        self.assertEqual(old.state_dict().keys(), new.state_dict().keys())
        self.same(old.state_dict(), new.state_dict())
        self.same(old_rng, torch.get_rng_state())

    def test_cache_does_not_connect_separate_backwards_or_survive_load(self):
        first = self.model(self.x, self.x)
        second = self.model(self.x, self.x)
        first['loss'].backward()
        self.model.zero_grad(set_to_none=True)
        second['loss'].backward()  # A stale cached graph would already be freed.
        changed = copy.deepcopy(self.model.state_dict())
        changed['nodes.A.core.lace.template_logits'].add_(torch.randn_like(changed['nodes.A.core.lace.template_logits']))
        self.model.load_state_dict(changed)
        with torch.no_grad():
            fast = self.model(self.x)
            self.model.set_execution_backend('reference')
            reference = self.model(self.x)
            self.same(fast['logits'], reference['logits'])
        with self.assertRaises(ValueError):
            self.model.set_execution_backend('unknown')
        self.model.set_execution_backend('optimized')
        with self.assertRaises(ValueError):
            self.model(self.x, rounds=0)
        self.assertIsNone(runtime._cache.get())

    def test_single_step_memory_float64_values_and_derivatives(self):
        old = MinKron(8, 2, 4, 3, safe_zero=True).double()
        new = runtime.RuntimeMinKron(8, 2, 4, 3, safe_zero=True).double()
        new.load_state_dict(old.state_dict())
        for reset in (False, True):
            for zero in (False, True):
                x = (torch.zeros(2, 1, 8) if zero else torch.randn(2, 1, 8)).double().requires_grad_()
                y = x.detach().clone().requires_grad_()
                state = torch.randn(2, 2, 4, 3, dtype=torch.float64, requires_grad=True)
                other = state.detach().clone().requires_grad_()
                old.zero_grad(set_to_none=True); new.zero_grad(set_to_none=True)
                a = old(x, state, 'step', reset)
                token = runtime._cache.set({})
                try:
                    b = new(y, other, 'step', reset)
                finally:
                    runtime._cache.reset(token)
                self.same(a, b)
                sum(t.square().sum() for t in a).backward()
                sum(t.square().sum() for t in b).backward()
                for p, q in [(x, y), (state, other), *zip(old.parameters(), new.parameters())]:
                    if p.grad is None:
                        self.assertIsNone(q.grad)
                    else:
                        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
