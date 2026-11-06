"""Reference equivalence, gradients, masking and cache lifetime for execution."""
import copy
from dataclasses import replace
import unittest
import torch
from ..hat_minkron.events import ByteBatch
from ..hat_minkron.hat import HAT
from ..hat_minkron.kron import MinKron
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
                # Sharing intermediate graphs changes FP32 gradient summation
                # order. Bound both each coordinate and total relative error.
                torch.testing.assert_close(p.grad, grads[name], rtol=1e-4, atol=2e-6, msg=name)
                relative = (p.grad-grads[name]).double().norm()/grads[name].double().norm().clamp_min(1e-12)
                self.assertLessEqual(float(relative), 5e-6, name)
        self.assertIsNone(runtime._cache.get())

    def test_text_forward_state_routes_and_gradients(self):
        self.compare(self.x, trace=True, return_view_activity=True, return_edge_activity=True,
                     return_round_activity=True, return_node_activity=('A', 'P'))

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
