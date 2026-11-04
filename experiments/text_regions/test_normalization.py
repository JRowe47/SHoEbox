"""Absent directions must not multiply arbitrary normalization derivatives."""
import copy
import unittest
import torch
from torch.nn import functional as F
from ..hat_minkron.hat import SymbolLevel
from ..hat_minkron.kron import normalize_direction
from .config import GraphConfig
from .episode_data import EpisodeStream
from .probe_train import build_probe_model

torch.set_num_threads(2)


class ZeroDirectionChecks(unittest.TestCase):
    def test_missing_and_negligible_directions_have_zero_gradient(self):
        x = torch.tensor([[0., 0.], [1e-14, -1e-14]], requires_grad=True)
        value = normalize_direction(x, safe_zero=True)
        self.assertEqual(float(value.detach().abs().sum()), 0)
        value.sum().backward()
        self.assertEqual(float(x.grad.abs().sum()), 0)
        legacy = torch.zeros(1, 2, requires_grad=True)
        normalize_direction(legacy).sum().backward()
        torch.testing.assert_close(legacy.grad, torch.full_like(legacy, 1e12), rtol=0, atol=0)

    def test_nonzero_normalization_preserves_values_and_gradients_exactly(self):
        torch.manual_seed(829)
        x = torch.randn(3, 4, requires_grad=True)
        clone = x.detach().clone().requires_grad_()
        ordinary, safe = F.normalize(x, dim=-1), normalize_direction(clone, safe_zero=True)
        torch.testing.assert_close(ordinary, safe, rtol=0, atol=0)
        weights = torch.randn_like(x)
        (ordinary*weights).sum().backward();(safe*weights).sum().backward()
        torch.testing.assert_close(x.grad, clone.grad, rtol=0, atol=0)

    def test_symbol_and_memory_nonzero_execution_preserves_exact_gradients(self):
        torch.manual_seed(914)
        cfg = GraphConfig.tiny().hat_config()
        ordinary = SymbolLevel(cfg)
        safe = copy.deepcopy(ordinary)
        safe.safe_zero = safe.memory.safe_zero = True
        x = torch.randn(2, 3, cfg.width, requires_grad=True)
        clone = x.detach().clone().requires_grad_()
        links = torch.zeros(2, 3, dtype=torch.long)
        state = {0: torch.randn_like(ordinary.memory.initial_state(2, x.device))*.01}
        first, s1, _ = ordinary(x, links, state, backend='step')
        second, s2, _ = safe(clone, links, state, backend='step')
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(s1[0], s2[0], rtol=0, atol=0)
        weights = torch.randn_like(first)
        (first*weights).sum().backward();(second*weights).sum().backward()
        torch.testing.assert_close(x.grad, clone.grad, rtol=0, atol=0)
        for a, b in zip(ordinary.parameters(), safe.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_absent_view_segment_has_finite_ordinary_scale_gradients(self):
        torch.manual_seed(20261104)
        model = build_probe_model(GraphConfig.tiny(seed=20261104), 'shared_private', True, True)
        # This segment contains four unavailable B occurrences after recurrent
        # state has been populated: both nested zero-query Jacobians matter.
        self.assertTrue(model.cfg.zero_safe_norm)
        self.assertTrue(all(module.safe_zero for module in model.modules() if hasattr(module, 'safe_zero')))
        batch = EpisodeStream('shared_private', 2, gap=0, seed=20261104).next_chunk(16)
        result = model(batch.inputs, batch.targets, reset=batch.reset, availability=batch.availability)
        result['loss'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        self.assertTrue(bool(torch.isfinite(norm)))
        self.assertLess(float(norm), 1000.)
        self.assertGreater(float(norm), 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
