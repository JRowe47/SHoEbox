"""Scaling regressions: six-layer causality, gradients, coverage and intervention."""
import io
import unittest
import torch
from model import Config, SHoEGPT
from train import diagnostics


class ScaledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model = SHoEGPT(Config.scaled_10m()).eval()
        cls.tokens = torch.randint(65, (2, 27))

    def test_parameter_count_and_membership(self):
        s = self.model.parameter_summary()
        self.assertEqual(s['total'], 10027388)
        self.assertEqual(s['expert_counts'], [683, 683, 683, 683, 682, 682])
        self.assertEqual(s['active_expert_weight_parameters_per_token'], 207360)
        for block, count in zip(self.model.blocks, s['expert_counts']):
            ids = block.router.membership.flatten()
            torch.testing.assert_close(ids[ids >= 0].sort().values, torch.arange(count))

    def test_causal_prefix_and_suffix(self):
        changed = self.tokens.clone()
        changed[:, 19:] = torch.randint(65, changed[:, 19:].shape)
        with torch.no_grad():
            a = self.model(self.tokens)['logits'][:, :19]
            b = self.model(changed)['logits'][:, :19]
            c = self.model(self.tokens[:, :19])['logits']
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(a, c, atol=3e-6, rtol=3e-6)

    def test_task_gradients_all_six_endpoints(self):
        self.model.zero_grad(set_to_none=True)
        self.model(self.tokens, self.tokens.roll(-1, 1))['ce'].backward()
        gradient = self.model.lace.template_logits.grad
        for layer in range(6):
            self.assertGreater(float(gradient[:, layer].abs().sum()), 0)
            for parameter in self.model.blocks[layer].parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_generalized_middle_endpoint_permutation(self):
        c = self.model.cfg
        evidence = torch.randn(1, 4, c.layers, c.groups)
        context = torch.randn(1, 4, c.width)
        perm = torch.randperm(c.groups)
        controller = self.model.lace
        with torch.no_grad():
            expected, _ = controller(evidence, context, permutation=perm)
            original = controller.template_logits.clone()
            controller.template_logits[:, c.layers//2].copy_(original[:, c.layers//2, perm])
            actual, _ = controller(evidence, context)
            controller.template_logits.copy_(original)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)

    def test_six_layer_routing_diagnostics(self):
        metrics = diagnostics(self.model, torch.randint(65, (1000,)), batches=1)
        self.assertEqual(len(metrics['experts_used_per_layer']), 6)
        self.assertEqual(len(metrics['valid_pair_fraction_per_layer']), 6)
        self.assertTrue(all(0 < x < 1 for x in metrics['valid_pair_fraction_per_layer']))
        self.assertEqual(metrics['attention_slots_per_query'], 32)

    def test_state_roundtrip(self):
        stream = io.BytesIO()
        torch.save(self.model.state_dict(), stream)
        stream.seek(0)
        clone = SHoEGPT(Config.scaled_10m()).eval()
        clone.load_state_dict(torch.load(stream, weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.tokens)['logits'],
                                       clone(self.tokens)['logits'], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
