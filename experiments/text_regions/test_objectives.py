"""Independent view masking, auxiliary gradients, and explicit eligibility QA."""
from dataclasses import replace
import unittest
import torch
from .objectives import SourceView, ViewObjectiveConfig, PairedViewObjective, DelayedUtility
from .ports import Dependency


def view(name, values, mask=None):
    if mask is None:
        mask = torch.ones(values.shape[:-1], dtype=torch.bool, device=values.device)
    return SourceView(name, values, mask)


def dependency(key, consumed, substituted=None):
    n = len(consumed)
    z = torch.zeros(n, dtype=torch.long)
    result = Dependency(key, 'A', 'latent', z, z, z, z, torch.tensor(consumed, dtype=torch.bool))
    if substituted is not None:
        result = replace(result, substituted=torch.tensor(substituted, dtype=torch.bool))
    return result


class ViewObjectiveChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(719)
        self.a = torch.randn(4, 3, 8, requires_grad=True)
        self.b = torch.randn(4, 3, 8, requires_grad=True)

    def test_disabled_defaults_are_zero_and_do_not_change_codes_from_labels(self):
        objective = PairedViewObjective(8)
        a, b = view('A', self.a), view('B', self.b)
        first = objective(a, b)
        second = objective(a, b, {'A': torch.arange(4), 'B': torch.arange(4)+20})
        self.assertEqual(float(first['loss'].detach()), 0)
        for key in ('agreement', 'prediction', 'variance', 'covariance', 'private', 'separation'):
            self.assertEqual(float(first[key].detach()), 0)
        for name in ('A', 'B'):
            for key in ('shared', 'private'):
                torch.testing.assert_close(first['codes'][name][key], second['codes'][name][key])

    def test_missing_rows_do_not_affect_paired_loss_or_gradients(self):
        cfg = ViewObjectiveConfig(agreement_enabled=True, prediction_enabled=True, variance_enabled=True,
                                  covariance_enabled=True, separation_enabled=True)
        objective = PairedViewObjective(8, cfg=cfg)
        mask = torch.ones(4, 3, dtype=torch.bool)
        mask[0] = False
        changed = self.b.detach().clone()
        changed[0] = float('nan')
        changed.requires_grad_()
        first = objective(view('A', self.a), view('B', self.b, mask))
        second = objective(view('A', self.a), view('B', changed, mask))
        torch.testing.assert_close(first['loss'], second['loss'])
        self.assertEqual(first['paired_samples'], 3)
        second['loss'].backward()
        self.assertEqual(float(changed.grad[0].abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(changed.grad).all()))

    def test_independent_pooling_retains_longer_view_tail(self):
        a = view('A', self.a, torch.tensor([[1, 0, 0]]*4, dtype=torch.bool))
        b = view('B', self.b, torch.tensor([[1, 1, 1]]*4, dtype=torch.bool))
        torch.testing.assert_close(a.pool().payload, self.a[:, 0])
        torch.testing.assert_close(b.pool().payload, self.b[:, 2])
        torch.testing.assert_close(b.pool('mean').payload, self.b.mean(1))
        objective = PairedViewObjective(8, cfg=ViewObjectiveConfig(agreement_enabled=True))
        first = objective(a, b)
        modified = self.b.detach().clone()
        modified[:, 2] += 10
        second = objective(a, view('B', modified, b.available))
        self.assertNotEqual(float(first['loss'].detach()), float(second['loss'].detach()))

    def test_nonindependent_or_malformed_views_are_rejected(self):
        with self.assertRaises(ValueError):
            SourceView('A', self.a, torch.ones(4, 3, dtype=torch.bool), 'node_after_feedback')
        with self.assertRaises(ValueError):
            SourceView('A', self.a, torch.ones(4, 3))
        with self.assertRaises(ValueError):
            PairedViewObjective(8)(view('B', self.a), view('A', self.b))

    def test_variance_penalizes_collapse_covariance_penalizes_redundancy(self):
        cfg = ViewObjectiveConfig(shared_dim=2, private_dim=2, pooling='none',
                                  variance_enabled=True, covariance_enabled=True)
        objective = PairedViewObjective(2, cfg=cfg)
        with torch.no_grad():
            for layer in objective.shared.values():
                layer.weight.copy_(torch.eye(2));layer.bias.zero_()
        zero = torch.zeros(4, 2)
        collapsed = objective(view('A', zero), view('B', zero))
        self.assertAlmostEqual(float(collapsed['variance'].detach()), .99, places=5)
        repeated = torch.tensor([[-2., -2.], [-1., -1.], [1., 1.], [2., 2.]])
        redundant = objective(view('A', repeated), view('B', repeated))
        self.assertEqual(float(redundant['variance'].detach()), 0)
        self.assertGreater(float(redundant['covariance'].detach()), 1.)

    def test_private_labels_only_supervise_available_own_view(self):
        objective = PairedViewObjective(8, cfg=ViewObjectiveConfig(private_enabled=True))
        ma = torch.tensor([[1, 1, 1]]*4, dtype=torch.bool)
        mb = ma.clone();mb[0] = False
        labels = {'A': torch.tensor([12, 13, 14, 15]), 'B': torch.tensor([90, 91, -100, 92])}
        first = objective(view('A', self.a, ma), view('B', self.b, mb), labels)
        altered = {**labels, 'B': labels['B'].clone()};altered['B'][0] = 255
        second = objective(view('A', self.a, ma), view('B', self.b, mb), altered)
        torch.testing.assert_close(first['private'], second['private'])
        first['loss'].backward()
        self.assertGreater(float(objective.private['A'].weight.grad.abs().sum()), 0)
        self.assertGreater(float(objective.private_readout['B'].weight.grad.abs().sum()), 0)
        self.assertEqual(float(self.b.grad[0].abs().sum()), 0)
        with self.assertRaises(ValueError):
            objective(view('A', self.a), view('B', self.b))

    def test_terms_are_independently_switchable_and_finite_for_single_sample(self):
        objective = PairedViewObjective(8, cfg=ViewObjectiveConfig(prediction_enabled=True, variance_enabled=True,
                                                                  covariance_enabled=True, separation_enabled=True))
        result = objective(view('A', self.a[:1]), view('B', self.b[:1]))
        self.assertEqual(float(result['agreement'].detach()), 0)
        self.assertEqual(float(result['variance'].detach()), 0)
        self.assertEqual(float(result['covariance'].detach()), 0)
        self.assertEqual(float(result['separation'].detach()), 0)
        self.assertTrue(bool(torch.isfinite(result['loss'])))
        result['loss'].backward()
        self.assertGreater(float(objective.predict['A'].weight.grad.abs().sum()), 0)

    def test_fully_missing_views_have_zero_loss_and_gradient(self):
        cfg = ViewObjectiveConfig(agreement_enabled=True, prediction_enabled=True, variance_enabled=True,
                                  covariance_enabled=True, separation_enabled=True, private_enabled=True)
        objective = PairedViewObjective(8, cfg=cfg)
        mask = torch.zeros(4, 3, dtype=torch.bool)
        result = objective(view('A', self.a, mask), view('B', self.b, mask),
                           {'A': torch.arange(4), 'B': torch.arange(4)})
        self.assertEqual(float(result['loss'].detach()), 0)
        self.assertEqual(result['paired_samples'], 0)
        result['loss'].backward()
        self.assertEqual(float(self.a.grad.abs().sum()+self.b.grad.abs().sum()), 0)


class DelayedUtilityChecks(unittest.TestCase):
    def setUp(self):
        self.utility = DelayedUtility(('A__U__forward', 'B__U__forward'), ('U',), laces=4,
                                      enabled=True, decay=.5, gain_clip=2.)
        self.trace = [{'event': 0, 'node': 'U',
                       'dependencies': (dependency('A__U__forward', [True, False]),
                                        dependency('B__U__forward', [False, False])),
                       'routing': {'rows': torch.tensor([0]), 'laces': torch.tensor([[[1, 3], [0, 2], [1, 2]]])}}]

    def packet(self, gain=1., **kwargs):
        baseline = torch.full((2, 1), gain, requires_grad=True)
        actual = torch.zeros(2, 1, requires_grad=True)
        return self.utility.make_packet(self.trace, baseline, actual, **kwargs)

    def test_packet_is_detached_bounded_and_does_not_commit(self):
        packet = self.packet()
        self.assertEqual(int(self.utility.feedback_updates), 0)
        self.assertEqual(float(self.utility.edge_mass.sum()), 0)
        for x in (packet.edge_mass, packet.edge_gain_sum, packet.lace_mass, packet.lace_gain_sum):
            self.assertFalse(x.requires_grad);self.assertIsNone(x.grad_fn)
        self.assertEqual(packet.edge_mass.tolist(), [.5, 0.])
        self.assertEqual(packet.lace_mass.shape, (1, 3, 4))

    def test_positive_and_negative_gains_update_only_used_paths(self):
        self.utility.commit_feedback(self.packet(1.))
        self.assertGreater(float(self.utility.edge_gain_sum[0]), 0)
        self.assertEqual(float(self.utility.edge_mass[1]), 0)
        self.assertEqual(float(self.utility.lace_mass[0, 0, 0]), 0)
        self.assertGreater(float(self.utility.lace_gain_sum[0, 0, 1]), 0)
        self.utility.commit_feedback(self.packet(-2.))
        self.assertLess(float(self.utility.edge_gain_sum[0]), 0)
        self.assertLess(float(self.utility.gate_bias()['A__U__forward']), 0)
        self.assertEqual(float(self.utility.gate_bias()['B__U__forward']), 0)
        biases = self.utility.lace_bias()['U']
        self.assertEqual(biases.shape, (3, 4))
        self.assertFalse(biases.requires_grad)
        self.assertEqual(float(biases[0, 0]), 0)
        self.assertLess(float(biases[0, 1]), 0)
        self.assertEqual(int(self.utility.feedback_updates), 2)

    def test_unobserved_paths_retain_their_previous_associations(self):
        self.utility.commit_feedback(self.packet())
        before = {k: v.clone() for k, v in self.utility.named_buffers()}
        packet = self.packet(supervised=torch.zeros(2, 1, dtype=torch.bool))
        self.utility.commit_feedback(packet)
        for key, x in self.utility.named_buffers():
            torch.testing.assert_close(x, before[key])

    def test_missing_and_substituted_rows_cannot_receive_credit(self):
        self.trace[0]['dependencies'] = (dependency('A__U__forward', [True, True], [True, False]),)
        packet = self.packet(1.)
        self.assertEqual(float(packet.edge_mass[0]), .5)
        packet = self.packet(1., supervised=torch.tensor([[True], [False]]))
        self.assertEqual(float(packet.edge_mass.sum()), 0.)

    def test_per_row_sign_clipping_and_nonfinite_supervision(self):
        self.trace[0]['dependencies'] = (dependency('A__U__forward', [True, True], [True, False]),)
        baseline = torch.tensor([[100.], [-100.]])
        actual = torch.zeros_like(baseline)
        packet = self.utility.make_packet(self.trace, baseline, actual)
        self.utility.commit_feedback(packet)
        self.assertAlmostEqual(float(self.utility.gate_bias(scale=1.)['A__U__forward']), -2.)
        with self.assertRaises(ValueError):
            self.utility.make_packet(self.trace, baseline.fill_(float('inf')), actual)

    def test_eval_and_disabled_paths_do_not_mutate(self):
        packet = self.packet()
        self.utility.eval()
        with self.assertRaises(RuntimeError):
            self.utility.commit_feedback(packet)
        self.assertEqual(int(self.utility.feedback_updates), 0)
        self.utility.train();self.utility.enabled = False
        self.assertIsNone(self.packet())
        self.utility.commit_feedback(packet)
        self.assertEqual(int(self.utility.feedback_updates), 0)
        self.assertTrue(all(float(v) == 0 for v in self.utility.gate_bias().values()))
        self.assertTrue(all(bool((v == 0).all()) for v in self.utility.lace_bias().values()))

    def test_incompatible_and_nonfinite_packets_are_rejected_before_mutation(self):
        good = self.packet()
        with self.assertRaises(ValueError):
            self.utility.commit_feedback(replace(good, edge_names=('other',)))
        bad = replace(good, edge_gain_sum=torch.full_like(good.edge_gain_sum, float('nan')))
        with self.assertRaises(ValueError):
            self.utility.commit_feedback(bad)
        self.assertEqual(int(self.utility.feedback_updates), 0)
        self.assertEqual(float(self.utility.edge_mass.sum()), 0)

    def test_repeated_rounds_count_execution_and_gains_remain_normalized(self):
        self.trace = self.trace*3
        packet = self.packet(1.)
        self.assertEqual(float(packet.edge_mass[0]), 1.5)
        self.utility.commit_feedback(packet)
        self.assertAlmostEqual(float(self.utility.gate_bias(scale=1.)['A__U__forward']), 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
