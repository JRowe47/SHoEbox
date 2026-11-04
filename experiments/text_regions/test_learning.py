"""Real-pass objective orchestration, commit boundaries, and checkpoint resume."""
import io
import unittest
import torch
from torch.nn import functional as F
from .config import GraphConfig
from .graph import EdgeSubstitution
from .learning import RegionLearningModel
from .model import RegionLanguageModel
from .objectives import DelayedUtility, PairedViewObjective, ViewObjectiveConfig
from .state import GraphState

torch.set_num_threads(2)


class LearningChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(832)
        self.cfg = GraphConfig.tiny(rounds=2, recurrent_feedback=True)
        self.model = RegionLanguageModel(self.cfg)
        self.x = torch.tensor([[65, 66], [70, 71]])
        self.targets = self.x + 1
        self.inputs = {'A': self.x, 'B': self.x+5, 'C': self.x+10}

    def utility(self):
        return DelayedUtility(tuple(self.model.connections.edges), tuple(self.model.nodes), self.cfg.laces,
                              enabled=True, decay=.5)

    def assert_states_equal(self, first, second):
        for a, b in zip(first.tensors(), second.tensors()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_disabled_wrapper_preserves_outputs_loss_state_and_parameter_count(self):
        rng = torch.get_rng_state().clone()
        learner = RegionLearningModel(self.model)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in learner.parameters()), sum(p.numel() for p in self.model.parameters()))
        plain = self.model(self.x, self.targets)
        wrapped = learner(self.x, self.targets)
        self.assertEqual(plain.keys(), wrapped.keys())
        for key in ('logits', 'loss', 'ce', 'aux', 'hat_aux'):
            torch.testing.assert_close(plain[key], wrapped[key], rtol=0, atol=0)
        self.assert_states_equal(plain['state'], wrapped['state'])
        self.assertIsNone(learner(self.x)['loss'])

    def test_view_losses_reach_independent_ingress_and_labels_never_change_inference(self):
        objective = PairedViewObjective(self.cfg.width, cfg=ViewObjectiveConfig(
            agreement_enabled=True, prediction_enabled=True, private_enabled=True))
        learner = RegionLearningModel(self.model, view_objective=objective)
        labels = {'A': torch.tensor([7, 8]), 'B': torch.tensor([30, 40])}
        first = learner(self.inputs, self.targets, private_targets=labels)
        other = learner(self.inputs, self.targets, private_targets={k: v+1 for k, v in labels.items()})
        torch.testing.assert_close(first['logits'], other['logits'], rtol=0, atol=0)
        self.assert_states_equal(first['state'], other['state'])
        for name in objective.views:
            torch.testing.assert_close(first['view_objective']['codes'][name]['shared'],
                                       other['view_objective']['codes'][name]['shared'], rtol=0, atol=0)
        torch.testing.assert_close(first['loss'], first['language_loss']+first['view_objective']['loss'])
        first['view_objective']['loss'].backward()
        self.assertGreater(float(self.model.hat.byte_embedding.weight.grad.abs().sum()), 0)
        self.assertGreater(float(objective.private_readout['A'].weight.grad.abs().sum()), 0)
        self.assertIsNone(self.model.readout.weight.grad)

    def test_local_baseline_uses_available_source_targets_and_reports_weighted_loss(self):
        learner = RegionLearningModel(self.model, baseline_enabled=True, baseline_view='A', baseline_weight=.3)
        available = torch.tensor([[True, False], [False, True]])
        targets = self.targets.clone();targets[1, 1] = -100
        result = learner(self.inputs, targets, availability={'A': available})
        expected = F.cross_entropy(result['baseline_logits'][0, :1], targets[0, :1])
        torch.testing.assert_close(result['baseline_ce'], expected)
        torch.testing.assert_close(result['baseline_loss'], expected*.3)
        torch.testing.assert_close(result['loss'], result['language_loss']+expected*.3)
        result['baseline_loss'].backward()
        self.assertGreater(float(learner.baseline.weight.grad.abs().sum()), 0)
        self.assertIsNone(result['utility_packet'])

    def test_detached_baseline_preserves_values_head_gradient_and_utility_but_isolates_graph(self):
        attached = RegionLearningModel(self.model, utility=self.utility(), baseline_view='A', baseline_enabled=True)
        detached = RegionLearningModel.from_config({**attached.get_config(), 'baseline_detach_features': True})
        detached.load_state_dict(attached.state_dict())
        available = torch.tensor([[True, False], [True, True]])
        targets = self.targets.clone(); targets[1, 1] = -100
        first = attached(self.inputs, targets, availability={'A': available})
        second = detached(self.inputs, targets, availability={'A': available})
        for key in ('logits', 'baseline_logits', 'baseline_ce', 'baseline_loss', 'language_loss', 'loss'):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        self.assert_states_equal(first['state'], second['state'])
        for key, value in vars(first['utility_packet']).items():
            other = getattr(second['utility_packet'], key)
            if torch.is_tensor(value):
                torch.testing.assert_close(value, other, rtol=0, atol=0)
            else:
                self.assertEqual(value, other)
        self.assertEqual(second['utility_packet'].supervised_events, 2)
        first['baseline_loss'].backward(retain_graph=True)
        second['baseline_loss'].backward(retain_graph=True)
        torch.testing.assert_close(attached.baseline.weight.grad, detached.baseline.weight.grad, rtol=0, atol=0)
        self.assertGreater(float(detached.baseline.weight.grad.abs().sum()), 0)
        self.assertGreater(float(attached.model.hat.byte_embedding.weight.grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in detached.model.parameters()))
        detached.zero_grad(set_to_none=True)
        second['language_loss'].backward()
        self.assertGreater(float(detached.model.hat.byte_embedding.weight.grad.abs().sum()), 0)
        self.assertGreater(float(detached.model.readout.weight.grad.abs().sum()), 0)
        self.assertIsNone(detached.baseline.weight.grad)

    def test_detach_config_roundtrip_and_legacy_default(self):
        original = RegionLearningModel(self.model, baseline_enabled=True, baseline_detach_features=True)
        restored = RegionLearningModel.from_config(original.get_config())
        self.assertTrue(restored.baseline_detach_features)
        self.assertEqual(restored.get_config(), original.get_config())
        legacy = original.get_config(); legacy.pop('baseline_detach_features')
        restored_legacy = RegionLearningModel.from_config(legacy)
        self.assertFalse(restored_legacy.baseline_detach_features)
        self.assertEqual(restored_legacy.get_config(), {**legacy, 'baseline_detach_features': False})

    def test_private_trained_model_supports_inference_without_private_labels(self):
        objective = PairedViewObjective(self.cfg.width, cfg=ViewObjectiveConfig(private_enabled=True))
        learner = RegionLearningModel(self.model, view_objective=objective).eval()
        prediction = learner(self.inputs)
        expected = self.model(self.inputs)
        torch.testing.assert_close(prediction['logits'], expected['logits'], rtol=0, atol=0)
        self.assertIsNone(prediction['loss'])
        self.assertIsNone(prediction['view_objective'])
        scored = learner(self.inputs, self.targets)
        torch.testing.assert_close(scored['loss'], scored['language_loss'], rtol=0, atol=0)
        learner.train()
        with self.assertRaises(ValueError):
            learner(self.inputs, self.targets)

    def test_utility_packet_is_detached_and_only_explicit_commit_changes_associations(self):
        learner = RegionLearningModel(self.model, utility=self.utility(), utility_scale=.8)
        before = {k: v.clone() for k, v in learner.utility.named_buffers()}
        result = learner(self.x, self.targets)
        packet = result['utility_packet']
        self.assertIsNotNone(packet)
        self.assertEqual(result['trace'], [])
        for value in (packet.edge_mass, packet.edge_gain_sum, packet.lace_mass, packet.lace_gain_sum):
            self.assertFalse(value.requires_grad);self.assertIsNone(value.grad_fn)
        for key, value in learner.utility.named_buffers():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        optimizer = torch.optim.Adam(learner.parameters(), lr=1e-3)
        result['loss'].backward()
        self.assertGreater(float(learner.baseline.weight.grad.abs().sum()), 0)
        optimizer.step()
        self.assertEqual(int(learner.utility.feedback_updates), 0)
        learner.commit_feedback(packet)
        self.assertEqual(int(learner.utility.feedback_updates), 1)
        self.assertGreater(float(learner.utility.edge_mass.sum()), 0)
        self.assertGreater(float(learner.utility.lace_mass.sum()), 0)
        expected = self.model(self.x, edge_bias=learner.utility.gate_bias(.8),
                              lace_bias=learner.utility.lace_bias(.8))
        actual = learner(self.x)
        torch.testing.assert_close(actual['logits'], expected['logits'], rtol=0, atol=0)
        self.assert_states_equal(actual['state'], expected['state'])
        self.assertEqual(int(learner.utility.feedback_updates), 1)

    def test_packet_respects_source_availability_masks_and_substitutions(self):
        learner = RegionLearningModel(self.model, utility=self.utility(), baseline_view='A')
        unavailable = torch.tensor([[False, False], [True, True]])
        mask = {'A__U__forward': torch.zeros_like(self.x, dtype=torch.bool)}
        sub = EdgeSubstitution(torch.randn(2, 2, self.cfg.width), torch.zeros(2, 2, 3, self.cfg.groups))
        result = learner(self.inputs, self.targets, availability={'A': unavailable}, edge_masks=mask,
                         substitutions={'B__U__forward': sub}, trace=True)
        packet = result['utility_packet']
        self.assertEqual(packet.supervised_events, 2)
        for key in ('A__U__forward', 'B__U__forward'):
            self.assertEqual(float(packet.edge_mass[packet.edge_names.index(key)]), 0)
        self.assertGreater(float(packet.edge_mass[packet.edge_names.index('C__V__forward')]), 0)
        self.assertTrue(result['trace'])

    def test_ignored_target_chunks_are_legal_and_do_not_create_feedback(self):
        ignored = torch.full_like(self.targets, -100)
        for learner in (RegionLearningModel(self.model), RegionLearningModel(self.model, utility=self.utility())):
            result = learner(self.x, ignored)
            self.assertEqual(float(result['loss'].detach()), 0)
            self.assertEqual(float(result['ce'].detach()), 0)
            self.assertIsNone(result.get('utility_packet'))
            result['loss'].backward()
            learner.zero_grad(set_to_none=True)

    def test_eval_does_not_prepare_or_commit_feedback(self):
        learner = RegionLearningModel(self.model, utility=self.utility()).eval()
        before = {k: v.clone() for k, v in learner.utility.named_buffers()}
        result = learner(self.x, self.targets)
        self.assertIsNone(result['utility_packet'])
        self.assertEqual(result['trace'], [])
        with self.assertRaises(RuntimeError):
            learner.commit_feedback(None)
        for key, value in learner.utility.named_buffers():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_config_state_dict_and_carried_state_resume(self):
        objective = PairedViewObjective(self.cfg.width, cfg=ViewObjectiveConfig(agreement_enabled=True))
        learner = RegionLearningModel(self.model, view_objective=objective, utility=self.utility(), baseline_view='A')
        first = learner(self.inputs, self.targets)
        learner.commit_feedback(first['utility_packet'])
        stream = io.BytesIO()
        torch.save({'config': learner.get_config(), 'weights': learner.state_dict(),
                    'state': first['state'].detach().to_dict()}, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = RegionLearningModel.from_config(saved['config'])
        restored.load_state_dict(saved['weights'])
        state = GraphState.from_dict(saved['state'])
        original = learner(self.inputs, self.targets, state=first['state'].detach())
        resumed = restored(self.inputs, self.targets, state=state)
        self.assertEqual(restored.get_config(), learner.get_config())
        for key in ('logits', 'loss', 'baseline_logits', 'baseline_ce'):
            torch.testing.assert_close(original[key], resumed[key], rtol=0, atol=0)
        self.assert_states_equal(original['state'], resumed['state'])
        torch.testing.assert_close(original['utility_packet'].edge_gain_sum, resumed['utility_packet'].edge_gain_sum,
                                   rtol=0, atol=0)
        self.assertEqual(int(restored.utility.feedback_updates), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
