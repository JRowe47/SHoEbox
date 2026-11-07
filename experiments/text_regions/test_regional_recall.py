"""Regional recall preserves causality, ownership, masking and continuation."""
from dataclasses import replace
import unittest
import torch

from .config import GraphConfig
from .graph import EdgeSubstitution
from .model import RegionLanguageModel
from .ports import tensor_list
from .state import GraphState

torch.set_num_threads(1)


class RegionalRecallChecks(unittest.TestCase):
    def model(self, **kwargs):
        torch.manual_seed(347)
        return RegionLanguageModel(GraphConfig.tiny(recall_slots=2, **kwargs))

    def same(self, a, b, exact=True):
        left, right = tensor_list(a), tensor_list(b)
        self.assertEqual(len(left), len(right))
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, rtol=0 if exact else 1e-4, atol=0 if exact else 3e-6)

    def test_legacy_initialization_rng_config_and_state_migration(self):
        torch.manual_seed(347)
        legacy = RegionLanguageModel(GraphConfig.tiny())
        rng = torch.get_rng_state()
        model = self.model()
        self.same(rng, torch.get_rng_state())
        for key, value in legacy.state_dict().items():
            self.same(value, model.state_dict()[key])
        config = legacy.cfg.to_dict()
        for key in tuple(config):
            if key.startswith('recall_'):
                config.pop(key)
        restored = RegionLanguageModel(GraphConfig.from_dict(config))
        restored.load_state_dict(legacy.state_dict(), strict=True)
        state = legacy.initial_state(2).to_dict(); state.pop('recall_feedback')
        old_state = GraphState.from_dict(state)
        x = torch.tensor([[65, 66], [67, 68]])
        self.same(legacy(x)['logits'], restored(x, state=old_state)['logits'])
        result = model(x, state=old_state)
        self.assertEqual(set(result['state'].recall_feedback), set(model.nodes))

    @torch.no_grad()
    def test_delayed_snapshot_timing_masks_and_seven_slots(self):
        model = self.model(rounds=2, recurrent_feedback=True)
        x = torch.tensor([[65, 66], [67, 68]])
        mask = torch.tensor([[False, True], [True, False]])
        result = model(x, edge_masks={'A__U__forward': mask}, return_recall_activity=True)
        record = result['recall_activity']['U']; edges = result['recall_edges']['U']
        a, delayed = edges.index('A__U__forward'), edges.index('P__U__descending')
        self.same(record['valid'][..., a], mask[:, :, None].expand(-1, -1, 2))
        self.assertFalse(bool(record['valid'][:, 0, 0, delayed].any()))
        self.assertTrue(bool(record['valid'][:, 0, 1, delayed].all()))
        self.same(record['round_indices'][:, 0, 1, delayed], torch.zeros(2, dtype=torch.long))
        self.same(record['versions'][:, 0, :, a], torch.tensor([[1, 2], [1, 2]]))
        for rec in result['recall_activity'].values():
            self.assertEqual(float((rec['weights'][..., :-1] * ~rec['valid'][..., None, :]).sum()), 0.)
            self.same(rec['weights'].sum(-1), torch.ones_like(rec['weights'][..., 0]))
        seven = self.model(recall_nodes=('P',))
        cfg = replace(seven.cfg, recall_slots=7)
        seven = RegionLanguageModel(cfg)
        out = seven(x, feedback=False, return_recall_activity=True)
        self.assertEqual(out['recall_activity']['P']['weights'].shape, (2, 2, 1, 7, 4))
        self.assertFalse(bool(out['recall_activity']['P']['valid'][..., -1].any()))

    @torch.no_grad()
    def test_masked_substitutions_and_blocked_inputs_do_not_leak(self):
        model = self.model(recall_nodes=('P',))
        state = model(torch.tensor([[65, 66], [67, 68]]))['state']
        x = torch.tensor([[69], [70]])
        a = EdgeSubstitution(torch.randn(2, 1, 32), torch.randn(2, 1, 3, 4))
        b = EdgeSubstitution(a.payload + 100, a.memberships - 100)
        controls = dict(state=state, edge_masks={'A__P__forward': torch.zeros(2, 1, dtype=torch.bool)})
        left = model(x, substitutions={'A__P__forward': a}, **controls)
        right = model(x, substitutions={'A__P__forward': b}, **controls)
        self.same(left['logits'], right['logits'])
        controls.pop('edge_masks')
        left = model(x, substitutions={'A__P__forward': a}, recall_control='blocked', **controls)
        right = model(x, substitutions={'A__P__forward': b}, recall_control='blocked', **controls)
        self.same(left['logits'], right['logits'])

    def test_gradients_reach_all_recall_controls_and_candidate_sources(self):
        model = self.model(recall_nodes=('P', 'Q'), recurrent_feedback=True)
        x = torch.tensor([list(b'hello'), list(b'world')])
        result = model(x, x, return_recall_activity=True)
        result['loss'].backward()
        for name, param in model.connections.recall.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(bool(torch.isfinite(param.grad).all()), name)
            self.assertGreater(float(param.grad.norm()), 0, name)
        self.assertGreater(float(model.connections.edges['A__P__forward'].value.weight.grad.norm()), 0.)

    @torch.no_grad()
    def test_chunk_serialization_reset_inactive_rows_and_stage_order(self):
        model = self.model(recurrent_feedback=True, rounds=2)
        x = torch.tensor([list(b'hello'), list(b'world')])
        full = model(x, return_recall_activity=True)
        first = model(x[:, :2])
        state = GraphState.from_dict(first['state'].to_dict())
        second = model(x[:, 2:], state=state)
        self.same(full['logits'], torch.cat((first['logits'], second['logits']), 1))
        self.same(full['state'], second['state'])
        self.same(full['logits'], model(x, reverse_independent=True)['logits'])
        self.same(full['logits'], model(x)['logits'])
        reset = state.reset(torch.tensor([True, False]))
        for name, value in reset.recall_feedback.items():
            self.same(value[0], torch.zeros_like(value[0]))
            self.same(value[1], state.recall_feedback[name][1])
        paused = model(x[:, :1], state=state,
            node_masks={'P': torch.tensor([[False], [True]])})['state']
        self.same(paused.recall_feedback['P'][0], state.recall_feedback['P'][0])
        fresh = model(x[:, :1])['logits'][0]
        self.same(model(x[:, :1], state=state, reset=torch.tensor([True, False]))['logits'][0], fresh)

    @torch.no_grad()
    def test_reference_backend_and_feedback_ablation(self):
        model = self.model(recurrent_feedback=True)
        for port in model.connections.recall.values():
            port.workspace.query_feedback.weight.normal_(std=.3)
            port.workspace.null_feedback.weight.normal_(std=.3)
        x = torch.tensor([list(b'hello'), list(b'world')])
        active = model(x)
        self.assertFalse(torch.equal(active['logits'], model(x, recall_control='no_feedback')['logits']))
        model.set_execution_backend('reference')
        other = model(x)
        self.same(active['logits'], other['logits'], exact=False)
        self.same(active['state'], other['state'], exact=False)

    @torch.no_grad()
    def test_request_feedback_advances_when_publication_is_held(self):
        model = self.model(recall_nodes=('P',), selective_publication=('P',))
        x = torch.tensor([list(b'hello'), list(b'world')])
        state = model(x)['state']
        result = model(x.flip(1), state=state, publication_overrides={'P': 'hold'})
        self.same(result['state'].ports.latest['P'].payload, state.ports.latest['P'].payload)
        self.same(result['state'].ports.latest['P'].version, state.ports.latest['P'].version)
        self.assertFalse(torch.equal(result['state'].recall_feedback['P'], state.recall_feedback['P']))


if __name__ == '__main__':
    unittest.main()
