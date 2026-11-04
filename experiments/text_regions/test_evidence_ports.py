"""Optional evidence-backed publication validity at unchanged expert budgets."""
from dataclasses import replace
import unittest
import torch
from .config import GraphConfig
from .model import RegionLanguageModel
from .objectives import DelayedUtility

torch.set_num_threads(2)


class EvidencePortChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(632)
        self.cfg = GraphConfig.tiny(evidence_gated_ports=True)
        self.model = RegionLanguageModel(self.cfg).eval()
        self.x = torch.tensor([[65], [66]])
        self.inputs = {'A': self.x, 'B': self.x+3, 'C': self.x+6}
        self.absent = torch.zeros_like(self.x, dtype=torch.bool)
        self.b_only = {'A': self.absent, 'C': self.absent}

    def ordinary(self):
        model = RegionLanguageModel(replace(self.cfg, evidence_gated_ports=False)).eval()
        model.load_state_dict(self.model.state_dict())
        return model

    def test_all_present_keeps_exact_predictions_losses_state_and_parameter_count(self):
        ordinary = self.ordinary()
        first = ordinary(self.inputs, self.x+1)
        second = self.model(self.inputs, self.x+1)
        for key in ('logits', 'loss', 'ce', 'aux', 'hat_aux'):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        for a, b in zip(first['state'].tensors(), second['state'].tensors()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(first['expert_executions'], second['expert_executions'])
        self.assertEqual(sum(p.numel() for p in ordinary.parameters()), sum(p.numel() for p in self.model.parameters()))
        self.assertFalse(GraphConfig.tiny().evidence_gated_ports)
        self.assertTrue(GraphConfig.from_dict(self.cfg.to_dict()).evidence_gated_ports)
        legacy = self.cfg.to_dict();legacy.pop('evidence_gated_ports')
        self.assertFalse(GraphConfig.from_dict(legacy).evidence_gated_ports)

    def test_cold_missing_sources_stay_invalid_without_reducing_expert_work(self):
        ordinary = self.ordinary()(self.inputs, availability=self.b_only)
        result = self.model(self.inputs, availability=self.b_only, trace=True)
        self.assertEqual(ordinary['expert_executions'], result['expert_executions'])
        self.assertEqual(ordinary['source_occurrences'], result['source_occurrences'])
        for name in ('A', 'C'):
            self.assertTrue(bool(ordinary['state'].ports.latest[name].valid.all()))
            self.assertFalse(bool(result['state'].ports.latest[name].valid.any()))
            self.assertEqual(result['state'].nodes[name].clock.tolist(), [1, 1])
        for name in ('B', 'U', 'P', 'V', 'Q'):
            self.assertTrue(bool(result['state'].ports.latest[name].valid.all()))
        for record in result['trace']:
            self.assertEqual(record['valid'].tolist(), [record['node'] not in ('A', 'C')]*2)

    def test_earlier_evidence_persists_through_absence_and_selective_reset_clears_it(self):
        observed = self.model(self.inputs)['state'].detach()
        unavailable = {view: self.absent for view in self.inputs}
        retained = self.model(self.inputs, state=observed, availability=unavailable)['state'].detach()
        for name in self.model.nodes:
            self.assertEqual(retained.ports.latest[name].valid.tolist(), [True, True])
        reset = self.model(self.inputs, state=retained, availability=unavailable,
                           reset=torch.tensor([True, False]))['state']
        for name in self.model.nodes:
            self.assertEqual(reset.ports.latest[name].valid.tolist(), [False, True])
        self.assertEqual(reset.epochs.tolist(), [1, 0])

    def test_inactive_node_cannot_gain_validity_from_available_upstream_evidence(self):
        result = self.model(self.inputs, availability={'B': self.absent, 'C': self.absent},
                            node_masks={'A': torch.tensor([[False], [True]])}, trace=True)
        self.assertEqual(result['state'].ports.latest['@A'].valid.tolist(), [True, True])
        self.assertEqual(result['state'].ports.latest['A'].valid.tolist(), [False, True])
        self.assertEqual(result['state'].nodes['A'].clock.tolist(), [0, 1])
        self.assertEqual(result['state'].ports.latest['P'].valid.tolist(), [False, True])
        record = next(record for record in result['trace'] if record['node'] == 'A')
        self.assertEqual(record['valid'].tolist(), [False, True])

    def test_second_round_feedback_can_validate_a_source_with_no_arriving_bytes(self):
        result = self.model(self.inputs, availability=self.b_only, rounds=2, feedback=True, trace=True)
        for source, edge in (('A', 'P__A__descending'), ('C', 'Q__C__descending')):
            records = [record for record in result['trace'] if record['node'] == source]
            self.assertEqual(records[0]['valid'].tolist(), [False, False])
            self.assertEqual(records[1]['valid'].tolist(), [True, True])
            consumed = next(dependency.consumed for dependency in records[1]['dependencies'] if dependency.edge == edge)
            self.assertTrue(bool(consumed.all()))
        ordinary = self.ordinary()(self.inputs, availability=self.b_only, rounds=2, feedback=True)
        self.assertEqual(result['expert_executions'], ordinary['expert_executions'])

    def test_zero_is_a_present_byte_and_can_establish_valid_evidence(self):
        inputs = {view: torch.zeros_like(self.x) for view in self.inputs}
        result = self.model(inputs)
        for name in (*self.model.nodes, '@A', '@B', '@C'):
            self.assertEqual(result['state'].ports.latest[name].valid.tolist(), [True, True])

    def test_invalid_publications_receive_no_lace_or_edge_utility(self):
        result = self.model(self.inputs, availability=self.b_only, trace=True)
        utility = DelayedUtility(tuple(self.model.connections.edges), tuple(self.model.nodes),
                                 self.cfg.laces, enabled=True)
        packet = utility.make_packet(result['trace'], torch.ones(2, 1), torch.zeros(2, 1))
        for name in ('A', 'C'):
            index = packet.node_names.index(name)
            self.assertEqual(float(packet.lace_mass[index].sum()), 0)
            self.assertEqual(float(packet.lace_gain_sum[index].sum()), 0)
        for name in ('B', 'U', 'P', 'V', 'Q'):
            self.assertGreater(float(packet.lace_mass[packet.node_names.index(name)].sum()), 0)
        for name in ('A__U__forward', 'A__P__forward', 'C__V__forward', 'C__Q__forward'):
            self.assertEqual(float(packet.edge_mass[packet.edge_names.index(name)]), 0)
        self.assertGreater(float(packet.edge_mass[packet.edge_names.index('B__U__forward')]), 0)
        self.assertEqual(int(utility.feedback_updates), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
