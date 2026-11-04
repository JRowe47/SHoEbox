"""Query-only interventions, paired budgets, and evaluation isolation."""
from dataclasses import replace
import unittest
import torch
from torch import nn
from .config import GraphConfig
from .learning import RegionLearningModel
from .model import RegionLanguageModel
from .query_interference import VARIANTS, evaluate_query_interference, query_availability
from .view_data import make_view_batch
from .view_train import tensor_fingerprint

torch.set_num_threads(2)


class QueryModel(nn.Module):
    """Known predictions; the signature rejects labels entering inference."""
    def __init__(self, change_budget=False, mutate=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.model = nn.Module()
        self.model.cfg = GraphConfig.tiny(evidence_gated_ports=True)
        self.child = nn.Dropout()
        self.register_buffer('utility_state', torch.tensor([.5, -.25]))
        self.change_budget, self.mutate = change_budget, mutate

    @property
    def cfg(self):
        return self.model.cfg

    def forward(self, inputs, *, reset, availability):
        batch, length = inputs['B'].shape
        logits = torch.zeros(batch, length, 256)
        logits[..., ord('A')] = 1
        logits[..., ord('D')] = (availability['A'] & availability['C']).float()*2
        if self.mutate:
            self.utility_state.add_(1)
        extra = int(bool(availability['A'].any())) if self.change_budget else 0
        return {'logits': logits, 'expert_executions': batch*length*84+extra,
                'external_events': batch*length, 'source_occurrences': sum(int(mask.sum()) for mask in availability.values())}


class QueryInterferenceChecks(unittest.TestCase):
    def test_only_actual_query_markers_are_added_and_labels_cannot_select_them(self):
        for gap in (0, 8):
            batch = make_view_batch('independent', 0, 4, seed=81, gap=gap, query_type='shared', control='b_only')
            original = {view: values.clone() for view, values in batch.inputs.items()}
            bad_labels = replace(batch, targets=torch.full_like(batch.targets, 255), answers=torch.full_like(batch.answers, 254))
            for variant in VARIANTS:
                available = query_availability(batch, variant)
                alternative = query_availability(bad_labels, variant)
                for view in available:
                    self.assertTrue(torch.equal(available[view], alternative[view]))
                expected = torch.zeros_like(batch.targets, dtype=torch.bool)
                if variant == 'final_query_ac':
                    expected[:, -1] = True
                elif variant == 'all_queries_ac':
                    expected = batch.targets != -100
                for view in ('A', 'C'):
                    self.assertTrue(torch.equal(available[view], expected))
                    self.assertTrue(bool((batch.inputs[view][available[view]] == ord('S')).all()))
                    for positions in batch.fact_positions[view].values():
                        rows = (positions >= 0).nonzero().flatten()
                        self.assertFalse(bool(available[view][rows, positions[rows]].any()))
                self.assertTrue(torch.equal(available['B'], batch.availability['B']))
            for view, values in original.items():
                self.assertTrue(torch.equal(values, batch.inputs[view]))

    def test_invalid_source_observations_or_fact_overlap_are_rejected(self):
        batch = make_view_batch('independent', 0, 2, query_type='shared', control='b_only')
        bad_input = batch.inputs['A'].clone();bad_input[:, -1] = ord('X')
        with self.assertRaises(ValueError):
            query_availability(replace(batch, inputs={**batch.inputs, 'A': bad_input}), 'final_query_ac')
        bad_positions = {**batch.fact_positions, 'A': {**batch.fact_positions['A'],
                          'shared': torch.full((2,), batch.length-1)}}
        with self.assertRaises(ValueError):
            query_availability(replace(batch, fact_positions=bad_positions), 'final_query_ac')
        with self.assertRaises(ValueError):
            query_availability(make_view_batch('independent', 0, 2, query_type='shared'), 'b_only')
        with self.assertRaises(ValueError):
            query_availability(make_view_batch('independent', 0, 2, query_type='left', control='b_only'), 'b_only')

    def test_paired_policies_preserve_cases_budgets_tensors_modes_config_and_rng(self):
        model = QueryModel().train();model.child.eval()
        fingerprint, config, rng = tensor_fingerprint(model), model.cfg, torch.get_rng_state().clone()
        result = evaluate_query_interference(model, policy='both', count=64, batch_size=17, gap=0, include_predictions=True)
        self.assertTrue(result['model_tensors_unchanged'])
        self.assertEqual(len(result['predictions']), 6)
        costs, cases, inputs = set(), set(), set()
        for policy, rows in result['results'].items():
            for variant, score in rows.items():
                self.assertEqual(score['exact_accuracy'], .25)
                self.assertEqual(score['evidence_gated_ports'], policy == 'evidence')
                self.assertTrue(all(row['episodes'] == 16 for row in score['by_answer'].values()))
                self.assertEqual(sum(row['episodes'] for row in score['by_B_template'].values()), 64)
                self.assertEqual(set(score['query_accuracy']), {'14', '15', '16'})
                costs.add(score['expert_executions']);cases.add(score['episode_sha256']);inputs.add(score['inputs_sha256'])
        self.assertEqual(costs, {64*17*84});self.assertEqual(len(cases), 1);self.assertEqual(len(inputs), 1)
        self.assertEqual(tensor_fingerprint(model), fingerprint)
        self.assertIs(model.cfg, config)
        self.assertTrue(model.training);self.assertFalse(model.child.training)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_budget_or_tensor_mutation_fails_and_restores_inference_policy(self):
        for kwargs in ({'change_budget': True}, {'mutate': True}):
            model = QueryModel(**kwargs).train()
            config = model.cfg
            with self.assertRaises((ValueError, RuntimeError)):
                evaluate_query_interference(model, policy='original', count=64, batch_size=64, gap=0)
            self.assertIs(model.cfg, config)
            self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            evaluate_query_interference(QueryModel(), count=65)

    def test_real_graph_keeps_expert_work_and_observes_only_added_queries(self):
        torch.manual_seed(913)
        model = RegionLearningModel(RegionLanguageModel(GraphConfig.tiny(evidence_gated_ports=True))).eval()
        batch = make_view_batch('independent', 0, 2, seed=12, gap=0, query_type='shared', control='b_only')
        fingerprint, costs = tensor_fingerprint(model), set()
        with torch.no_grad():
            for variant in VARIANTS:
                available = query_availability(batch, variant)
                result = model(batch.inputs, reset=batch.reset, availability=available)
                costs.add(result['expert_executions'])
                for view in ('A', 'C'):
                    self.assertEqual(result['state'].ports.latest[view].valid.tolist(), [variant != 'b_only']*2)
                    clocks = result['state'].ingress[view].clocks
                    observed = clocks.get(0, torch.zeros(2, dtype=torch.long))
                    self.assertTrue(torch.equal(observed, available[view].sum(1)))
        self.assertEqual(len(costs), 1)
        self.assertEqual(tensor_fingerprint(model), fingerprint)


if __name__ == '__main__':
    unittest.main(verbosity=2)
