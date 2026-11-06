"""Whole-query interventions and state-batching integrity."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import torch
from .ports import tensor_list
from .task_memory import TaskMemoryConfig, TaskMemoryModel
from .task_stream import TaskStreamConfig
from .task_query_audit import (select_state, concatenate_states,
    shuffle_publication_content, aged_publication_shuffle, query_forks, audit)

torch.set_num_threads(1)


class QueryAuditChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(52)
        self.model = TaskMemoryModel(TaskMemoryConfig(width=32, owners=3,
            experts_per_block=32, shared_per_block=16, groups=4,
            address_dim=8, laces=16, hat_symbols=16)).eval()
        with torch.no_grad():
            self.state = self.model(torch.tensor([list(b'A x=1;'), list(b'B x=3;')]),
                                    write_override='always')['state']

    def same(self, left, right):
        a, b = tensor_list(left), tensor_list(right)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_state_rebatching_preserves_owner_rows_and_clocks(self):
        combined = concatenate_states([select_state(self.state, [0]), select_state(self.state, [1])])
        self.same(combined, self.state)
        self.assertEqual(combined.ingress.batch_size, 2)
        self.assertEqual(select_state(self.state, [1]).producer.clock.shape, (3,))

    def test_shuffle_changes_content_without_changing_recency_or_ownership(self):
        snapshot = self.state.clone()
        shuffled = shuffle_publication_content(self.state, [1, 0])
        self.same(self.state, snapshot)
        self.same(self.state.ingress, shuffled.ingress)
        for name, pub in shuffled.publications.items():
            original = self.state.publications[name]
            self.same(pub.payload, original.payload.flip(0))
            self.same(pub.header.memberships, original.header.memberships.flip(0))
            for field in ('version', 'occurrence', 'epoch', 'valid'):
                self.same(getattr(pub, field), getattr(original, field))
            self.assertEqual(pub.source, name)

    def test_query_forks_have_equal_budgets_and_hold_prequery_versions(self):
        snapshot = self.state.clone()
        values = torch.tensor([list(b'A x?'), list(b'B x?')])
        outputs = query_forks(self.model, values, self.state, donors=[1, 0])
        self.same(self.state, snapshot)
        budgets = {item['result']['expert_executions'] for item in outputs.values()}
        self.assertEqual(len(budgets), 1)
        self.same(outputs['hold']['result']['state'].publications, self.state.publications)
        self.same(outputs['hold_blocked']['result']['state'].publications, self.state.publications)
        old_versions = torch.stack([p.version for p in self.state.publications.values()], 1)
        self.same(outputs['hold']['result']['read_versions'], old_versions[:, None].expand(-1, 4, -1))
        self.assertEqual([r['read_ages'][0] for r in outputs['hold']['trace'][0]], [1, 2, 3, 4])
        self.assertTrue(all(r['effective_read_mass'] == 0 for r in outputs['blocked']['trace'][0]))
        self.assertGreater(float((outputs['intact']['result']['logits'][:, -1]-
                                  outputs['blocked']['result']['logits'][:, -1]).abs().max()), 1e-6)

    def test_aged_shuffle_counts_eligible_and_actual_changes_separately(self):
        memory = self.state.memory_tensor().clone()
        memory[1, 0] = memory[0, 0]+1  # An old cell with different donor content.
        memory[1, 1] = memory[0, 1]  # An old but identical bootstrap-like cell.
        memory[1, 2] = memory[0, 2]+1  # A recent cell must stay unchanged.
        state = self.state.with_memory_tensor(memory)
        state = replace(state, publications={name: replace(pub,
            occurrence=torch.tensor([2 if i < 2 else 3, 5]))
            for i, (name, pub) in enumerate(state.publications.items())})
        snapshot = state.clone()
        shuffled, info = aged_publication_shuffle(state, [1, 0], age_threshold=3)
        self.same(state, snapshot)
        self.assertEqual(info['age_eligible'].tolist(), [[True, True, False], [False, False, False]])
        self.assertEqual(info['replaced'].tolist(), info['age_eligible'].tolist())
        self.assertEqual(info['content_changed'].tolist(), [[True, False, False], [False, False, False]])
        self.same(shuffled.memory_tensor()[0, 0], memory[1, 0])
        self.same(shuffled.memory_tensor()[0, 2], memory[0, 2])
        self.same(shuffled.memory_tensor()[1], memory[1])
        for name, pub in shuffled.publications.items():
            for field in ('version', 'occurrence', 'epoch', 'valid', 'round_index'):
                self.same(getattr(pub, field), getattr(state.publications[name], field))
        # Invalid donor cells cannot erase a valid old publication.
        names = list(state.publications)
        original = state.publications[names[0]]
        invalid_donor = replace(state, publications={**state.publications,
            names[0]: replace(original, valid=torch.tensor([True, False]))})
        blocked, info = aged_publication_shuffle(invalid_donor, [1, 0], age_threshold=3)
        self.assertFalse(bool(info['replaced'][0, 0]))
        self.same(blocked.memory_tensor()[0], invalid_donor.memory_tensor()[0])

    def test_fresh_read_trace_has_zero_age_and_no_committed_version(self):
        values = torch.tensor([list(b'A x?'), list(b'B x?')])
        outputs = query_forks(self.model, values, self.state, include_fresh=True)
        fresh = outputs['fresh']
        self.assertTrue(bool(fresh['result']['read_versions'].eq(-1).all()))
        for row in fresh['trace']:
            for part in row:
                self.assertEqual(part['read_source'], 'candidate')
                self.assertEqual(part['read_ages'], [0, 0, 0])
                self.assertEqual(part['weighted_publication_age'], 0.)
        self.assertEqual(fresh['result']['expert_executions'], outputs['intact']['result']['expert_executions'])

    def test_fresh_trained_audit_keeps_live_and_unmodified_forks_on_candidates(self):
        config = TaskStreamConfig(cycles=1, queries_per_cycle=2, delays=(0,),
                                  semantic_fraction=0, contextual_questions=False)
        with patch.object(self.model, 'forward', wraps=self.model.forward) as calls:
            summary, records = audit(self.model, config, count=2, data_seed=89,
                                     live_control='fresh', include_fresh=True)
        self.assertEqual(summary['live_control'], 'fresh')
        self.assertEqual(summary['live_read_source'], 'candidate')
        self.assertEqual(summary['live_fork_argmax_changes'], 0)
        self.assertEqual({call.kwargs['control'] for call in calls.call_args_list}, {'fresh', 'blocked'})
        for row in records:
            for branch in ('intact', 'hold', 'shuffle', 'hold_shuffle', 'fresh', 'aged_shuffle'):
                self.assertTrue(all(p['read_source'] == 'candidate' for p in row['branches'][branch]['trace']))
                self.assertEqual(row['branches'][branch]['prediction'], row['branches']['intact']['prediction'])
            self.assertAlmostEqual(row['branches']['hold']['nll'], row['branches']['intact']['nll'], places=6)
        self.assertEqual(summary['groups']['all']['intact']['prequery_publication_read_mass'], 0.)
        self.assertEqual(summary['aged_shuffle']['groups']['all']['intact_changed_read_mass'], 0.)

    def test_audit_reconstructs_live_predictions_and_uses_full_query_events(self):
        config = TaskStreamConfig(cycles=1, queries_per_cycle=2, delays=(0,),
                                  semantic_fraction=0, contextual_questions=False)
        self.model.train()
        summary, records = audit(self.model, config, count=2, data_seed=89, age_threshold=10_000)
        self.assertTrue(self.model.training)
        self.assertEqual(summary['query_count'], 4)
        self.assertEqual(summary['live_fork_argmax_changes'], 0)
        self.assertLess(summary['live_fork_max_logit_error'], 1e-4)
        aged = summary['aged_shuffle']['groups']['all']
        self.assertGreater(aged['existing_slot_count'], 0)
        self.assertEqual(aged['age_eligible_slot_count'], 0)
        self.assertEqual(aged['changed_query_count'], 0)
        self.assertEqual(aged['changed_fraction_of_existing'], 0.)
        self.assertEqual(aged['changed_queries']['intact'], dict(count=0, accuracy=None, nll=None))
        self.assertEqual(summary['groups']['all']['aged_shuffle']['nll'], summary['groups']['all']['intact']['nll'])
        for row in records:
            self.assertEqual(row['query_text'][-1], '?')
            self.assertEqual(row['query_stop']-row['query_start'], len(row['query_text']))
            self.assertEqual(row['position'], row['query_stop']-1)
            self.assertEqual(len(row['branches']['blocked']['trace']), len(row['query_text']))
            self.assertEqual(len(row['branches']['terminal_blocked']['trace']), 1)
            self.assertNotEqual(row['document'], row['branches']['hold_shuffle']['donor_document'])


if __name__ == '__main__':
    unittest.main()
