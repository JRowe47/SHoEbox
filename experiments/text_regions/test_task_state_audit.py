"""Content-only producer interventions and correct state commit ages."""
from dataclasses import replace
import unittest
import torch

from .ports import tensor_list
from .task_memory import TaskMemoryModel
from .task_query_audit import aged_producer_shuffle, query_forks, audit
from .task_stream import TaskStreamConfig
from .test_task_retention_integration import model_config

torch.set_num_threads(1)


class StateAuditChecks(unittest.TestCase):
    def same(self, a, b):
        left, right = tensor_list(a), tensor_list(b)
        self.assertEqual(len(left), len(right))
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    @torch.no_grad()
    def test_shuffle_preserves_other_state_and_uses_actual_commit_age(self):
        torch.manual_seed(21)
        model = TaskMemoryModel(model_config(state_commit='publication'))
        state = model(torch.tensor([[65, 66, 67], [72, 73, 74]]), write_override='always')['state']
        state = replace(state, events_seen=torch.tensor([100, 12]))
        shuffled, info = aged_producer_shuffle(state, [1, 0], 64, 'publication')
        self.assertEqual(info['ages'].tolist(), [[98] * 3, [10] * 3])
        self.assertEqual(info['content_changed'].tolist(), [[True] * 3, [False] * 3])
        self.same(state.publications, shuffled.publications)
        self.same(state.ingress, shuffled.ingress)
        self.same(state.composer, shuffled.composer)
        self.same(state.producer.clock, shuffled.producer.clock)
        self.same(state.events_seen, shuffled.events_seen)
        for old, new in zip((*state.producer.contextualizer, *state.producer.blocks),
                            (*shuffled.producer.contextualizer, *shuffled.producer.blocks)):
            self.same(new[:3], old[3:])
            self.same(new[3:], old[3:])
        unchanged, recent = aged_producer_shuffle(state, [1, 0], 64, 'always')
        self.same(state, unchanged)
        self.assertEqual(recent['ages'].tolist(), [[1] * 3] * 2)
        self.assertFalse(bool(recent['age_eligible'].any()))
        empty = model.initial_state(2)
        empty_shuffle, empty_info = aged_producer_shuffle(empty, [1, 0], 0, 'publication')
        self.same(empty, empty_shuffle)
        self.assertFalse(bool(empty_info['existing'].any()))

    @torch.no_grad()
    def test_forks_keep_compute_and_recipient_metadata(self):
        torch.manual_seed(22)
        model = TaskMemoryModel(model_config(state_commit='publication'))
        state = model(torch.tensor([list(b'A x=1;'), list(b'A x=3;')]), write_override='always')['state']
        forks = query_forks(model, torch.tensor([list(b'A x?')] * 2), state,
                            donors=[1, 0], age_threshold=0, include_producer=True)
        expected = forks['intact']['result']['expert_executions']
        self.assertTrue(all(item['result']['expert_executions'] == expected for item in forks.values()))
        self.assertIn('aged_producer_shuffle', forks)
        for key in ('contextualizer', 'blocks'):
            self.same(getattr(state.producer, key), getattr(forks['hold']['result']['state'].producer, key))
        self.assertTrue(bool(forks['producer_shuffle']['age_filter']['content_changed'].all()))

    def test_audit_preserves_live_predictions_with_held_and_fresh_reads(self):
        torch.manual_seed(23)
        model = TaskMemoryModel(model_config(state_commit='publication'))
        config = TaskStreamConfig(entities='AB', queries_per_cycle=2, delays=(0, 2), cycles=1,
                                  interference=True)
        for control in ('intact', 'fresh'):
            summary, records = audit(model, config, data_seed=91, count=4,
                age_threshold=0, include_producer=True, live_control=control)
            self.assertEqual(summary['live_fork_argmax_changes'], 0)
            self.assertLess(summary['live_fork_max_logit_error'], 1e-4)
            self.assertEqual(summary['state_commit'], 'publication')
            self.assertEqual(summary['aged_producer_shuffle']['age_source'], 'publication commit')
            info = summary['producer_shuffle']['groups']['all']
            self.assertGreater(info['changed_query_count'], 0)
            self.assertEqual(info['changed_query_count'], sum(any(r['branches']['producer_shuffle']['age_filter']['content_changed']) for r in records))


if __name__ == '__main__':
    unittest.main()
