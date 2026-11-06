"""Causality, correction provenance and independent stream boundaries."""
from dataclasses import replace
import unittest
import torch
from .task_stream import TaskStreamConfig, TaskByteStreams, make_document, make_documents


class TaskStreamChecks(unittest.TestCase):
    def test_answers_are_not_inputs_and_corrections_have_provenance(self):
        doc = make_document(TaskStreamConfig(semantic_fraction=0), seed=3)
        self.assertTrue(any(q.correction for q in doc.queries))
        for event in doc.events:
            if event.kind == 'query':
                q = doc.queries[event.query_index]
                self.assertEqual(event.text[-1], ord('?'))
                self.assertEqual(event.stop - 1, q.position)
                self.assertEqual(doc.answer_targets[q.position], q.answer)
                if q.kind == 'exact':
                    self.assertNotIn(q.answer, event.text)
                    self.assertEqual(doc.data[q.source_position], q.answer)
                    self.assertGreater(q.delay, 0)
                    if q.correction:
                        self.assertNotEqual(q.answer, q.previous_answer)
        self.assertTrue(all(v == -100 for e in doc.events if e.kind != 'query'
                            for v in doc.answer_targets[e.start:e.stop]))

    def test_delayed_queries_cross_chunk_horizon(self):
        cfg = TaskStreamConfig(delays=(2, 40), semantic_fraction=0)
        short = make_document(cfg, split='test', seed=4)
        long = make_document(cfg, split='long_delay', seed=4)
        self.assertTrue(any(q.delay > 16 for q in short.queries if q.kind == 'exact'))
        self.assertTrue(any(q.delay >= 120 for q in long.queries if q.kind == 'exact'))
        self.assertTrue(any(q.kind == 'context' for q in short.queries))

    def test_split_reproducibility_and_query_composition_holdout(self):
        cfg = TaskStreamConfig(attributes=('x', 'y'), holdout_combinations=True,
                               semantic_fraction=0, cycles=2, queries_per_cycle=4)
        train = make_documents(cfg, split='train', count=4, seed=9)
        test = make_documents(cfg, split='combinations', count=4, seed=9)
        self.assertEqual(train, make_documents(cfg, split='train', count=4, seed=9))
        known = {(q.entity, q.attribute) for d in train for q in d.queries if q.kind == 'exact'}
        held = {(q.entity, q.attribute) for d in test for q in d.queries if q.kind == 'exact'}
        self.assertFalse(known & held)
        self.assertNotEqual(train[0].data, make_document(cfg, split='validation', seed=9).data)

    def test_resume_and_resets_only_on_new_document(self):
        docs = make_documents(count=4, seed=7)
        stream = TaskByteStreams(docs, batch_size=2)
        first = stream.next_chunk(17)
        self.assertEqual(int(first.reset.sum()), 2)
        state = stream.state_dict()
        a = stream.next_chunk(200)
        other = TaskByteStreams(docs, batch_size=2)
        other.load_state_dict(state)
        b = other.next_chunk(200)
        for key in ('values', 'ordinary_targets', 'answer_targets', 'reset', 'document_ids', 'positions'):
            torch.testing.assert_close(getattr(a, key), getattr(b, key), rtol=0, atol=0)
        torch.testing.assert_close(a.reset, a.positions.eq(0))
        for row in range(2):
            for t in range(200):
                d, p = int(a.document_ids[row, t]), int(a.positions[row, t])
                self.assertEqual(int(a.answer_targets[row, t]), docs[d].answer_targets[p])
                if p == len(docs[d].data) - 1:
                    self.assertEqual(int(a.ordinary_targets[row, t]), -100)

    def test_resume_rejects_changed_labels_or_data(self):
        docs = make_documents(count=2)
        cursor = TaskByteStreams(docs, 2).state_dict()
        changed = replace(docs[0], answer_targets=tuple(-100 for _ in docs[0].data))
        with self.assertRaises(ValueError):
            TaskByteStreams((changed, docs[1]), 2).load_state_dict(cursor)


if __name__ == '__main__':
    unittest.main()
