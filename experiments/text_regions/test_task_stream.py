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

    def test_default_stream_fingerprints_and_legacy_config_stay_exact(self):
        expected = {
            'train': 'b29f29aab26f88d6ad3a5771da50464bf92b06e24f1e064088f581a0c2ecd937',
            'validation': '4bd50eefb5120713f8cd6e83462467e3811e9da260d287275980023fcdea2af2',
            'test': 'd0b4f949d6e48e34d0f7b0167def3265ee94a5659c02fa46e04dc63ee364ab4a',
            'long_delay': 'a3db4ee47c8effcbe13898b0294ed235fe468d78386cb3675dfa1ff35521be1a',
        }
        old = TaskStreamConfig().to_dict()
        old.pop('interference')
        restored = TaskStreamConfig.from_dict(old)
        self.assertFalse(restored.interference)
        for split, fingerprint in expected.items():
            docs = make_documents(restored, split=split, seed=123, count=16)
            self.assertEqual(TaskByteStreams(docs, 4).fingerprint, fingerprint)

    def test_interference_preserves_latest_fact_answers_and_provenance(self):
        # Multiple attributes and relationship questions exercise exclusion of
        # the actual queried owner, including an indirect relationship owner.
        cfg = TaskStreamConfig(entities='ABCD', attributes=('x', 'y'),
                               queries_per_cycle=8, delays=(2, 144), interference=True,
                               semantic_fraction=.3, relations=True)
        saw_extra_writes = saw_later_use = saw_semantic = saw_relation = False
        for doc in make_documents(cfg, split='test', seed=51, count=12):
            current, previous, sources, versions, serials, relations = {}, {}, {}, {}, {}, {}
            write_count = 0
            for event in doc.events:
                if event.kind in ('observation', 'correction'):
                    entity, binding = event.text.decode().removesuffix(';').split(' ')
                    attribute, value = binding.split('=')
                    pair = (entity, attribute)
                    previous[pair] = current.get(pair)
                    if event.kind == 'correction':
                        self.assertIn(pair, current)
                        self.assertNotEqual(current[pair], value)
                    current[pair] = value
                    versions[pair] = versions.get(pair, 0) + 1
                    sources[pair] = event.stop - 2
                    write_count += 1
                    serials[pair] = write_count
                elif event.kind == 'relation_observation':
                    entity, _, leader = event.text.decode().removesuffix(';').split(' ')
                    relations[entity] = leader
                elif event.kind == 'query':
                    q = doc.queries[event.query_index]
                    if q.kind == 'context':
                        continue
                    owner = relations[q.entity] if q.kind == 'relation' else q.entity
                    pair = (owner, q.attribute)
                    value = current[pair]
                    expected = ord(value)
                    old = previous[pair]
                    previous_answer = None if old is None else ord(old)
                    if q.kind == 'semantic':
                        threshold = len(cfg.values) // 2
                        expected = ord('H' if cfg.values.index(value) >= threshold else 'L')
                        previous_answer = None if old is None else ord('H' if cfg.values.index(old) >= threshold else 'L')
                    self.assertEqual(q.answer, expected)
                    self.assertEqual(q.previous_answer, previous_answer)
                    self.assertEqual(q.source_position, sources[pair])
                    self.assertEqual(q.source_version, versions[pair])
                    self.assertEqual(q.intervening_writes, write_count-serials[pair])
                    self.assertEqual(q.correction, versions[pair] > 1)
                    self.assertLess(q.source_position, event.start)
                    self.assertEqual(doc.data[q.source_position], ord(value))
                    self.assertEqual(doc.answer_targets[q.position], expected)
                    saw_later_use |= versions[pair] > 2
                    saw_semantic |= q.kind == 'semantic'
                    saw_relation |= q.kind == 'relation'
            saw_extra_writes |= write_count > len(cfg.entities)*len(cfg.attributes)+cfg.cycles-1
        self.assertTrue(saw_extra_writes and saw_later_use and saw_semantic and saw_relation)

    def test_interference_updates_other_real_entities_during_delay(self):
        cfg = TaskStreamConfig(entities='ABCD', queries_per_cycle=4, delays=(2, 48),
                               interference=True, semantic_fraction=0)
        doc = make_document(cfg, seed=2)
        # A long-query gap must contain real writes, not just punctuation.
        long_queries = [q for q in doc.queries if q.kind == 'exact' and q.delay > 48]
        self.assertTrue(long_queries)
        self.assertTrue(any(q.intervening_writes >= 2 for q in long_queries))
        for q in long_queries:
            updates = [e for e in doc.events if e.kind in ('observation', 'correction')
                       and e.start > q.source_position and e.stop <= q.position]
            self.assertEqual(len(updates), q.intervening_writes)
            for event in updates:
                self.assertFalse(event.text.startswith(f'{q.entity} {q.attribute}='.encode()))
                self.assertIn(chr(event.text[0]), cfg.entities)
        self.assertEqual(set(q.entity for q in doc.queries if q.kind == 'exact'), set(cfg.entities))

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
