"""Whole-source scheduling coverage, legacy identity and deterministic replay."""
from collections import Counter, defaultdict
import hashlib
import json
import random
import unittest
import torch
from .view_data import (FACTOR_SOURCES, IGNORE, PRIVATE_FACTORS, QUERY_TYPES, VALUES, VIEWS,
                        ViewStream, make_view_batch, split_combinations)


def canonical(value):
    if torch.is_tensor(value):
        return {'dtype': str(value.dtype), 'shape': list(value.shape), 'values': value.tolist()}
    if isinstance(value, dict): return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [canonical(item) for item in value]
    return value


class SourceScheduleDataChecks(unittest.TestCase):
    def test_balanced_horizon_covers_every_supported_subset_answer_and_train_tuple(self):
        expected = {'shared': {'A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC'},
                    'left': {'A', 'AB', 'AC', 'ABC'}, 'right': {'C', 'AC', 'BC', 'ABC'}}
        for query in QUERY_TYPES:
            batch = make_view_batch('train', 0, 1280, seed=29, gap=0,
                                    query_type=query, source_schedule='balanced')
            by_subset, by_tuple = defaultdict(Counter), defaultdict(Counter)
            for row in range(1280):
                subset = ''.join(view for index, view in enumerate(VIEWS) if batch.retained_views[row, index])
                factors = tuple(int(batch.factors[factor][row]) for factor in QUERY_TYPES)
                by_subset[subset][int(batch.answers[row])] += 1
                by_tuple[factors][subset] += 1
            self.assertEqual(set(by_subset), expected[query])
            self.assertEqual(set(by_tuple), set(split_combinations('train')))
            for answers in by_subset.values():
                self.assertEqual(set(answers), set(VALUES))
                self.assertEqual(len(set(answers.values())), 1)
            for counts in by_tuple.values():
                self.assertEqual(set(counts), expected[query])
                self.assertEqual(set(counts.values()), {5, 6} if query == 'shared' else {10})
            self.assertEqual(Counter(sum(row.values()) for row in by_subset.values()),
                             Counter({184: 5, 180: 2}) if query == 'shared' else Counter({320: 4}))
            self.assertTrue(bool(batch.answer_available[batch.targets != IGNORE].all()))

    def test_balanced_changes_only_whole_source_availability_and_causal_metadata(self):
        base = make_view_batch('train', 17, 96, seed=7, gap=8)
        random.seed(73); torch.manual_seed(73)
        python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
        balanced = make_view_batch('train', 17, 96, seed=7, gap=8, source_schedule='balanced')
        self.assertEqual(python_rng, random.getstate())
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        for key in ('inputs', 'targets', 'reset', 'final', 'answers', 'factors', 'query_types',
                    'identities', 'fact_positions', 'template_ids', 'description_end', 'time_offset'):
            self.assertEqual(canonical(getattr(base, key)), canonical(getattr(balanced, key)), key)
        for index, view in enumerate(VIEWS):
            kept = balanced.retained_views[:, index]
            self.assertTrue(torch.equal(balanced.availability[view], base.availability[view] & kept[:, None]))
            self.assertFalse(bool(balanced.availability[view][~kept].any()))
            self.assertTrue(torch.equal(balanced.shared_targets[view][kept], base.shared_targets[view][kept]))
            self.assertTrue(bool(balanced.shared_targets[view][~kept].eq(IGNORE).all()))
            if view in PRIVATE_FACTORS:
                self.assertTrue(torch.equal(balanced.private_targets[view][kept], base.private_targets[view][kept]))
                self.assertTrue(bool(balanced.private_targets[view][~kept].eq(IGNORE).all()))
            for factor in QUERY_TYPES:
                self.assertTrue(torch.equal(balanced.observation_positions[view][factor][kept],
                                            base.observation_positions[view][factor][kept]))
                self.assertTrue(bool(balanced.observation_positions[view][factor][~kept].eq(-1).all()))
            for row, query in enumerate(balanced.query_types):
                expected = bool(kept[row]) and view in FACTOR_SOURCES[query]
                self.assertEqual(bool(balanced.baseline_observed_mask(view)[row, -1]), expected)

    def test_schedule_uses_query_local_ordinal_not_batch_boundaries_or_generation_mode(self):
        mixed = make_view_batch('train', 0, 192, seed=13, gap=0, source_schedule='balanced')
        for query_index, query in enumerate(QUERY_TYPES):
            fixed = make_view_batch('train', 0, 64, seed=13, gap=0,
                                    query_type=query, source_schedule='balanced')
            self.assertEqual(mixed.identities[query_index::3], fixed.identities)
            self.assertTrue(torch.equal(mixed.retained_views[query_index::3], fixed.retained_views))
        parts = [make_view_batch('train', start, size, seed=13, gap=0, source_schedule='balanced')
                 for start, size in ((0, 17), (17, 54), (71, 121))]
        self.assertTrue(torch.equal(mixed.retained_views, torch.cat([part.retained_views for part in parts])))
        self.assertEqual(mixed.identities, sum((part.identities for part in parts), ()))
        other = make_view_batch('train', 0, 192, seed=14, gap=0, source_schedule='balanced')
        self.assertFalse(torch.equal(mixed.retained_views, other.retained_views))

    def test_balanced_stream_replays_chunks_and_legacy_missing_schedule_stays_dropout(self):
        for schedule, dropout in (('balanced', 0.), ('dropout', .7)):
            original = ViewStream(batch_size=5, gap=8, seed=37, dropout=dropout, source_schedule=schedule)
            original.next_chunk(16)
            saved = json.loads(json.dumps(original.state_dict()))
            self.assertEqual(saved['spec']['source_schedule'], schedule)
            if schedule == 'dropout': saved['spec'].pop('source_schedule')
            unmodified = json.loads(json.dumps(saved))
            restored = ViewStream(batch_size=5, gap=8, seed=37, dropout=dropout, source_schedule=schedule)
            restored.load_state_dict(saved)
            self.assertEqual(saved, unmodified)
            for length in (9, 7, 9, 16, 16, 9, 16):
                self.assertEqual(canonical(vars(original.next_chunk(length))),
                                 canonical(vars(restored.next_chunk(length))))
            self.assertEqual(original.state_dict(), restored.state_dict())
        legacy = ViewStream(batch_size=5, gap=8, seed=37).state_dict()
        legacy['spec'].pop('source_schedule')
        with self.assertRaisesRegex(ValueError, 'protocol mismatch'):
            ViewStream(batch_size=5, gap=8, seed=37, source_schedule='balanced').load_state_dict(legacy)

    def test_default_and_explicit_dropout_match_pre_change_full_batch_fingerprints(self):
        expected = {0.: '395e4de09e3d0ede1f8fc4269f74c22b459f25b9a78af87c722517a712b5d11e',
                    .25: 'ae9a2524814b28dbb82557155952040a9f2a6a01c71753d27d0231965f8135fd'}
        for dropout, fingerprint in expected.items():
            for kwargs in ({}, {'source_schedule': 'dropout'}):
                batch = make_view_batch('train', 37, 48, seed=20261106, gap=8, dropout=dropout, **kwargs)
                encoded = json.dumps(canonical(vars(batch)), sort_keys=True, separators=(',', ':')).encode()
                self.assertEqual(hashlib.sha256(encoded).hexdigest(), fingerprint)

    def test_balanced_rejects_dropout_evaluation_controls_and_unknown_schedule(self):
        for kwargs in ({'dropout': .25}, {'control': 'a_only'}, {'control': 'no_private'}):
            with self.assertRaisesRegex(ValueError, 'train split, all control, and dropout=0'):
                make_view_batch('train', 0, 2, source_schedule='balanced', **kwargs)
        for split in ('validation', 'test', 'independent'):
            with self.assertRaises(ValueError): make_view_batch(split, 0, 2, source_schedule='balanced')
        with self.assertRaises(ValueError): ViewStream(source_schedule='balanced', dropout=.25)
        with self.assertRaises(ValueError): make_view_batch('train', 0, 2, source_schedule='unknown')


if __name__ == '__main__': unittest.main()
