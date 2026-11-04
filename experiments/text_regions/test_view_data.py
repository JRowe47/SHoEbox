"""Factor balance, causal source labels, control integrity and cursor replay."""
from collections import Counter, defaultdict
import json
import random
import unittest
import torch
from .config import GraphConfig
from .model import RegionLanguageModel
from .view_data import (CONTROLS, DESCRIPTION_END, FACTOR_SOURCES, IGNORE, MARKERS,
                        PRIVATE_FACTORS, QUERY_TYPES, SPLITS, VALUES, VIEWS,
                        ViewStream, make_view_batch, make_view_episode, split_combinations)

torch.set_num_threads(2)


def equal_tree(test, left, right):
    if torch.is_tensor(left): torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left: equal_tree(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right): equal_tree(test, a, b)
    else: test.assertEqual(left, right)


class ViewDataChecks(unittest.TestCase):
    def test_disjoint_factor_holdouts_cover_all_marginals(self):
        sets = {split: set(split_combinations(split)) for split in SPLITS}
        self.assertEqual([len(sets[split]) for split in SPLITS], [32, 16, 16, 64])
        for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
            self.assertFalse(sets[a] & sets[b])
        self.assertEqual(sets['train'] | sets['validation'] | sets['test'], sets['independent'])
        for combinations in sets.values():
            for axis in range(3):
                self.assertEqual(Counter(values[axis] for values in combinations),
                                 Counter({value: len(combinations)//4 for value in VALUES}))

    def test_independent_cohort_has_uniform_missing_private_conditional(self):
        for hidden in (1, 2):
            conditions = defaultdict(Counter)
            for factors in split_combinations('independent'):
                conditions[tuple(value for axis, value in enumerate(factors) if axis != hidden)][factors[hidden]] += 1
            self.assertEqual(len(conditions), 16)
            for counts in conditions.values(): self.assertEqual(counts, Counter({value: 1 for value in VALUES}))

    def test_fixed_query_balances_512_answers_and_complete_factor_cycles(self):
        for query in QUERY_TYPES:
            batch = make_view_batch('independent', 0, 512, seed=29, gap=0, query_type=query)
            self.assertEqual(set(batch.query_types), {query})
            self.assertEqual(Counter(batch.answers.tolist()), Counter({value: 128 for value in VALUES}))
            combinations = Counter(zip(*(batch.factors[factor].tolist() for factor in QUERY_TYPES)))
            self.assertEqual(combinations, Counter({values: 8 for values in split_combinations('independent')}))
        mixed = make_view_batch('train', 0, 96, seed=29, gap=0)
        self.assertEqual(Counter(zip(mixed.query_types, mixed.answers.tolist())),
                         Counter({(query, value): 8 for query in QUERY_TYPES for value in VALUES}))

    def test_determinism_rng_isolation_and_disjoint_opaque_identifiers(self):
        random.seed(73); torch.manual_seed(73)
        python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
        identifiers = set()
        for split in SPLITS:
            for index in range(96):
                episode = make_view_episode(split, index, seed=19)
                self.assertEqual(episode, make_view_episode(split, index, seed=19))
                self.assertNotIn(episode.identity, identifiers); identifiers.add(episode.identity)
                self.assertEqual(len(episode.identity), 6)
                self.assertTrue(all(char in '0123456789abcdef' for char in episode.identity))
        # Index-mod-four is deliberately balanced, so raw sequential hex IDs
        # would leak the answer. Opaque IDs do not expose that digit mapping.
        episodes = [make_view_episode('train', i, seed=19, query_type='left') for i in range(64)]
        by_digit = defaultdict(set)
        for episode in episodes: by_digit[episode.identity[-1]].add(episode.answer)
        self.assertTrue(any(len(answers) > 2 for answers in by_digit.values()))
        self.assertNotEqual(episodes[0].identity, make_view_episode('train', 0, seed=20, query_type='left').identity)
        self.assertEqual(python_rng, random.getstate())
        torch.testing.assert_close(torch_rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_source_templates_observations_padding_and_no_answer_feedback(self):
        template_pairs, shared_positions = set(), set()
        for index in range(96):
            episode = make_view_episode('train', index, seed=13, gap=32)
            template_pairs.add((episode.template_ids['A'], episode.template_ids['C']))
            shared_positions.add((episode.fact_positions['A']['shared'], episode.fact_positions['C']['shared']))
            self.assertEqual(len(episode.targets), 49)
            self.assertEqual(episode.query_positions[:2], (14, 15))
            for view in VIEWS:
                text, available = episode.views[view], episode.availability[view]
                physical = [position for position in episode.fact_positions[view].values() if position >= 0]
                self.assertEqual([i for i, byte in enumerate(text) if byte in VALUES], sorted(physical))
                for factor, position in episode.fact_positions[view].items():
                    if view in FACTOR_SOURCES[factor]:
                        self.assertGreaterEqual(position, 7); self.assertLess(position, DESCRIPTION_END)
                        self.assertEqual(text[position], episode.factors[factor]); self.assertTrue(available[position])
                    else: self.assertEqual(position, -1)
                for t, active in enumerate(available):
                    if not active:
                        self.assertEqual(text[t], 0); self.assertLess(t, DESCRIPTION_END)
                for t, target in enumerate(episode.targets):
                    self.assertEqual(target != IGNORE, t in episode.query_positions)
                    if target != IGNORE:
                        self.assertEqual(text[t], MARKERS[episode.query_type]); self.assertTrue(available[t])
                        self.assertEqual(target, episode.answer)
            # Every answer's write and the first two query losses share a
            # 16-event autograd segment, while the final query crosses it.
            self.assertGreater(episode.final_query-max(episode.fact_positions[v][episode.query_type]
                               for v in FACTOR_SOURCES[episode.query_type]), 16)
        self.assertEqual(len(template_pairs), 9)
        self.assertTrue(any(a != c for a, c in shared_positions))

    def test_controls_preserve_tokens_labels_and_exact_private_mask_positions(self):
        base = make_view_batch('independent', 0, 12, seed=3, gap=8)
        for control in CONTROLS:
            batch = make_view_batch('independent', 0, 12, seed=3, gap=8, control=control)
            equal_tree(self, base.inputs, batch.inputs)
            equal_tree(self, base.targets, batch.targets)
            equal_tree(self, base.factors, batch.factors)
            self.assertEqual(base.identities, batch.identities)
            self.assertEqual(batch.length, 25)
            if control == 'no_private':
                for view in VIEWS:
                    changed = (base.availability[view] != batch.availability[view]).nonzero().tolist()
                    expected = [[row, int(base.fact_positions[view][PRIVATE_FACTORS[view]][row])]
                                for row in range(12)] if view in PRIVATE_FACTORS else []
                    self.assertEqual(changed, expected)
            if control == 'reset_query':
                self.assertEqual(int(batch.reset.sum()), 24)
                equal_tree(self, base.availability, batch.availability)
            else: self.assertEqual(int(batch.reset.sum()), 12)

    def test_private_targets_are_causal_and_respect_dropout_withholding_and_reset(self):
        for control in CONTROLS:
            batch = make_view_batch('independent', 0, 12, seed=11, gap=8, control=control)
            for view, factor in PRIVATE_FACTORS.items():
                equal_tree(self, batch.private_masks[view], batch.private_targets[view] != IGNORE)
                for row in range(12):
                    position = int(batch.observation_positions[view][factor][row])
                    labels = batch.private_targets[view][row]
                    if position < 0: self.assertTrue(bool(labels.eq(IGNORE).all()))
                    else:
                        self.assertTrue(bool(labels[:position].eq(IGNORE).all()))
                        self.assertEqual(int(labels[position]), int(batch.factors[factor][row]))
                        self.assertTrue(bool(labels[~batch.availability[view][row]].eq(IGNORE).all()))
                    if control == 'reset_query': self.assertEqual(int(labels[-1]), IGNORE)
            for row, query in enumerate(batch.query_types):
                supported = control == 'all' or control == 'no_private' and query == 'shared'
                if control.endswith('_only'):
                    supported = control[0].upper() in FACTOR_SOURCES[query]
                self.assertEqual(bool(batch.answer_available[row, -1]), supported)

    def test_training_dropout_keeps_a_source_with_the_requested_fact(self):
        batch = make_view_batch('train', 0, 192, seed=41, gap=4, dropout=1.)
        self.assertTrue(bool(batch.retained_views.any(1).all()))
        self.assertTrue(bool((batch.retained_views.sum(1) < 3).all()))
        self.assertTrue(bool(batch.answer_available[batch.targets != IGNORE].all()))
        observed_subsets = defaultdict(set)
        for row, query in enumerate(batch.query_types):
            retained = tuple(view for i, view in enumerate(VIEWS) if bool(batch.retained_views[row, i]))
            self.assertTrue(set(retained) & set(FACTOR_SOURCES[query]))
            observed_subsets[query].add(retained)
            for view in PRIVATE_FACTORS:
                if view not in retained: self.assertTrue(bool(batch.private_targets[view][row].eq(IGNORE).all()))
        self.assertEqual(observed_subsets['shared'], {('A',), ('B',), ('C',), ('A', 'B'), ('A', 'C'), ('B', 'C')})
        equal_tree(self, vars(batch), vars(make_view_batch('train', 0, 192, seed=41, gap=4, dropout=1.)))
        with self.assertRaises(ValueError): make_view_batch('test', 0, 2, dropout=.5)
        with self.assertRaises(ValueError): make_view_batch('train', 0, 2, dropout=.5, control='no_private')

    def test_baseline_observed_mask_respects_source_query_fact_time_and_controls(self):
        for control in CONTROLS:
            batch = make_view_batch('independent', 0, 12, seed=11, gap=8, control=control)
            for view in VIEWS:
                observed = batch.baseline_observed_mask(view)
                self.assertEqual(observed.dtype, torch.bool)
                self.assertEqual(observed.shape, batch.targets.shape)
                for row, query in enumerate(batch.query_types):
                    position = int(batch.observation_positions[view][query][row])
                    expected = torch.zeros(batch.length, dtype=torch.bool)
                    if position >= 0:
                        expected[position:] = batch.availability[view][row, position:]
                    if control == 'reset_query': expected[-1] = False
                    torch.testing.assert_close(observed[row], expected, rtol=0, atol=0)
                    if position >= 0:
                        self.assertTrue(bool(observed[row, position]))
                        self.assertFalse(bool(observed[row, :position].any()))
            # Computing loss metadata leaves all generated bytes and labels intact.
            equal_tree(self, vars(batch), vars(make_view_batch('independent', 0, 12, seed=11,
                                                              gap=8, control=control)))
        with self.assertRaises(ValueError): batch.baseline_observed_mask('text')

    def test_baseline_observed_mask_does_not_infer_unsupported_facts_from_equal_values(self):
        batch = make_view_batch('independent', 0, 192, seed=23, gap=0)
        equal_values = batch.factors['shared'].eq(batch.factors['left']) & batch.factors['left'].eq(batch.factors['right'])
        for view in VIEWS:
            for query in QUERY_TYPES:
                if view in FACTOR_SOURCES[query]: continue
                rows = equal_values & torch.tensor([kind == query for kind in batch.query_types])
                self.assertTrue(bool(rows.any()), (view, query))
                self.assertTrue(bool(batch.shared_targets[view][rows, -1].eq(batch.answers[rows]).all()))
                self.assertFalse(bool(batch.baseline_observed_mask(view)[rows].any()), (view, query))

    def test_baseline_observed_mask_slices_preserve_past_knowledge_and_reset(self):
        for control in ('all', 'no_private', 'reset_query'):
            batch = make_view_batch('train', 0, 12, seed=7, gap=8, control=control)
            for view in VIEWS:
                observed = batch.baseline_observed_mask(view)
                for start, stop in ((0, 7), (7, 14), (14, 16), (16, batch.length), (batch.length-1, batch.length)):
                    chunk = batch.slice(start, stop)
                    torch.testing.assert_close(chunk.baseline_observed_mask(view), observed[:, start:stop], rtol=0, atol=0)
                self.assertFalse(bool(batch.slice(0, 7).baseline_observed_mask(view).any()))
                nested = batch.slice(7, batch.length).slice(9, batch.length-7)
                torch.testing.assert_close(nested.baseline_observed_mask(view), observed[:, 16:], rtol=0, atol=0)
                if control == 'all':
                    supported = torch.tensor([view in FACTOR_SOURCES[query] for query in batch.query_types])
                    self.assertTrue(bool(batch.slice(16, batch.length).baseline_observed_mask(view)[supported].all()))
                if control == 'reset_query':
                    self.assertFalse(bool(batch.slice(batch.length-1, batch.length).baseline_observed_mask(view).any()))

    def test_pooled_private_targets_do_not_label_prefix_or_reset_segment(self):
        batch = make_view_batch('train', 0, 12, seed=7, gap=8)
        for values in batch.slice(0, 7).pooled_private_targets().values(): self.assertTrue(bool(values.eq(IGNORE).all()))
        for start, stop in ((0, 16), (16, batch.length)):
            chunk = batch.slice(start, stop)
            self.assertEqual(chunk.time_offset, start)
            for view, factor in PRIVATE_FACTORS.items(): equal_tree(self, chunk.pooled_private_targets()[view], batch.factors[factor])
        reset = make_view_batch('train', 0, 12, seed=7, gap=8, control='reset_query')
        for values in reset.slice(reset.length-1, reset.length).pooled_private_targets().values():
            self.assertTrue(bool(values.eq(IGNORE).all()))

    def test_stream_cursor_json_roundtrip_preserves_chunks_and_resets(self):
        original = ViewStream(batch_size=5, gap=8, seed=37, dropout=.7)
        first = original.next_chunk(16)
        self.assertTrue(bool(first.reset[:, 0].all()))
        restored = ViewStream(batch_size=5, gap=8, seed=37, dropout=.7)
        restored.load_state_dict(json.loads(json.dumps(original.state_dict())))
        for _ in range(7):
            a, b = original.next_chunk(16), restored.next_chunk(16)
            equal_tree(self, vars(a), vars(b))
            self.assertEqual(int(a.reset.sum()), 5 if a.time_offset == 0 else 0)
        with self.assertRaises(ValueError): ViewStream(batch_size=5, gap=8, seed=38, dropout=.7).load_state_dict(original.state_dict())


class ViewModelBudgetChecks(unittest.TestCase):
    @torch.no_grad()
    def test_every_information_control_executes_identical_expert_budget(self):
        torch.manual_seed(5)
        model = RegionLanguageModel(GraphConfig.tiny()).eval()
        calls = []
        for control in CONTROLS:
            batch = make_view_batch('independent', 0, 2, seed=5, gap=0, control=control)
            result = model(batch.inputs, reset=batch.reset, availability=batch.availability)
            calls.append(result['expert_executions'])
            if control == 'reset_query': self.assertEqual(result['state'].events_seen.tolist(), [1, 1])
        self.assertEqual(len(set(calls)), 1)
        self.assertEqual(calls[0], 17*2*model.parameter_summary()['expert_calls_per_round_per_external_event'])


if __name__ == '__main__': unittest.main()
