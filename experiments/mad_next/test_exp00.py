from collections import Counter, defaultdict
from dataclasses import replace
import itertools
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch

from .tasks import (Spec, TASKS, VALUE, dataset, exhaustive_oracles, fingerprint,
                    fold, parse, reencode, shortcuts, solve, World, COPY, SWAP)
from .controls import PositiveControl
from .calibration import decision, normalized
from .measurements import native_categories
from experiments.mad.backend import BenchModel

torch.set_num_threads(1)


def specs():
    return [Spec('binding'), Spec('one_hop', vocab=16, items=3, key_pool=16)] + [
        Spec(t, items=4, key_pool=4) for t in ('initialize', 'set', 'copy', 'swap')]


class FixtureTests(unittest.TestCase):
    def test_exhaustive_tiny_oracles_at_all_64_offsets(self):
        self.assertGreater(exhaustive_oracles(), 10000)

    def test_determinism_unique_world_splits_and_explicit_lineage(self):
        for spec in specs():
            sets = [dataset(spec, 64, part) for part in ('train', 'dev', 'test')]
            self.assertEqual(fingerprint(sets[0]), fingerprint(dataset(spec, 64, 'train')))
            for a, b in itertools.combinations(sets, 2):
                self.assertFalse({e.split_key for e in a} & {e.split_key for e in b})
            for name, cases in zip(('train', 'dev', 'test'), sets):
                self.assertEqual(len({e.digest for e in cases}), len(cases))
                for e in cases:
                    self.assertEqual(fold(e.split_key), name)
                    renamed = reencode(e, spec, rename={i: spec.key_pool-1-i for i in range(spec.key_pool)})
                    self.assertEqual(renamed.split_key, e.split_key)
                    self.assertEqual(solve(renamed, spec), renamed.answer)

    def test_cartesian_record_query_position_balance(self):
        for spec in specs():
            groups = defaultdict(list)
            for e in dataset(spec, 64, 'dev'):
                records, _ = parse(e)
                groups[e.split_key, tuple(records[spec.items:])].append(e)
            for group in groups.values():
                self.assertEqual(len(group), spec.items**2)
                combos = Counter()
                for ep in group:
                    records, queries = parse(ep)
                    for r, (_, k, _) in enumerate(records[:spec.items]):
                        combos[k, r, queries.index(k)] += 1
                self.assertEqual(set(combos.values()), {1})

    def test_scored_payload_and_stored_answers_never_enter_interpreter(self):
        for spec in specs():
            ep = dataset(spec, 1, 'dev')[0]
            changed = replace(ep, data=bytes(255 if m else b for b, m in zip(ep.data, ep.mask)),
                              answer=tuple(999 for _ in ep.answer))
            self.assertEqual(parse(changed), parse(ep))
            self.assertEqual(solve(changed, spec), ep.answer)

    def test_offsets_queries_and_operation_counterfactuals(self):
        for spec in specs():
            ep = dataset(spec, 1, 'dev')[0]
            for offset in range(64):
                varied = reencode(ep, spec, offset=offset)
                self.assertEqual(varied.answer, ep.answer)
                self.assertTrue(all(i % 64 == offset for i, m in enumerate(varied.mask) if m))
            swap = reencode(ep, spec, reverse_queries=True)
            self.assertEqual(swap.answer, ep.answer[::-1])
            if spec.task in ('set', 'copy', 'swap'):
                cf = reencode(ep, spec, change_operation=True)
                self.assertEqual(cf.split_key, ep.split_key)
                self.assertNotEqual(cf.answer, ep.answer)
                self.assertEqual(solve(cf, spec), cf.answer)
                records, _ = parse(cf)
                self.assertEqual(records[-1][0], parse(ep)[0][-1][0])

    def test_operations_on_the_same_initial_state_share_a_fold(self):
        records = ((0, 1), (1, 2), (2, 3), (3, 4))
        for task, op in (('copy', COPY), ('swap', SWAP)):
            a = World(task, records, (op, 0, 1))
            b = World(task, records[::-1], (op, 2, 3))
            self.assertEqual(a.group_key, b.group_key)
            self.assertEqual(fold(a.group_key), fold(b.group_key))
        self.assertEqual(World('swap', records, (SWAP, 0, 1)).content_key,
                         World('swap', records, (SWAP, 1, 0)).content_key)

    def test_shortcuts_do_not_solve_balanced_binding_or_atomic_updates(self):
        spec = Spec('binding')
        scores = shortcuts(dataset(spec, 128, 'dev'), spec)
        self.assertEqual(scores['input_order']['exact_accuracy'], .5)
        self.assertEqual(scores['input_order']['token_accuracy'], .5)
        for spec in specs():
            if spec.task in ('set', 'copy', 'swap'):
                scores = shortcuts(dataset(spec, 128, 'dev'), spec)
                self.assertEqual(scores['ignore_updates']['exact_accuracy'], 0.)
                self.assertLess(max(r['exact_accuracy'] for r in scores.values()), 1.)

    def test_invalid_specs_fail(self):
        for args in ({'task': 'bad'}, {'task': 'binding', 'items': 4, 'vocab': 2},
                     {'task': 'copy'}, {'task': 'binding', 'answer_offset': 64}):
            with self.assertRaises(ValueError):
                Spec(**args)


class ReferenceTests(unittest.TestCase):
    def test_reference_inputs_exclude_answers_and_future_queries(self):
        for spec in specs():
            model = PositiveControl(spec)
            ep = dataset(spec, 1, 'train')[0]
            corrupt = replace(ep, data=bytes(255 if m else b for b, m in zip(ep.data, ep.mask)))
            np.testing.assert_array_equal(model.predict([ep]), model.predict([corrupt]))
            for offset in (1, 17, 63):
                np.testing.assert_array_equal(model.predict([ep]), model.predict([reencode(ep, spec, offset=offset)]))
            # Reversing the observed queries only reverses independently decoded answers.
            np.testing.assert_array_equal(model.predict([ep])[:, ::-1], model.predict([reencode(ep, spec, reverse_queries=True)]))

    def test_training_changes_predictions_and_checkpoint_restores_optimizer(self):
        for spec in (Spec('binding'), Spec('set', items=4, key_pool=4)):
            a, b = PositiveControl(spec), PositiveControl(spec, seed=7)
            batch = dataset(spec, 32, 'train')
            before = a.predict(batch)
            a.update(batch)
            self.assertFalse(np.array_equal(before, a.predict(batch)))
            b.loads(a.dumps())
            self.assertEqual(a.update(batch), b.update(batch))
            np.testing.assert_array_equal(a.predict(batch), b.predict(batch))
            self.assertTrue(a.inventory()['not_a_matched_resource_comparison'])


class CalibrationTests(unittest.TestCase):
    def test_shortcut_ceiling_and_below_shortcut_never_promote(self):
        self.assertIsNone(normalized(.9, 1.))
        rows = [{'seed': s, 'dev': {'cases': 512, 'exact_accuracy': .3}} for s in (1, 7, 19)]
        positive = [{'seed': s, 'dev': {'cases': 512, 'exact_accuracy': 1.}} for s in (1, 7, 19)]
        self.assertEqual(decision(rows, positive, {'order': {'exact_accuracy': .5}})['status'], 'diagnostic_repair_required')
        self.assertFalse(decision(rows, positive, {'leak': {'exact_accuracy': 1.}})['eligible_for_architecture_comparison'])

    def test_only_dev_and_complete_seed_coverage_select(self):
        rows = [{'seed': s, 'dev': {'cases': 512, 'exact_accuracy': .75}, 'test': {'exact_accuracy': 0}} for s in (1, 7, 19)]
        positive = [{'seed': s, 'dev': {'cases': 512, 'exact_accuracy': 1.}} for s in (1, 7, 19)]
        self.assertEqual(decision(rows, positive, {'order': {'exact_accuracy': .5}})['status'], 'calibrated')
        with self.assertRaises(ValueError):
            decision(rows, positive[:-1], {'order': {'exact_accuracy': .5}})


class NativeMeasurementTests(unittest.TestCase):
    def test_category_masks_partition_the_same_teacher_forced_predictions(self):
        for task in ('set','copy','swap'):
            spec = Spec(task,items=4,key_pool=4)
            cases = dataset(spec,16,'dev')
            with BenchModel(spec.native(),500000,seed=7) as model:
                categories = native_categories(model,cases,spec)
                with model.clone() as frozen:
                    raw = np.sum([frozen.score(ep) for ep in cases],axis=0)
                changed,unchanged = (categories[k+'_targets'] for k in ('changed','unchanged'))
                self.assertEqual(changed+unchanged,int(raw[0]))
                self.assertAlmostEqual(categories['changed_accuracy']*changed+
                                       categories['unchanged_accuracy']*unchanged,raw[3])
                self.assertAlmostEqual(categories['native_changed_accuracy']*changed+
                                       categories['native_unchanged_accuracy']*unchanged,raw[6])

    def test_reference_budget_is_exact_and_partial_episode_has_no_labels(self):
        from .run import train_reference
        spec = Spec('binding')
        cases = dataset(spec,16,'train')
        config = {'reference_policy':{'attention_lr':.05,'gru_lr':.003,'batch_size':32},
                  'order_seed':1067}
        with tempfile.TemporaryDirectory() as d:
            row = train_reference(spec,1,cases,cases,config,Path(d)/'test.pt',129*3+7)
        self.assertEqual(row['bytes'],394)
        self.assertEqual(row['partial_episode_bytes_without_update'],7)
        self.assertEqual(row['supervised_targets'],6)


if __name__ == '__main__':
    unittest.main()
