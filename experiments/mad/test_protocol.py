import copy
from dataclasses import replace
import argparse
import json
import unittest

import numpy as np

from .backend import BenchModel, CompressionHead
from .metrics import at_budget, calibrate, choose_hyperparameters, paired_interval
from .tasks import TASKS, TaskSpec, VALUE, dataset, fingerprint, fixed_table, solve


class TaskTests(unittest.TestCase):
    def test_all_tasks_reproduce_and_separate_folds(self):
        for task in TASKS:
            with self.subTest(task=task):
                spec = TaskSpec(task)
                sets = [dataset(spec, 16, split) for split in ("train", "dev", "test")]
                self.assertEqual(fingerprint(sets[0]), fingerprint(dataset(spec, 16, "train")))
                for i in range(3):
                    for j in range(i):
                        self.assertFalse({e.split_key for e in sets[i]} & {e.split_key for e in sets[j]})
                for cases in sets:
                    for ep in cases:
                        self.assertEqual(solve(ep, spec, fixed_table(spec, 47)), ep.answer)

    def test_oracle_never_reads_stored_answer(self):
        for task in TASKS:
            if task == "compression":
                continue
            spec = TaskSpec(task)
            ep = dataset(spec, 1, "train")[0]
            raw = bytes(VALUE+(x-VALUE+1) % spec.vocab if mask else x for x, mask in zip(ep.data, ep.mask))
            changed = replace(ep, data=raw, answer=tuple((x+1) % spec.vocab for x in ep.answer))
            self.assertEqual(solve(changed, spec, fixed_table(spec, 47)), ep.answer)

    def test_memorization_coverage_and_fixed_mapping(self):
        spec = TaskSpec("memorization", vocab=16)
        table = fixed_table(spec, 47)
        for split in ("train", "dev", "test"):
            cases = dataset(spec, 16, split)
            self.assertEqual(sorted(e.answer[0] for e in cases), list(range(16)))
            self.assertTrue(all(solve(e, spec, table) == e.answer for e in cases))

    def test_noisy_and_multibyte_settings(self):
        for spec in (TaskSpec("noisy_recall", noise=.8, delay=128),
                     TaskSpec("fuzzy_recall", key_width=4, value_width=4),
                     TaskSpec("pointer_chasing", length=8, depth=6)):
            for ep in dataset(spec, 16, "test"):
                self.assertEqual(solve(ep, spec, fixed_table(spec, 47)), ep.answer)

    def test_answer_queries_complete_upper_constituents(self):
        for task in TASKS:
            for ep in dataset(TaskSpec(task), 8, "train"):
                # First byte of each answer follows a completed 64-byte input.
                for j in range(1, len(ep.mask)):
                    if ep.mask[j] and not ep.mask[j-1]:
                        self.assertEqual(j % 64, 0)

    def test_invalid_spaces_fail(self):
        for kwargs in (dict(task="recall", vocab=2, length=4), dict(task="composition", vocab=7),
                       dict(task="pointer_chasing", depth=4), dict(task="compression", noise=float("nan"))):
            with self.assertRaises(ValueError):
                TaskSpec(**kwargs)
        with self.assertRaises(ValueError):
            dataset(TaskSpec("compression", vocab=2, length=1), 8, "dev")


class MetricTests(unittest.TestCase):
    def test_complete_default_plan_and_cli_validation(self):
        from .run import make_plan
        args = argparse.Namespace(preset="screen", tasks=None, settings=None, sizes=None,
            seeds=None, steps=None, lr_scales=None, auxiliary=None, budget=None,
            eval_every=None, cases=None, train_examples=None, audit_probes=False)
        plan = make_plan(args)
        self.assertGreater(plan["fits"], 100)
        self.assertEqual(plan["total_training_bytes"], plan["fits"]*plan["budget"])
        self.assertEqual(plan["core"], "full")
        args.core = "minimal"
        self.assertEqual(make_plan(args)["steps"], [0])
        args.steps = [2]
        with self.assertRaises(ValueError):make_plan(args)
        args.core, args.steps = "full", None
        args.tasks, args.train_examples = ["memorization"], 2
        with self.assertRaises(ValueError):
            make_plan(args)
        args.train_examples, args.steps = 16, [0, 0]
        with self.assertRaises(ValueError):
            make_plan(args)

    def test_selection_uses_only_dev_and_matches_seeds(self):
        rows = []
        for lr, loss in ((.5, 2.), (1., 1.)):
            for seed in (1, 7):
                rows.append(dict(lr_scale=lr, auxiliary=.1, seed=seed,
                                 curve=[dict(dev=dict(task_nll=loss))], test=dict(task_nll=99-loss)))
        self.assertEqual(choose_hyperparameters(rows), (1., .1))
        with self.assertRaises(ValueError):
            choose_hyperparameters(rows[:-1])

    def test_measured_cost_points_and_unresolved_thresholds(self):
        curve = [dict(cpu_seconds=i, bytes=i*100) for i in (0, 1, 3)]
        self.assertEqual(at_budget(curve, "cpu_seconds", 2)["bytes"], 100)
        self.assertIsNone(paired_interval([.1])["ci95"])
        self.assertEqual(calibrate([dict(setting="easy", chance_normalized_accuracy=.01)])["status"], "uncalibrated")
        self.assertEqual(calibrate([dict(setting="mid", chance_normalized_accuracy=.4),
                                    dict(setting="easy", chance_normalized_accuracy=.99)])["setting"], "mid")


class BackendTests(unittest.TestCase):
    def test_compression_decoder_finite_differences(self):
        head = CompressionHead(4, 8, 7)
        x = np.random.default_rng(1).normal(0, .2, 64).astype(np.float32)
        y = [1, 4, 2, 7]
        _, dx, gradients = head.gradients(x, y)
        epsilon = .005
        for key, index in (("w1", (5, 8)), ("w2", (1, 4, 8)), ("b1", (5,)), ("b2", (2, 2))):
            value = getattr(head, key)
            old = value[index].copy()
            value[index] = old+epsilon
            plus = head.gradients(x, y)[0]
            value[index] = old-epsilon
            minus = head.gradients(x, y)[0]
            value[index] = old
            self.assertAlmostEqual((plus-minus)/(2*epsilon), float(gradients[key][index]), delta=5e-5)
        for j in (0, 7, 21):
            old = x[j].copy()
            x[j] = old+epsilon
            plus = head.gradients(x, y)[0]
            x[j] = old-epsilon
            minus = head.gradients(x, y)[0]
            x[j] = old
            self.assertAlmostEqual((plus-minus)/(2*epsilon), float(dx[j]), delta=5e-5)

    def test_native_predictions_are_causal_and_frozen(self):
        spec = TaskSpec("memorization")
        ep = dataset(spec, 1, "test")[0]
        with BenchModel(spec, 500000) as m:
            before = m.model.dumps()
            with m.clone() as frozen:
                row = frozen.score(ep)
                first = ep.mask.index(1)
                frozen.model.reset_context(clear_memory=True)
                for byte in ep.data[:first]:
                    lp = frozen.model.observe(byte)
                self.assertAlmostEqual(row[1], -float(lp[ep.data[first]]), places=6)
            self.assertEqual(m.model.dumps(), before)

    def test_live_checkpoint_continuation(self):
        for task in ("recall", "compression"):
            with self.subTest(task=task):
                spec = TaskSpec(task)
                cases = dataset(spec, 3, "train")
                with BenchModel(spec, 500000, steps=2) as m:
                    m.train_episode(cases[0], 10000)
                    # Preserve pending native credit across serialization.
                    m.model.run(b"abcde", b"bcdef")
                    with BenchModel.loads(m.dumps()) as restored:
                        self.assertEqual(m.train_episode(cases[1], 10000), restored.train_episode(cases[1], 10000))
                        self.assertEqual(m.model.dumps(), restored.model.dumps())
                        if m.head:
                            for k in m.head.arrays:
                                np.testing.assert_array_equal(m.head.arrays[k], restored.head.arrays[k])

    def test_loop_and_intervention_audits_preserve_fitted_source(self):
        spec = TaskSpec("recall")
        ep = dataset(spec, 1, "train")[0]
        with BenchModel(spec, 500000, steps=2) as m:
            m.train_episode(ep, 10000)
            m.model.finish()
            before = m.model.dumps()
            for steps in (0, 1, 2, 4):
                with m.clone(steps=steps) as frozen:
                    self.assertTrue(np.isfinite(frozen.score(ep)).all())
            for name in ("no_memory", "permute_l2", "permute_l3"):
                with m.clone(intervention=name) as frozen:
                    self.assertTrue(np.isfinite(frozen.score(ep)).all())
            self.assertEqual(before, m.model.dumps())

    def test_compression_uses_one_state_and_credit_changes_encoder(self):
        spec = TaskSpec("compression")
        ep = dataset(spec, 1, "train")[0]
        with BenchModel(spec, 500000) as m:
            self.assertEqual(len(ep.data) % 64, 0)
            m.train_episode(ep, 10000)
            self.assertEqual(m.inventory()["compression_bottleneck_bytes"], 256)
            with m.clone() as f:
                f.score(ep)
                x = f.features()
                p = f.head.predict(x)
                # Decoder computation uses only x and learned readout weights;
                # destroying the source context cannot change this prediction.
                f.model.reset_context(clear_memory=True)
                np.testing.assert_array_equal(p, f.head.predict(x))
            # A completed compression label produces live task credit in L3.
            self.assertGreater(m.model.stats()["task_gradient_energy"][3], 0)

    def test_budget_and_head_parameter_accounting(self):
        for task in ("recall", "compression"):
            spec = TaskSpec(task)
            ep = dataset(spec, 1, "train")[0]
            with BenchModel(spec, 500000) as m:
                used, _, _, task_loss = m.train_episode(ep, 5)
                self.assertEqual(used, 5)
                self.assertIsNone(task_loss)
                inv = m.inventory()
                self.assertLessEqual(inv["parameters"], 500000)
                self.assertGreater(inv["parameters"], 490000)


if __name__ == "__main__":
    unittest.main()
