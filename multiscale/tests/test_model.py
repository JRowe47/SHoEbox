import unittest

import numpy as np

from shoe_multiscale import Model


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.m = Model(parameters=200_000)

    def tearDown(self):
        self.m.close()

    def test_streaming_and_checkpoint_at_every_phase(self):
        a = np.random.default_rng(1).integers(0, 256, 1100, dtype=np.uint8)
        with self.m.clone() as whole:
            expected = whole.run(a[:-1], a[1:])
            weighted = 0
            start = 0
            for end in [1, 4, 8, 15, 32, 63, 64, 511, 512, 513, 1099]:
                weighted += self.m.run(a[start:end], a[start+1:end+1]) * (end-start)
                restored = Model.loads(self.m.dumps())
                self.m.close()
                self.m = restored
                start = end
            self.assertAlmostEqual(weighted/1099, expected, places=12)
            self.assertEqual(self.m.dumps(), whole.dumps())

    def test_evaluation_preserves_source(self):
        self.m.train(b"some training sequence!" * 20)
        before = self.m.dumps()
        self.assertTrue(np.isfinite(self.m.evaluate(b"unseen test text" * 20)))
        self.assertEqual(before, self.m.dumps())

    def test_future_targets_cannot_change_current_prediction(self):
        a = b"Prefix which has no future information yet"
        with self.m.clone() as other:
            self.m.run(a, b"a" * len(a))
            other.run(a, b"a" * (len(a)-1) + b"b")
            # The state and routing are from observed inputs; only the online
            # output-head update differs after the last target is consumed.
            self.assertEqual(self.m.stats()["state_energy"], other.stats()["state_energy"])
            self.assertEqual(self.m.stats()["events"], other.stats()["events"])

    def test_frozen_interventions(self):
        self.m.train(b"the slow states should affect the decoder. " * 50)
        text = b"the slow states should affect the decoder. " * 10
        baseline = self.m.evaluate(text)
        for level in (1, 2, 3):
            self.assertNotEqual(baseline, self.m.evaluate(text, mask=level))
            self.assertNotEqual(baseline, self.m.evaluate(text, shuffle=level))

    def test_store_payload_and_checkpoint(self):
        handle = self.m.bind("key", bytes(range(256)))
        self.m.train(b"distractor information " * 100)
        with Model.loads(self.m.dumps()) as restored:
            self.assertEqual(restored.lookup("key"), handle)
            self.assertEqual(restored.payload(handle), bytes(range(256)))
        self.assertIsNone(self.m.lookup("missing"))

    def test_learning_and_generation(self):
        data = b"The quick brown fox jumps over the lazy dog.\n" * 100
        before = self.m.evaluate(data)
        self.m.train(data)
        self.m.finish()
        self.assertLess(self.m.evaluate(data), before-2)
        self.assertEqual(len(self.m.generate("The ", length=32)), 36)

    def test_reset_invalidates_old_payload_handles(self):
        old = self.m.bind("key", b"old payload")
        self.m.reset_context(clear_memory=True)
        new = self.m.bind("key", b"new payload")
        self.assertNotEqual(old, new)
        with self.assertRaises(RuntimeError):
            self.m.payload(old)
        self.assertEqual(self.m.payload(new), b"new payload")

    def test_addressed_memory_and_native_clocks(self):
        self.m.train(b"four ordered children make a useful local pattern. " * 50)
        self.m.finish()
        s = self.m.stats()
        self.assertEqual(s["checkpoint_version"], 2)
        self.assertEqual(s["clock_ticks"], [0] + s["events"][1:])
        self.assertEqual(s["address_rows"], [0, 32, 32, 32])
        self.assertEqual(s["address_values"], [0, 16, 16, 16])
        self.assertEqual(s["reconstruction_targets"], [0] + s["events"][1:])
        self.assertTrue(all(v > 0 for v in s["query_content_energy"][1:]))
        text = b"four ordered children make a useful local pattern. " * 10
        baseline = self.m.evaluate(text)
        for mode in ("no_read", "no_time", "no_content"):
            self.assertNotEqual(baseline, self.m.evaluate(text, memory=mode))
        with self.m.inference_copy(cold=False) as warm, Model.loads(self.m.dumps()).inference_copy(cold=False) as restored:
            np.testing.assert_array_equal(warm.observe(65), restored.observe(65))
        self.m.reset_context()
        self.assertEqual(self.m.stats()["clock_ticks"], [0, 0, 0, 0])
        self.assertEqual(self.m.stats()["address_energy"], [0, 0, 0, 0])

    def test_reconstruction_reaches_local_experts(self):
        with self.m.clone() as control:
            self.m.configure(head_rate=0, auxiliary_weight=0, route_rate=0, reconstruction_weight=4)
            control.configure(head_rate=0, auxiliary_weight=0, route_rate=0, reconstruction_weight=0)
            data = b"composition and local feature learning " * 30
            self.m.train(data)
            control.train(data)
            self.m.finish()
            control.finish()
            self.assertEqual(control.stats()["reconstruction_targets"], [0, 0, 0, 0])
            self.assertNotEqual(self.m.evaluate(data), control.evaluate(data))

    def test_legacy_checkpoint_automatic_dispatch_and_continuation(self):
        a = np.random.default_rng(2).integers(0, 256, 550, dtype=np.uint8)
        with Model(parameters=150_000, memory_profile="legacy") as legacy:
            legacy.run(a[:67], a[1:68])
            with Model.loads(legacy.dumps()) as restored:
                self.assertEqual(restored.stats()["memory_profile"], "legacy")
                self.assertEqual(restored.stats()["checkpoint_version"], 1)
                self.assertEqual(legacy.run(a[67:-1], a[68:]), restored.run(a[67:-1], a[68:]))
                self.assertEqual(legacy.dumps(), restored.dumps())
                with self.assertRaises(ValueError):
                    restored.configure(reconstruction_weight=.1)

    def test_fast_only_profile_keeps_previous_arithmetic(self):
        data = b"a fast-only control uses precisely the original arithmetic. " * 20
        with Model(levels=1) as new, Model(levels=1, memory_profile="legacy") as old:
            self.assertEqual(new.train(data), old.train(data))
            np.testing.assert_array_equal(new.logits(), old.logits())

    def test_errors_and_lifecycle(self):
        for data in (b"", b"a"):
            with self.assertRaises(ValueError):
                self.m.train(data)
        with self.assertRaises(TypeError):
            self.m.run(np.array([1,2]), np.array([1,2]))
        with self.assertRaises(RuntimeError):
            self.m.configure(learning_rate=float("nan"))
        with self.assertRaises(RuntimeError):
            self.m.observe(12)
        with self.assertRaises(ValueError):
            Model.loads(self.m.dumps()[:-1])
        with self.assertRaises(RuntimeError):
            self.m.intervene_memory("no_read")
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.m.configure(reconstruction_weight=value)
        with self.m.inference_copy() as frozen:
            with self.assertRaises(RuntimeError):
                frozen.train(b"abc")
            with self.assertRaises(RuntimeError):
                frozen.intervene(mask=4)
        self.m.close()
        with self.assertRaises(RuntimeError):
            self.m.stats()


if __name__ == "__main__":
    unittest.main()
