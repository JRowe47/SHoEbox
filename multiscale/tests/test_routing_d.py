"""Selected public defaults, historical policies, and CLI continuation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from shoe_multiscale import Model, fresh_model


class RoutingDTests(unittest.TestCase):
    def test_default_matches_explicit_c_at_every_stream_phase(self):
        data = np.random.default_rng(5).integers(0, 256, 2004, dtype=np.uint8)
        for seed in (1, 7, 19):
            with fresh_model(seed=seed) as default, Model(
                seed=seed, router="scan", semantic_laces=False, routing_arm="C", balance_policy="warmup"
            ) as explicit:
                self.assertEqual(default.stats()["routing_arm"], "C")
                self.assertEqual(default.stats()["checkpoint_version"], 10)
                self.assertFalse(default.stats()["semantic_laces"])
                self.assertEqual(default.parameter_count, 999_615)
                self.assertEqual(default.dumps(), explicit.dumps())
                start = 0
                for end in (1, 4, 8, 31, 64, 511, 512, 513, 2003):
                    x, y = data[start:end], data[start+1:end+1]
                    self.assertEqual(default.run(x, y), explicit.run(x, y))
                    with Model.loads(default.dumps()) as resumed:
                        self.assertEqual(resumed.dumps(), default.dumps())
                        self.assertEqual(resumed.stats()["routing_arm"], "C")
                    start = end
                self.assertEqual(default.dumps(), explicit.dumps())
                before = default.dumps()
                self.assertTrue(np.isfinite(default.evaluate(data)))
                self.assertEqual(default.dumps(), before)
                with Model.loads(before) as resumed:
                    self.assertEqual(default.train(data), resumed.train(data))
                    self.assertEqual(default.dumps(), resumed.dumps())

    def test_explicit_historical_arms_keep_saved_behavior(self):
        data = b"Historical policy must survive reload and continuation. " * 23
        for arm, version in (("A", 7), ("B", 8), ("C", 9), ("D", 9)):
            with Model(routing_arm=arm) as source:
                self.assertEqual(source.stats()["routing_arm"], arm)
                self.assertEqual(source.stats()["checkpoint_version"], version)
                self.assertEqual(source.stats()["semantic_laces"], arm == "A")
                source.train(data)
                with Model.loads(source.dumps()) as restored:
                    self.assertEqual(restored.stats()["routing_arm"], arm)
                    self.assertEqual(source.train(data), restored.train(data))
                    self.assertEqual(source.dumps(), restored.dumps())
        with Model(semantic_laces=True) as old, Model(routing_arm="A") as explicit:
            self.assertEqual(old.dumps(), explicit.dumps())

    def test_invalid_policy_combinations_are_rejected(self):
        for options in (dict(routing_arm="H"), dict(routing_arm="D", semantic_laces=True),
                        dict(routing_arm="D", router="indexed"),
                        dict(memory_profile="legacy", routing_arm="D")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Model(**options)

    def test_cli_default_resume_and_inspection(self):
        data = b"A short stream for exact CLI continuation. " * 40
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "first.txt").write_bytes(data[:514])
            (folder / "next.txt").write_bytes(data[513:])
            checkpoint = folder / "model.shoe"
            command = [sys.executable, "-m", "multiscale.cli"]
            subprocess.run(command + ["train", "--data", str(folder / "first.txt"),
                           "--out", str(checkpoint)], check=True, capture_output=True)
            with Model() as expected:
                expected.train(data[:514])
                self.assertEqual(checkpoint.read_bytes(), expected.dumps())
                subprocess.run(command + ["train", "--data", str(folder / "next.txt"),
                               "--out", str(checkpoint), "--resume", str(checkpoint)],
                               check=True, capture_output=True)
                expected.train(data[513:])
                self.assertEqual(checkpoint.read_bytes(), expected.dumps())
            result = subprocess.run(command + ["inspect", "--checkpoint", str(checkpoint)],
                                    check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)["routing_arm"], "C")
            result = subprocess.run(command + ["train", "--data", str(folder / "next.txt"),
                                    "--out", str(checkpoint), "--resume", str(checkpoint),
                                    "--routing-arm", "A"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("resume preserves saved routing", result.stderr)


if __name__ == "__main__":
    unittest.main()
