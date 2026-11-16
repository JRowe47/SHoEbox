"""Large expert-bank construction and lossless checkpoint continuation."""
import hashlib
import unittest
import numpy as np
from shoe_multiscale import Model


class LargeModelTests(unittest.TestCase):
    def test_100m_budget_and_checkpoint_continuation(self):
        data = np.random.default_rng(23).integers(0, 256, 779, dtype=np.uint8)
        with Model(parameters=100_000_000, seed=7, core="full") as m:
            self.assertEqual(m.parameter_count, 99_997_503)
            self.assertEqual(m.stats()['experts'], [11816, 5908, 3544, 2365])
            self.assertEqual(m.stats()['routing_arm'], 'C')
            m.run(data[:513], data[1:514])
            raw = m.dumps()
            with Model.loads(raw) as resumed:
                self.assertEqual(hashlib.sha256(resumed.dumps()).digest(), hashlib.sha256(raw).digest())
                del raw
                self.assertEqual(m.run(data[513:-1], data[514:]), resumed.run(data[513:-1], data[514:]))
                self.assertEqual(hashlib.sha256(m.dumps()).digest(), hashlib.sha256(resumed.dumps()).digest())

    def test_out_of_range_budget_is_rejected(self):
        with self.assertRaises(RuntimeError):
            Model(parameters=100_000_001)


if __name__ == '__main__':
    unittest.main()
