"""E8 changes exactly one initial coefficient, with distinct persistence."""
import hashlib
import struct
import unittest
import numpy as np
from .adapter import Adapter
from .tasks import dataset
from .adapter import prefix_labels
from experiments.joint_request_gate.adapter import Adapter as E7


def applicability_offset(payload):
    core_size = struct.unpack_from('<Q', payload, 12)[0]
    return 20 + core_size + 7605 * 4 + 4 + 2 + 8 + 16 + 4


class PriorTests(unittest.TestCase):
    def test_only_local_bias_changes(self):
        for seed in (1, 7, 19):
            for kind in ('native', 'paired_gated', 'paired_local'):
                with Adapter(seed=seed, kind=kind) as a, E7(seed=seed, kind=kind) as b:
                    x, y = bytearray(a.dumps()[32:]), bytearray(b.dumps()[32:])
                    self.assertNotEqual(x[:8], y[:8])
                    x[:8] = y[:8]
                    if kind == 'paired_local':
                        at = applicability_offset(x) + 64 * 4
                        self.assertAlmostEqual(struct.unpack_from('<f', x, at)[0], -np.log(255.), places=6)
                        x[at:at+4] = y[at:at+4]
                        self.assertAlmostEqual(a.gate(), 1/256, places=8)
                        self.assertEqual(b.gate(), .5)
                    self.assertEqual(x, y)

    def test_same_weights_keep_exact_e7_policy(self):
        with E7() as old:
            payload = bytearray(old.dumps()[32:])
            struct.pack_into('<Q', payload, 0, 0x38455041514853)
            # Test-only explicit format conversion; never used for E8 fitting.
            with Adapter.loads(hashlib.sha256(payload).digest() + payload) as new:
                for ep in dataset(8, 'train'):
                    old.reset(); new.reset()
                    args = (ep.data[:-1], ep.data[1:])
                    labels = prefix_labels(ep)
                    self.assertEqual(old.run(*args, labels=labels), new.run(*args, labels=labels))
                    self.assertEqual(old.dumps()[40:], new.dumps()[40:])
                    np.testing.assert_array_equal(old.log_probs(), new.log_probs())

    def test_e7_and_e8_reject_each_others_checkpoints(self):
        with Adapter() as a, E7() as b:
            with self.assertRaises(RuntimeError): E7.loads(a.dumps())
            with self.assertRaises(RuntimeError): Adapter.loads(b.dumps())

if __name__ == '__main__':
    unittest.main()
