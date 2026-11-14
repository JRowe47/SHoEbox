"""Protect benchmark boundaries: a fast result is useless if work differs."""
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from measure_current_baseline import fixtures, segments, summarize


class BaselineProtocolTests(unittest.TestCase):
    def test_document_targets_never_cross_boundaries(self):
        docs = [b'abc', b'DEFG', b'12']
        rows = list(segments(docs, 2, cuts=(1, 4)))
        self.assertEqual(b''.join(x for _, x, _, _ in rows), b'abDEF1')
        self.assertEqual(b''.join(y for _, _, y, _ in rows), b'bcEFG2')
        self.assertEqual(sum(end for _, _, _, end in rows), 3)
        self.assertEqual([lo for lo, _, _, _ in rows], [0, 1, 2, 4, 5])

    def test_fixed_workloads_have_exact_target_and_validation_counts(self):
        for workload in ('repeated', 'natural', 'documents'):
            # Unit tests must also work in a shallow CI checkout. Actual pinned
            # repository bytes/hashes are verified by the measurement suites.
            def source(*args):
                self.assertEqual(args[0], 'show')
                if args[1].endswith(':docs/cortical_fineweb.md'):
                    return b'Distinct held-out source. ' * 100
                self.assertTrue(args[1].endswith((':README.md', ':STATUS.md')))
                return b'Fixed natural text with explicit byte boundaries. ' * 1000
            with mock.patch('measure_current_baseline.git', side_effect=source):
                docs, validation, metadata = fixtures(workload, 17408)
            rows = list(segments(docs, 512, cuts=(1024, 3072, 5120, 9216, 17408)))
            self.assertEqual(sum(len(x) for _, x, _, _ in rows), 17408)
            self.assertEqual(len(validation), 1025)
            self.assertEqual(sum(len(x) for lo, x, _, _ in rows if lo < 1024), 1024)
            self.assertIn(1024, [lo for lo, _, _, _ in rows])
            self.assertEqual(metadata['documents'], len(docs))
            self.assertNotEqual(metadata['training_sha256'], metadata['validation_sha256'])

    def test_rejects_worker_divergence_and_repeat_drift(self):
        base = dict(workload='natural', targets=16, profile='compact_serial',
                    logits_sha256='a', continuation_sha256='b', bytes_per_second=100.,
                    seconds=.16, process_cpu_seconds=.15, heldout_nll=3., fitted_parameter_count=1000)
        summarize([base, {**base, 'profile': 'compact_threaded'}])
        with self.assertRaises(AssertionError):
            summarize([base, {**base, 'profile': 'compact_threaded', 'continuation_sha256': 'bad'}])
        with self.assertRaises(AssertionError):
            summarize([base, {**base, 'logits_sha256': 'bad'}])


if __name__ == '__main__':
    unittest.main()
