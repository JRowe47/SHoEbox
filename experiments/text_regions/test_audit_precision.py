"""Audit portability accepts roundoff but rejects altered scientific results."""
import copy
import math
import unittest

from .audit_precision import same_audit_report


class AuditPrecisionChecks(unittest.TestCase):
    def test_roundoff_in_nested_losses(self):
        original = {'selected_strength': .25, 'selected_weighted_nll': .5324103793644503,
                    'candidates': [{'weighted_nll': .6003168299265393, 'eligible': True}],
                    'groups': [{'answer_nats_per_byte': .6823113150679168, 'correct': 32}]}
        rebuilt = copy.deepcopy(original)
        rebuilt['selected_weighted_nll'] += 2e-16
        rebuilt['candidates'][0]['weighted_nll'] += 2e-16
        rebuilt['groups'][0]['answer_nats_per_byte'] -= 3e-16
        self.assertTrue(same_audit_report(rebuilt, original))

    def test_real_loss_changes_and_nonfinite_values_fail(self):
        for key in ('weighted_nll', 'selected_weighted_nll', 'answer_nats_per_byte'):
            for value in (1.000001, math.nan, math.inf):
                with self.subTest(key=key, value=value):
                    self.assertFalse(same_audit_report({key: value}, {key: 1.0}))

    def test_discrete_decisions_and_metadata_are_exact(self):
        original = {'selected_strength': .25, 'eligible': True, 'correct': 32,
                    'predictions': [65, 66], 'inputs_sha256': 'abc'}
        for key, value in [('selected_strength', .25 + 1e-15), ('eligible', False),
                           ('correct', 33), ('predictions', [65, 67]),
                           ('inputs_sha256', 'abd')]:
            with self.subTest(key=key):
                rebuilt = {**original, key: value}
                self.assertFalse(same_audit_report(rebuilt, original))
        self.assertFalse(same_audit_report({'correct': 1}, {'correct': True}))


if __name__ == '__main__':
    unittest.main()
