"""Compare rebuilt audit reports without weakening discrete result checks."""
import math

# Only these computed loss fields may differ by floating-point roundoff.
LOSS_FIELDS = frozenset(('weighted_nll', 'selected_weighted_nll', 'answer_nats_per_byte'))


def same_audit_report(actual, expected, *, field=None):
    """Allow 1e-12 loss roundoff; keep decisions, counts and other data exact."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        return (actual.keys() == expected.keys() and
                all(same_audit_report(actual[k], expected[k], field=k) for k in actual))
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return (len(actual) == len(expected) and
                all(same_audit_report(a, b, field=field)
                    for a, b in zip(actual, expected)))
    if field in LOSS_FIELDS and type(actual) is float and type(expected) is float:
        return (math.isfinite(actual) and math.isfinite(expected) and
                math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12))
    return type(actual) is type(expected) and actual == expected
