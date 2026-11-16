"""Shortcut-aware development calibration; test metrics never select a setting."""
import math


def normalized(accuracy, baseline):
    if not 0 <= accuracy <= 1 or not 0 <= baseline <= 1:
        raise ValueError('accuracy and baseline must be probabilities')
    return (accuracy-baseline)/(1-baseline) if baseline < 1 else None


def decision(native_rows, positive_rows, shortcuts, *, min_seeds=3, min_cases=512):
    """All rows supplied here are DEVELOPMENT rows for one fixed setting."""
    if not native_rows or not positive_rows:
        return {'status': 'incomplete', 'eligible_for_architecture_comparison': False}
    if len({r['seed'] for r in native_rows}) != len(native_rows) or len({r['seed'] for r in positive_rows}) != len(positive_rows):
        raise ValueError('duplicate seed')
    if {r['seed'] for r in native_rows} != {r['seed'] for r in positive_rows}:
        raise ValueError('unmatched seed coverage')
    weakest = min(r['dev']['exact_accuracy'] for r in positive_rows)
    learned = sum(r['dev']['exact_accuracy'] for r in positive_rows)/len(positive_rows)
    baseline_name = max(shortcuts, key=lambda k: shortcuts[k]['exact_accuracy'])
    baseline = shortcuts[baseline_name]['exact_accuracy']
    score = sum(r['dev']['exact_accuracy'] for r in native_rows)/len(native_rows)
    gain = normalized(score, baseline)
    enough = len(native_rows) >= min_seeds and min(r['dev']['cases'] for r in native_rows+positive_rows) >= min_cases
    positive_pass = learned >= .95 and weakest >= .90
    eligible = enough and positive_pass and gain is not None and .1 <= gain <= .9
    if baseline == 1:
        status = 'degenerate_shortcut_ceiling'
    elif not enough:
        status = 'provisional'
    elif not positive_pass:
        status = 'learned_positive_control_unresolved'
    elif eligible:
        status = 'calibrated'
    elif gain is not None and gain > .9:
        status = 'control_saturated'
    else:
        status = 'diagnostic_repair_required'
    return {'status': status, 'eligible_for_architecture_comparison': eligible,
            'native_mean_exact': score, 'strongest_shortcut': baseline_name,
            'strongest_shortcut_exact': baseline, 'shortcut_normalized_exact': gain,
            'positive_mean_exact': learned, 'positive_min_seed_exact': weakest,
            'learned_positive_pass': positive_pass,
            'interpretation': 'input-content calibration, not unseen-structure qualification'}
