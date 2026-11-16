"""Selection and cost accounting without interpolated test checkpoints."""
import math
import numpy as np


def normalized(accuracy, vocab):
    return (accuracy-1/vocab)/(1-1/vocab)


def at_budget(curve, key, limit):
    """Last actually measured checkpoint within a budget; no interpolation."""
    candidates = [p for p in curve if p[key] <= limit]
    return candidates[-1] if candidates else None


def first_threshold(curve, threshold=.9):
    # Require two consecutive measured checkpoints to suppress one-off peaks.
    for a, b in zip(curve, curve[1:]):
        if min(a["dev"]["task_accuracy"], b["dev"]["task_accuracy"]) >= threshold:
            return {"bytes": b["bytes"], "cpu_seconds": b["cpu_seconds"], "threshold": threshold}
    return None


def paired_interval(differences):
    """Seed-level bootstrap; a single seed has no estimated interval."""
    values = np.asarray(differences, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("finite paired differences required")
    out = {"n_seeds": len(values), "mean": float(values.mean()), "ci95": None}
    if len(values) > 1:
        rng = np.random.default_rng(47)
        means = rng.choice(values, (10000, len(values)), replace=True).mean(1)
        out["ci95"] = np.quantile(means, [.025, .975]).tolist()
    return out


def choose_hyperparameters(rows):
    """Use development loss only; reject unequal seed coverage."""
    groups = {}
    for r in rows:
        key = (r["lr_scale"], r["auxiliary"])
        groups.setdefault(key, []).append(r)
    seed_sets = [{r["seed"] for r in group} for group in groups.values()]
    if not groups or any(s != seed_sets[0] for s in seed_sets):
        raise ValueError("hyperparameter selection needs matched seed coverage")
    if any(len(g) != len({r["seed"] for r in g}) for g in groups.values()):
        raise ValueError("duplicate seed in hyperparameter candidate")
    scores = {k: float(np.mean([r["curve"][-1]["dev"]["task_nll"] for r in rs])) for k, rs in groups.items()}
    if not all(math.isfinite(v) for v in scores.values()):
        raise ValueError("nonfinite development score")
    return min(scores, key=lambda k: (scores[k], k))


def calibrate(rows, minimum=.1, maximum=.9, target=.5):
    candidates = []
    for r in rows:
        score = r["chance_normalized_accuracy"]
        if minimum <= score <= maximum:
            candidates.append(r)
    if not candidates:
        return {"status": "uncalibrated", "setting": None,
                "reason": "No control development score is in the difficulty band."}
    row = min(candidates, key=lambda r: (abs(r["chance_normalized_accuracy"]-target), r["setting"]))
    return {"status": "calibrated", "setting": row["setting"],
            "control_development_score": row["chance_normalized_accuracy"]}
