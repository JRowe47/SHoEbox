"""Fixed-recipe, paired minimal/full comparison; no test-set selection.

python -m experiments.minimal_core.run --data PATH --out results.json
Checkpoints are temporary. Only provenance and small numerical results persist.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from multiscale.model import Model
from experiments.mad.backend import BenchModel
from experiments.mad.run import atomic_json, evaluate, provenance
from experiments.mad.tasks import TaskSpec, dataset, fingerprint


def compact(value):
    return {k: v for k, v in value.items() if not k.startswith("per_case_")}


def language(core, seed, train, dev, checkpoint):
    with Model(parameters=500000, seed=seed, core=core) as m:
        cpu = wall = 0.
        curve = []
        for start in range(0, 1000000, 250000):
            c, w = time.process_time(), time.perf_counter()
            loss = m.train(train[start:start+250001])
            m.finish()
            cpu += time.process_time()-c
            wall += time.perf_counter()-w
            curve.append(dict(bytes=start+250000, train_nll=loss,
                              dev_nll=m.evaluate(dev, clear_memory=True), cpu_seconds=cpu))
        checkpoint.write_bytes(m.dumps())
        with Model.loads(checkpoint.read_bytes()) as restored:
            assert restored.evaluate(dev, clear_memory=True) == curve[-1]["dev_nll"]
        stats = m.stats()
        return dict(task="fineweb", core=core, seed=seed, parameters=stats["parameters"],
                    experts=stats["experts"], bytes=1000000, cpu_seconds=cpu, wall_seconds=wall,
                    bytes_per_cpu_second=1000000/cpu, curve=curve,
                    dev_nll=curve[-1]["dev_nll"], checkpoint_reload_exact=True)


def capability(core, seed, spec, lr, train, dev, checkpoint):
    with BenchModel(spec, parameters=500000, seed=seed, lr_scale=lr, auxiliary=0., core=core) as m:
        total = targets = 0
        cpu = wall = 0.
        order = []
        rng = random.Random(1067)
        digest = hashlib.sha256()
        while total < 250000:
            if not order:
                order = list(range(len(train)))
                rng.shuffle(order)
            index = order.pop()
            c, w = time.process_time(), time.perf_counter()
            used, scored, _, _ = m.train_episode(train[index], 250000-total)
            cpu += time.process_time()-c
            wall += time.perf_counter()-w
            total += used
            targets += scored
            digest.update(f"{index}:{used};".encode())
        c, w = time.process_time(), time.perf_counter()
        m.model.finish()
        cpu += time.process_time()-c
        wall += time.perf_counter()-w
        development = evaluate(m, dev)
        checkpoint.write_bytes(m.dumps())
        with BenchModel.loads(checkpoint.read_bytes()) as restored:
            check = evaluate(restored, dev)
            assert check["per_case_accuracy"] == development["per_case_accuracy"]
            assert check["task_nll"] == development["task_nll"]
        inventory = m.inventory()
        return dict(task=spec.task, core=core, seed=seed, parameters=inventory["parameters"],
                    experts=inventory["experts_per_level"], inventory=inventory,
                    bytes=total, native_targets=targets, cpu_seconds=cpu, wall_seconds=wall,
                    bytes_per_cpu_second=total/cpu, dev=compact(development),
                    training_order_sha256=digest.hexdigest(), checkpoint_reload_exact=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cpu = min(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    streams = {key: (args.data/name).read_bytes() for key, name in
               (("train", "train.bin"), ("dev", "validation.bin"), ("test", "test.bin"))}
    assert list(map(len, streams.values())) == [1000001, 65537, 65537]
    recipes = [(TaskSpec("selective_copy", length=2, noise=.5), 2.),
               (TaskSpec("compression", vocab=16), .5)]
    cases = {spec.task: {fold: dataset(spec, count, fold) for fold, count in
             (("train", 256), ("dev", 128), ("test", 128))} for spec, _ in recipes}
    result = dict(schema_version=1, status="training", provenance=provenance(), cpu=cpu,
        protocol=dict(seeds=[1, 7, 19], nominal_parameters=500000, cores=["full", "minimal"],
            language_bytes=1000000, capability_bytes=250000, auxiliary=0.,
            recipes=[dict(spec=asdict(spec), lr_scale=lr) for spec, lr in recipes],
            selection="Fixed historical recipes; all fits frozen before testing; no tuning on this run.",
            criteria="Language mean NLL delta <= 0.01; capability mean token accuracy loss <= 0.01.",
            timing="One pinned process; CPU time; initialization, evaluation and I/O excluded; reset and finish included.",
            inference_timing="One frozen native pass, includes reset; excludes cloning and compilation."),
        data_sha256={k: hashlib.sha256(v).hexdigest() for k, v in streams.items()},
        task_sha256={t: {f: fingerprint(e) for f, e in folds.items()} for t, folds in cases.items()}, runs=[])
    atomic_json(args.out, result)
    with tempfile.TemporaryDirectory(prefix="shoe-minimal-qualification-") as folder:
        checkpoints = {}
        for task in ("fineweb", "selective_copy", "compression"):
            for seed in (1, 7, 19):
                # Alternate pair order to reduce systematic timing drift.
                for core in (("full", "minimal") if seed != 7 else ("minimal", "full")):
                    path = Path(folder)/f"{task}-{seed}-{core}.checkpoint"
                    checkpoints[task, seed, core] = path
                    if task == "fineweb":
                        row = language(core, seed, streams["train"], streams["dev"], path)
                    else:
                        spec, lr = next(x for x in recipes if x[0].task == task)
                        row = capability(core, seed, spec, lr, cases[task]["train"], cases[task]["dev"], path)
                    result["runs"].append(row)
                    atomic_json(args.out, result)
                    print(json.dumps({k: row[k] for k in ("task", "seed", "core", "parameters", "cpu_seconds")} |
                                     {"dev": row.get("dev_nll", row.get("dev"))}), flush=True)
        result["status"] = "frozen_test"
        atomic_json(args.out, result)
        for row in result["runs"]:
            task = row["task"]
            raw = checkpoints[task, row["seed"], row["core"]].read_bytes()
            if task == "fineweb":
                with Model.loads(raw) as m:
                    row["test_nll"] = m.evaluate(streams["test"], clear_memory=True)
                    times = []
                    with m.inference_copy(clear_memory=True) as frozen:
                        for _ in range(3):
                            c = time.process_time()
                            frozen.reset_context(clear_memory=True)
                            frozen.run(streams["test"][:-1], streams["test"][1:])
                            times.append(time.process_time()-c)
                    row["inference_bytes_per_cpu_second"] = 65536/float(np.median(times))
            else:
                with BenchModel.loads(raw) as m:
                    row["test"] = compact(evaluate(m, cases[task]["test"]))
                    probe = "no_memory" if task == "selective_copy" else "zero_bottleneck"
                    row["probe"] = dict(intervention=probe, **compact(evaluate(m, cases[task]["test"], intervention=probe)))
            atomic_json(args.out, result)
    result["summary"] = []
    for task in ("fineweb", "selective_copy", "compression"):
        by_core = {core: [r for r in result["runs"] if r["task"] == task and r["core"] == core]
                   for core in ("full", "minimal")}
        score = lambda r: r["test_nll"] if task == "fineweb" else r["test"]["task_accuracy"]
        means = {c: float(np.mean([score(r) for r in rows])) for c, rows in by_core.items()}
        delta = means["minimal"]-means["full"]
        result["summary"].append(dict(task=task, full=means["full"], minimal=means["minimal"], delta=delta,
            passes_preregistered_margin=bool(delta <= .01 if task == "fineweb" else delta >= -.01),
            mean_training_throughput={c: float(np.mean([r["bytes_per_cpu_second"] for r in rows])) for c, rows in by_core.items()}))
    result["status"] = "complete"
    atomic_json(args.out, result)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
