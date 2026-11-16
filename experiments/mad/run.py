"""Plan or execute paired, reproducible MAD-style capability experiments."""
import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import time

# The small decoder is serial too. Set before importing NumPy/BLAS.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np

from .backend import BenchModel, ROOT
from .metrics import at_budget, calibrate, choose_hyperparameters, first_threshold, normalized, paired_interval
from .tasks import TASKS, TaskSpec, dataset, fingerprint

HERE = Path(__file__).resolve().parent


def evaluate(model, cases, *, spec=None, steps=None, intervention=None, rollout=True):
    cpu, wall = time.process_time(), time.perf_counter()
    totals = np.zeros(7)
    greedy = unrestricted = 0
    correct, exact = [], []
    with model.clone(steps=steps, intervention=intervention) as frozen:
        for ep in cases:
            row = frozen.score(ep)
            totals += row
            correct.append(row[3]/row[0])
            if rollout and not model.head:
                free = frozen.score(ep, 1)
                full = frozen.score(ep, 2)
                greedy += free[4]
                unrestricted += full[5]
                exact.append(int(free[4]))
            else:
                greedy += row[4]
                unrestricted += row[5]
                exact.append(int(row[4]))
        active = frozen.intervention_count
    s = spec or model.spec
    return dict(cases=len(cases), answer_tokens=int(totals[0]),
                task_accuracy=float(totals[3]/totals[0]), task_nll=float(totals[2]/totals[0]),
                native_byte_accuracy=float(totals[6]/totals[0]) if not model.head else None,
                native_byte_nll=float(totals[1]/totals[0]) if not model.head else None,
                exact_answer_accuracy=greedy/len(cases), native_exact_accuracy=unrestricted/len(cases) if not model.head else None,
                chance_normalized_accuracy=normalized(totals[3]/totals[0], s.vocab),
                chance_token_accuracy=1/s.vocab,
                cpu_seconds=time.process_time()-cpu, wall_seconds=time.perf_counter()-wall,
                evaluated_input_bytes=sum(len(e.data)-(0 if model.head else 1) for e in cases),
                active_experts_permuted=active, per_case_accuracy=correct, per_case_exact=exact)


def case_config(task, setting, protocol):
    base = TaskSpec(task, noise=.5 if task in ("noisy_recall", "selective_copy") else 0.)
    changes = protocol["settings"][setting].copy()
    count = changes.pop("train_examples", protocol["train_examples"])
    # Noise is a controlled distractor variable, not a different task family.
    if setting == "noise" and task in ("compression", "memorization"):
        return None
    return replace(base, **changes), count


def make_plan(args):
    protocol = json.loads((HERE/"protocol.json").read_text())
    smoke = args.preset == "smoke"
    config = dict(schema_version=1, preset=args.preset, protocol=protocol,
                  tasks=args.tasks or list(TASKS), settings=args.settings or (["base"] if smoke else list(protocol["settings"])),
                  sizes=args.sizes or ([500000] if smoke else protocol["sizes"]),
                  seeds=args.seeds or ([7] if smoke else protocol["seeds"]),
                  steps=args.steps or ([0, 2] if smoke else protocol["steps"]),
                  lr_scales=args.lr_scales or ([1.] if smoke else protocol["lr_scales"]),
                  auxiliary_weights=args.auxiliary or ([.1] if smoke else protocol["auxiliary_weights"]),
                  budget=args.budget if args.budget is not None else (4096 if smoke else protocol["budget_bytes"]),
                  eval_every=args.eval_every if args.eval_every is not None else (2048 if smoke else protocol["eval_every_bytes"]),
                  eval_examples=args.cases if args.cases is not None else (8 if smoke else protocol["eval_examples"]),
                  train_examples=args.train_examples if args.train_examples is not None else (16 if smoke else None),
                  audit_probes=args.audit_probes)
    if config["budget"] < 1 or config["eval_every"] < 1 or config["eval_examples"] < 1 or (config["train_examples"] is not None and config["train_examples"] < 1):
        raise ValueError("byte budgets and example counts must be positive")
    for name in ("tasks", "settings", "sizes", "seeds", "steps", "lr_scales", "auxiliary_weights"):
        if len(config[name]) != len(set(config[name])):
            raise ValueError(f"duplicate {name} would break matched selection")
    if any(s not in range(5) for s in config["steps"]) or any(s < 0 or s > 2**32-1 for s in config["seeds"]):
        raise ValueError("steps must be 0..4 and seeds uint32")
    if any(not np.isfinite(v) or v <= 0 for v in config["lr_scales"]) or any(not np.isfinite(v) or v < 0 for v in config["auxiliary_weights"]):
        raise ValueError("invalid learning policy grid")
    if any(n < 500000 or n > 100000000 for n in config["sizes"]):
        raise ValueError("protocol supports 500K..100M total parameter budgets")
    cells = []
    for task, setting, size, steps in itertools.product(config["tasks"], config["settings"], config["sizes"], config["steps"]):
        item = case_config(task, setting, protocol)
        if item is None:
            continue
        spec, count = item
        if task == "memorization" and (config["train_examples"] or count) < spec.vocab:
            raise ValueError("memorization training must cover every fixed-table key")
        cells.append(dict(task=task, setting=setting, size=size, steps=steps, spec=asdict(spec),
                          train_examples=config["train_examples"] or count))
    config["cells"] = cells
    config["fits"] = len(cells)*len(config["seeds"])*len(config["lr_scales"])*len(config["auxiliary_weights"])
    config["total_training_bytes"] = config["fits"]*config["budget"]
    return config


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def provenance():
    paths = sorted((ROOT/"multiscale").glob("*.hpp"))+[ROOT/"multiscale/native.cpp", ROOT/"multiscale/model.py"]
    paths += sorted(HERE.glob("*.py"))+[HERE/"native.cpp", HERE/"protocol.json"]
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.relative_to(ROOT)).encode()+b"\0"+p.read_bytes())
    return dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
                source_sha256=h.hexdigest(), python=platform.python_version(), numpy=np.__version__,
                platform=platform.platform(), compiler=subprocess.check_output([os.environ.get("CXX", "g++"), "--version"], text=True).splitlines()[0])


def train_fit(cell, config, seed, lr, auxiliary, train, dev, checkpoint):
    spec = TaskSpec(**cell["spec"])
    # Compilation and initialization are outside the timers.
    with BenchModel(spec, cell["size"], seed, cell["steps"], lr, auxiliary) as m:
        row = dict(seed=seed, lr_scale=lr, auxiliary=auxiliary, curve=[], inventory=m.inventory())
        total = native_targets = epochs = 0
        cpu = wall = loss_sum = 0.
        events = [0]*4
        passes = [0]*4
        order_rng = random.Random(1067)
        order = []
        digest = hashlib.sha256()
        next_eval = config["eval_every"]
        row["curve"].append(dict(bytes=0, cpu_seconds=0., wall_seconds=0., dev=evaluate(m, dev, rollout=False)))
        while total < config["budget"]:
            if not order:
                order = list(range(len(train)))
                order_rng.shuffle(order)
                epochs += 1
            index = order.pop()
            ep = train[index]
            begin, wall_begin = time.process_time(), time.perf_counter()
            n, targets, loss, _ = m.train_episode(ep, config["budget"]-total)
            cpu += time.process_time()-begin
            wall += time.perf_counter()-wall_begin
            total += n
            native_targets += targets
            loss_sum += loss
            digest.update(f"{index}:{n};".encode())
            # Memory inspection stays outside training timers and the byte loop.
            stats = m.model.stats()
            events = [a+b for a, b in zip(events, stats["events"])]
            passes = [a+b["passes"] for a, b in zip(passes, stats["loops"]["levels"])]
            if total >= next_eval or total == config["budget"]:
                begin, wall_begin = time.process_time(), time.perf_counter()
                m.model.finish()
                cpu += time.process_time()-begin
                wall += time.perf_counter()-wall_begin
                row["curve"].append(dict(bytes=total, cpu_seconds=cpu, wall_seconds=wall,
                                         dev=evaluate(m, dev, rollout=False)))
                while next_eval <= total:
                    next_eval += config["eval_every"]
        row.update(cpu_seconds=cpu, wall_seconds=wall, bytes=total, native_targets=native_targets,
                   train_byte_nll=loss_sum/native_targets if native_targets else None, bytes_per_cpu_second=total/cpu,
                   training_order_sha256=digest.hexdigest(), epochs_started=epochs,
                   native_events=events, refinement_passes=passes,
                   inventory=m.inventory(), first_confirmed_90pct=first_threshold(row["curve"]))
        checkpoint.write_bytes(m.dumps())
        with BenchModel.loads(checkpoint.read_bytes()) as restored:
            check = evaluate(restored, dev, rollout=False)
            final = row["curve"][-1]["dev"]
            if check["per_case_accuracy"] != final["per_case_accuracy"] or check["task_nll"] != final["task_nll"]:
                raise AssertionError("checkpoint predictions changed")
        row["checkpoint_reload_exact"] = True
        return row


def summarize(result):
    calibrated = []
    groups = {}
    for cell in result["cells"]:
        if cell["steps"] == 0:
            groups.setdefault((cell["task"], cell["size"]), []).append(dict(setting=cell["setting"],
                chance_normalized_accuracy=float(np.mean([r["curve"][-1]["dev"]["chance_normalized_accuracy"] for r in cell["selected_runs"]]))))
    for (task, size), rows in groups.items():
        decision = calibrate(rows)
        if decision["status"] == "calibrated" and (result["config"]["preset"] == "smoke" or
                len(result["config"]["seeds"]) < 3 or result["config"]["eval_examples"] < 128):
            decision["status"] = "provisional"
            decision["reason"] = "Calibration requires a non-smoke run, three seeds and at least 128 development cases."
        calibrated.append(dict(task=task, size=size, **decision))
    comparisons = []
    for cell in result["cells"]:
        if cell["steps"] == 0:
            continue
        controls = [c for c in result["cells"] if c["steps"] == 0 and all(c[k] == cell[k] for k in ("task", "size", "setting"))]
        if not controls:
            continue
        control = controls[0]
        a = {r["seed"]: r for r in control["selected_runs"]}
        b = {r["seed"]: r for r in cell["selected_runs"]}
        if a.keys() != b.keys():
            raise AssertionError("unpaired comparison")
        deltas, cost_rows = [], []
        for seed in sorted(a):
            x, y = a[seed], b[seed]
            # No checkpoint is invented at an interpolated training time.
            cap = min(x["cpu_seconds"], y["cpu_seconds"])
            px, py = at_budget(x["curve"], "cpu_seconds", cap), at_budget(y["curve"], "cpu_seconds", cap)
            deltas.append(y["test"]["task_accuracy"]-x["test"]["task_accuracy"])
            cost_rows.append(dict(seed=seed, common_cpu_budget=cap,
                control_bytes=px["bytes"], candidate_bytes=py["bytes"],
                control_dev_nll=px["dev"]["task_nll"], candidate_dev_nll=py["dev"]["task_nll"],
                throughput_ratio=y["bytes_per_cpu_second"]/x["bytes_per_cpu_second"]))
        comparisons.append(dict(task=cell["task"], size=cell["size"], setting=cell["setting"], steps=cell["steps"],
                                test_accuracy_delta=paired_interval(deltas), matched_time_development=cost_rows))
    # Equal weight to each setting, then task; byte length never weights a task.
    macro = []
    for size, steps in sorted({(c["size"], c["steps"]) for c in result["cells"]}):
        by_task = {}
        for c in result["cells"]:
            if (c["size"], c["steps"]) == (size, steps):
                by_task.setdefault(c["task"], []).append(float(np.mean([r["test"]["task_accuracy"] for r in c["selected_runs"]])))
        scores = {k: float(np.mean(v)) for k, v in by_task.items()}
        macro.append(dict(size=size, steps=steps, per_task=scores, macro_accuracy=float(np.mean(list(scores.values())))))
    return dict(calibration=calibrated, comparisons=comparisons, macro_scores=macro,
                promotion="not assessed; requires calibrated multi-seed evidence and a separate natural-language guardrail")


def execute(config, out):
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {affinity})
    result = dict(config=config, provenance=provenance(), cpu=affinity, cells=[], status="running")
    atomic_json(out, result)
    datasets = {}
    # Temporary checkpoints allow all hyperparameters to be selected BEFORE
    # any test labels are evaluated, without retaining giant model snapshots.
    with tempfile.TemporaryDirectory(prefix="shoe-mad-checkpoints-") as temporary:
        for number, cell in enumerate(config["cells"]):
            s = TaskSpec(**cell["spec"])
            key = (s, cell["train_examples"])
            if key not in datasets:
                kwargs = dict(seed=config["protocol"]["data_seed"], map_seed=config["protocol"]["map_seed"])
                train = dataset(s, cell["train_examples"], "train", **kwargs)
                dev = dataset(s, config["eval_examples"], "dev", **kwargs)
                test = dataset(s, config["eval_examples"], "test", **kwargs)
                datasets[key] = train, dev, test
            train, dev, test = datasets[key]
            record = dict(**cell, hashes={k: fingerprint(v) for k, v in zip(("train", "dev", "test"), (train, dev, test))}, runs=[])
            checkpoints = {}
            for lr, aux, seed in itertools.product(config["lr_scales"], config["auxiliary_weights"], config["seeds"]):
                path = Path(temporary)/f"{number}_{lr}_{aux}_{seed}.mad"
                row = train_fit(cell, config, seed, lr, aux, train, dev, path)
                checkpoints[lr, aux, seed] = path
                record["runs"].append(row)
                print(f"{number+1}/{len(config['cells'])} {cell['task']} {cell['setting']} {cell['size']} steps={cell['steps']} seed={seed} lr={lr} aux={aux}: dev={row['curve'][-1]['dev']['task_accuracy']:.4f}", flush=True)
            orders = {r["training_order_sha256"] for r in record["runs"]}
            if len(orders) != 1:
                raise AssertionError("unmatched training exposure")
            lr, aux = choose_hyperparameters(record["runs"])
            record["selected_hyperparameters"] = dict(lr_scale=lr, auxiliary=aux)
            record["total_tuning_cpu_seconds"] = sum(r["cpu_seconds"] for r in record["runs"])
            selected = [r for r in record["runs"] if (r["lr_scale"], r["auxiliary"]) == (lr, aux)]
            for row in selected:
                with BenchModel.loads(checkpoints[lr, aux, row["seed"]].read_bytes()) as m:
                    row["test"] = evaluate(m, test)
                    if config["audit_probes"]:
                        row["depth_audit"] = {str(k): evaluate(m, test, steps=k) for k in (0, 1, 2, 4)}
                        names = ("no_memory", "permute_l2", "permute_l3")+( ("zero_bottleneck",) if m.head else ())
                        row["interventions"] = {k: evaluate(m, test, intervention=k) for k in names}
                        # Difficulty OOD changes delay only: decoding dimensions
                        # and vocabulary remain identical, including compression.
                        ood_spec = replace(s, delay=min(4096, max(64, 2*s.delay)))
                        ood = dataset(ood_spec, config["eval_examples"], "test", seed=19810423,
                                      map_seed=config["protocol"]["map_seed"])
                        row["delay_ood"] = dict(spec=asdict(ood_spec), hash=fingerprint(ood), metrics=evaluate(m, ood, spec=ood_spec))
            record["selected_runs"] = selected
            # Verify paired data, state capacities and total parameters across arms.
            for prior in result["cells"]:
                if all(prior[k] == record[k] for k in ("task", "setting", "size")):
                    if prior["hashes"] != record["hashes"]:
                        raise AssertionError("unmatched datasets")
                    for field in ("parameters", "recurrent_vector_bytes", "minkron_matrix_bytes", "address_matrix_bytes", "retained_record_bytes"):
                        if prior["runs"][0]["inventory"][field] != record["runs"][0]["inventory"][field]:
                            raise AssertionError(f"unmatched {field}")
            result["cells"].append(record)
            atomic_json(out, result)
            for path in checkpoints.values():
                path.unlink()
    result["summary"] = summarize(result)
    result["status"] = "complete"
    atomic_json(out, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("smoke", "screen"), default="smoke")
    p.add_argument("--execute", action="store_true", help="otherwise print/write the plan without fitting models")
    p.add_argument("--out", type=Path)
    p.add_argument("--tasks", nargs="+", choices=TASKS)
    p.add_argument("--settings", nargs="+", choices=("base", "length", "vocabulary", "delay", "noise", "low_data"))
    p.add_argument("--sizes", type=int, nargs="+")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--steps", type=int, nargs="+")
    p.add_argument("--lr-scales", type=float, nargs="+")
    p.add_argument("--auxiliary", type=float, nargs="+")
    p.add_argument("--budget", type=int)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--cases", type=int)
    p.add_argument("--train-examples", type=int)
    p.add_argument("--audit-probes", action="store_true")
    args = p.parse_args()
    try:
        plan = make_plan(args)
        if args.execute:
            if args.out is None:
                p.error("--execute requires --out")
            if args.out.exists():
                p.error("output already exists; select a new file to preserve previous evidence")
            execute(plan, args.out)
        elif args.out:
            if args.out.exists():
                p.error("output already exists")
            atomic_json(args.out, plan)
            print(f"Planned {plan['fits']} fits / {plan['total_training_bytes']:,} training bytes: {args.out}")
        else:
            print(json.dumps(plan, indent=2))
    except ValueError as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
