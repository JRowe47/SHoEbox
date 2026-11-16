"""MAD decision rules applied to the earlier structured-state experiment.

This is a registered extension, not a replacement encoding of the nine canonical
MAD byte tasks. Native next-byte training and the identical four-register head
are retained to isolate the earlier supervision question.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

from experiments.structured_state.adapter import Adapter
from experiments.structured_state.tasks import episode, verify
from experiments.structured_state.run import evaluate, train_episode, training_mean
from experiments.structured_state.diagnose import extract, ridge
from experiments.mad.metrics import at_budget, calibrate, normalized, paired_interval
from experiments.mad.run import atomic_json

ARMS = ("final_only", "head_only", "state_aux")
SEEDS = (1, 7, 19)


def cases(seed, count, fold, length):
    rng = random.Random(seed)
    result, seen = [], set()
    while len(result) < count:
        ep = episode(rng, ("tracking", "composition")[len(result) % 2], length, fold)
        verify(ep)
        if ep.text in seen:
            continue
        seen.add(ep.text)
        result.append(ep)
    return result


def digest(rows):
    return hashlib.sha256(b"".join(ep.text for ep in rows)).hexdigest()


def evaluate_metric(model, rows):
    out = evaluate(model, rows)
    out["task_accuracy"] = out["accuracy"]
    out["task_nll"] = out["nll"]
    out["chance_normalized_accuracy"] = normalized(out["accuracy"], 8)
    return out


def fit(size, seed, arm, length, budget, train, dev, path):
    # The old adapter adds 4,128 fitted head weights outside its core request.
    with Adapter(size - 4128, seed, arm, budget) as model:
        row = dict(size=size, seed=seed, arm=arm, length=length,
                   parameters=int(model.info()[0]), curve=[])
        total, seconds, wall_seconds, next_eval = 0, 0., 0., budget // 5
        order, rng = [], random.Random(1067)
        exposure = hashlib.sha256()
        row["curve"].append(dict(bytes=0, cpu_seconds=0., dev=evaluate_metric(model, dev)))
        while total < budget:
            if not order:
                order = list(range(len(train)))
                rng.shuffle(order)
            index = order.pop()
            begin, wall = time.process_time(), time.perf_counter()
            used, _ = train_episode(model, train[index], budget-total)
            total += used
            seconds += time.process_time()-begin
            wall_seconds += time.perf_counter()-wall
            exposure.update(f"{index}:{used};".encode())
            if total >= next_eval or total == budget:
                begin, wall = time.process_time(), time.perf_counter()
                model.finish()
                seconds += time.process_time()-begin
                wall_seconds += time.perf_counter()-wall
                row["curve"].append(dict(bytes=total, cpu_seconds=seconds,
                                         dev=evaluate_metric(model, dev)))
                while next_eval <= total:
                    next_eval += budget // 5
        row.update(bytes=total, cpu_seconds=seconds, wall_seconds=wall_seconds,
                   bytes_per_cpu_second=total/seconds,
                   training_order_sha256=exposure.hexdigest(), info=model.info())
        path.write_bytes(model.dumps())
        with Adapter.loads(path.read_bytes()) as restored:
            check = evaluate_metric(restored, dev)
            assert check["correct"] == row["curve"][-1]["dev"]["correct"]
            assert check["nll"] == row["curve"][-1]["dev"]["nll"]
        row["checkpoint_reload_exact"] = True
        return row


def run(out, size=1000000, budget=1000000):
    if out.exists():
        raise ValueError("Refusing to overwrite prior evidence")
    if hasattr(os, "sched_getaffinity"):
        cpu = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpu})
    else:
        cpu = None
    settings = (1, 2, 4)
    data = {}
    for length in settings:
        data[length] = tuple(cases(seed, count, fold, length) for seed, count, fold in
            ((20261116, 256, "train"), (20261117, 128, "dev"), (20261118, 256, "audit")))
        train, dev, test = data[length]
        for a, b in ((train, dev), (train, test), (dev, test)):
            assert not {e.template for e in a} & {e.template for e in b}
    config = dict(size=size, budget=budget, seeds=list(SEEDS), lengths=list(settings),
        train_cases=256, dev_cases=128, test_cases=256, cpu=cpu,
        search_budget="one fixed matched policy per arm; no test-based tuning",
        schedule="native shared anchor 1 to 0.1; auxiliary 0.25; task head 0.2",
        calibration="final_only development accuracy across three seeds",
        encoding="existing verified 64-byte records; all native next-byte targets",
        total_fitted_budget_includes_head=True,
        state_capacity="four 64-float recurrent states, 16 retained slots per level",
        hashes={str(k):dict(zip(("train","dev","test"),map(digest,v))) for k,v in data.items()},
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    result = dict(config=config, calibration_fits=[], selected_runs=[], status="running")
    atomic_json(out, result)
    with tempfile.TemporaryDirectory(prefix="mad-state-") as temporary:
        root = Path(temporary)
        for length in settings:
            train, dev, _ = data[length]
            for seed in SEEDS:
                path = root / f"{length}_{seed}_final_only.task"
                row = fit(size,seed,"final_only",length,budget,train,dev,path)
                result["calibration_fits"].append(row)
                print("STATE_CALIBRATION",length,seed,row["curve"][-1]["dev"]["accuracy"],flush=True)
                atomic_json(out,result)
        scores = [dict(setting=str(length), chance_normalized_accuracy=float(np.mean(
            [r["curve"][-1]["dev"]["chance_normalized_accuracy"] for r in result["calibration_fits"] if r["length"]==length]))) for length in settings]
        choice = calibrate(scores)
        # If no point calibrates, the easiest preregistered task remains a
        # diagnostic rerun only; it cannot yield a promotion decision.
        length = int(choice["setting"]) if choice["setting"] else 1
        result["calibration"] = dict(**choice, scores=scores, diagnostic_length=length)
        train, dev, test = data[length]
        selected = [r for r in result["calibration_fits"] if r["length"]==length]
        for arm in ARMS[1:]:
            for seed in SEEDS:
                path = root / f"{length}_{seed}_{arm}.task"
                row = fit(size,seed,arm,length,budget,train,dev,path)
                selected.append(row)
                print("STATE_FIT",arm,seed,row["curve"][-1]["dev"]["accuracy"],flush=True)
        assert len({r["training_order_sha256"] for r in selected}) == 1
        assert len({r["parameters"] for r in selected}) == 1
        assert selected[0]["parameters"] <= size
        probe_train = cases(60019,1024,"train",length)
        long = cases(19810424,128,"audit",8) + cases(19810425,128,"audit",12)
        # Every test use comes after task difficulty and all fit policies freeze.
        for row in selected:
            path = root / f'{length}_{row["seed"]}_{row["arm"]}.task'
            with Adapter.loads(path.read_bytes()) as model:
                row["test"] = evaluate_metric(model,test)
                row["long"] = evaluate_metric(model,long)
                mean = training_mean(model,probe_train[:256])
                row["mean_replacement"] = evaluate(model,test,mean=mean)
                row["permutations"] = [evaluate(model,test,permutation=s) for s in (23,47,71)]
                x,y,q,_ = extract(model,probe_train)
                v,vy,vq,online = extract(model,test)
                row["frozen_standardized_probes"] = {}
                for stage in (0,1):
                    key = "before_query" if stage==0 else "after_query"
                    row["frozen_standardized_probes"][key] = ridge(x[stage],y,v[stage],vy,vq,np.array([e.family for e in test]))
                row["feature_mean_energy_fraction"] = float(np.square(x[1].mean(0)).sum()/np.square(x[1]).sum(1).mean())
            result["selected_runs"].append(row)
            atomic_json(out,result)
            print("STATE_RESULT",json.dumps({k:row[k] for k in ("seed","arm","parameters","bytes","cpu_seconds")}|{"test_accuracy":row["test"]["accuracy"],"dev_accuracy":row["curve"][-1]["dev"]["accuracy"],"probe_accuracy":row["frozen_standardized_probes"]["after_query"]["accuracy"],"long_accuracy":row["long"]["accuracy"]}),flush=True)
        comparisons=[]
        for a,b in (("final_only","head_only"),("head_only","state_aux"),("final_only","state_aux")):
            aa={r["seed"]:r for r in selected if r["arm"]==a}
            bb={r["seed"]:r for r in selected if r["arm"]==b}
            costs=[]
            for seed in SEEDS:
                cap=min(aa[seed]["cpu_seconds"],bb[seed]["cpu_seconds"])
                pa,pb=(at_budget(z["curve"],"cpu_seconds",cap) for z in (aa[seed],bb[seed]))
                costs.append(dict(seed=seed,cpu_budget=cap,control_bytes=pa["bytes"],candidate_bytes=pb["bytes"],control_dev_nll=pa["dev"]["nll"],candidate_dev_nll=pb["dev"]["nll"],throughput_ratio=bb[seed]["bytes_per_cpu_second"]/aa[seed]["bytes_per_cpu_second"]))
            comparisons.append(dict(control=a,candidate=b,test_accuracy_delta=paired_interval(
                [bb[s]["test"]["accuracy"]-aa[s]["test"]["accuracy"] for s in SEEDS]),matched_time=costs))
        result["comparisons"]=comparisons
        # Task-aware simple baseline: ignore all updates, retain only init.
        result["initial_binding_baseline"]=float(np.mean([e.states[0][e.query]==e.answer for e in test]))
        result["status"]="complete"
        atomic_json(out,result)
        print("STATE_SUMMARY",json.dumps({"calibration":result["calibration"],"comparisons":comparisons,"initial_binding_baseline":result["initial_binding_baseline"]}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--size",type=int,default=1000000)
    p.add_argument("--budget",type=int,default=1000000)
    args=p.parse_args()
    if args.budget < 1000:
        p.error("budget must be at least 1000")
    run(args.out,args.size,args.budget)


if __name__=="__main__":
    main()
