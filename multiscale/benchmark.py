"""Reproducible matched-byte language comparison and inference interventions.

python -m multiscale.benchmark --data /path/to/input.txt --out /tmp/rebuild
No datasets, checkpoints, or compiled binaries are written inside the repository.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .model import Model


def atomic_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def run(args):
    os.sched_setaffinity(0, {args.cpu})
    raw = Path(args.data).read_bytes()
    split = int(len(raw)*.9)
    if args.targets >= split:
        raise ValueError("training budget crosses the held-out split")
    data = np.frombuffer(raw, np.uint8)
    test = data[split:]
    all_results = []
    for seed in args.seeds:
        for levels in args.levels:
            with Model(levels=levels, parameters=args.parameters, seed=seed,
                       associations=args.associations, router=args.router,
                       memory_profile=args.memory_profile) as model:
                model.configure(reconstruction_weight=args.reconstruction_weight)
                result = dict(seed=seed, levels=levels, parameters=model.parameter_count,
                              experts=model.stats()["experts"], cpu=args.cpu, targets=args.targets,
                              learning_rate=args.lr, auxiliary_weight=args.aux,
                              router=args.router, associations=args.associations,
                              memory_profile=args.memory_profile,
                              reconstruction_weight=model.stats().get("reconstruction_weight", 0),
                              data_sha256=hashlib.sha256(raw).hexdigest(), split_index=split,
                              curve=[], training_seconds=0., cpu_seconds=0.)
                result["curve"].append(dict(targets=0, seconds=0., nll=model.evaluate(test[:16385])))
                elapsed = cpu_elapsed = 0
                for start in range(0, args.targets, 100_000):
                    end = min(args.targets, start+100_000)
                    loss = 0.
                    for lo in range(start, end, 4096):
                        hi = min(end, lo+4096)
                        progress = (lo+hi)/2/args.targets
                        schedule = max(.01, progress/.01) if progress < .01 else .1+.9*.5*(1+math.cos(math.pi*(progress-.01)/.99))
                        model.configure(learning_rate=args.lr*schedule, head_rate=.16*schedule,
                                        auxiliary_weight=args.aux, route_rate=.002)
                        t, c = time.perf_counter(), time.process_time()
                        loss += model.run(data[lo:hi], data[lo+1:hi+1])*(hi-lo)
                        elapsed += time.perf_counter()-t
                        cpu_elapsed += time.process_time()-c
                    row = dict(targets=end, seconds=elapsed, nll=model.evaluate(test[:16385]),
                               training_nll=loss/(end-start))
                    result["curve"].append(row)
                    print(json.dumps(dict(seed=seed, levels=levels, **row)), flush=True)
                t, c = time.perf_counter(), time.process_time()
                model.finish()
                elapsed += time.perf_counter()-t
                cpu_elapsed += time.process_time()-c
                result.update(training_seconds=elapsed, cpu_seconds=cpu_elapsed,
                              bytes_per_second=args.targets/elapsed, test_nll=model.evaluate(test),
                              test_targets=len(test)-1, stats=model.stats())
                result["intervention_baseline_nll"] = model.evaluate(test[:16385])
                result["interventions"] = {str(l):dict(mask=model.evaluate(test[:16385], mask=l),
                                                          shuffle=model.evaluate(test[:16385], shuffle=l))
                                           for l in range(1, levels)}
                if args.memory_profile == "addressed":
                    result["memory_interventions"] = {mode:model.evaluate(test[:16385], memory=mode)
                                                       for mode in ("no_read", "no_time", "no_content")}
                with model.inference_copy() as frozen:
                    t = time.perf_counter()
                    frozen.run(test[:-1], test[1:])
                    result["teacher_forced_inference_bps"] = (len(test)-1)/(time.perf_counter()-t)
                t = time.perf_counter()
                result["samples"] = [model.generate(p, length=256, seed=100+i).decode("utf-8", "replace")
                                     for i,p in enumerate(["ROMEO:\n", "First Citizen:\n", "To be, or not to be"])]
                result["sample_generation_bps"] = 768/(time.perf_counter()-t)
                if args.checkpoints:
                    model.save(Path(args.out)/f"level{levels}_seed{seed}.shoe")
                all_results.append(result)
                atomic_json(Path(args.out)/"language.json", all_results)
                print(json.dumps({k:v for k,v in result.items() if k not in ("curve", "samples", "stats")}), flush=True)
    return all_results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--targets", type=int, default=1_000_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[7,17,29])
    p.add_argument("--levels", type=int, nargs="+", default=[1,2,4])
    p.add_argument("--parameters", type=int, default=1_000_000)
    p.add_argument("--lr", type=float, default=.08)
    p.add_argument("--aux", type=float, default=.1)
    p.add_argument("--cpu", type=int, default=0)
    p.add_argument("--router", choices=["indexed","scan"], default="indexed")
    p.add_argument("--associations", action="store_true")
    p.add_argument("--checkpoints", action="store_true")
    p.add_argument("--memory-profile", choices=["addressed", "legacy"], default="addressed")
    p.add_argument("--reconstruction-weight", type=float)
    run(p.parse_args())
