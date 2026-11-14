"""Reproduce v3/v4 delayed recall, counterfactuals and matched language runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import numpy as np

from . import benchmark, dependencies
from .model import Model


def task_example(rng, bit, gap, shift, task, other=0, question=0):
    if task == "two_cues":
        first = b"AB" if bit == 0 else b"BA"
        second = b"CD" if other == 0 else b"DC"
        answer = other if question else bit
        return (b"." * shift + first + b"x" * gap + second + b"x" * gap
                + (b"!" if question else b"?") + bytes([48 + answer]) + b"\n")
    cue = b"AB" if bit == 0 else b"BA"
    distractors = rng.choice(np.frombuffer(b"xyz", np.uint8), gap).tobytes()
    return b"." * shift + cue + distractors + b"?" + bytes([48 + bit]) + b"\n"


def extra_score(model, task, recall="full"):
    rng = np.random.default_rng(1201)
    rows = []
    with model.inference_copy() as frozen:
        if recall != "full":
            frozen.intervene_recall(recall)
        for gap in (16, 48, 96, 192, 768):
            correct = byte_correct = total = 0
            nll = 0.
            for shift in range(8):
                for other in range(2 if task == "two_cues" else 1):
                    for question in range(2 if task == "two_cues" else 1):
                        for bit in (0, 1):
                            sequence = task_example(rng, bit, gap, shift, task, other, question)
                            prefix, target = sequence[:-2], sequence[-2]
                            frozen.reset_context()
                            frozen.run(prefix[:-1], prefix[1:])
                            logits = frozen.observe(prefix[-1])
                            correct += int(np.argmax(logits[[48, 49]]) == target-48)
                            byte_correct += int(np.argmax(logits) == target)
                            nll -= float(logits[target])
                            total += 1
            rows.append(dict(gap=gap, accuracy=correct/total, byte_accuracy=byte_correct/total,
                             answer_nll=nll/total, trials=total))
    return rows


def delayed(profile, seed, examples):
    rng = np.random.default_rng(seed)
    data = b"".join(dependencies.example(int(rng.integers(2)), int(rng.choice([16,48,96])),
                                         int(rng.integers(4))) for _ in range(examples))
    with Model(seed=seed, memory_profile=profile) as model:
        start = time.perf_counter()
        nll = model.train(data)
        model.finish()
        seconds = time.perf_counter()-start
        row = dict(profile=profile, seed=seed, parameters=model.parameter_count,
                   train_targets=len(data)-1, train_sha256=hashlib.sha256(data).hexdigest(),
                   training_nll=nll, seconds=seconds,
                   scores=dependencies.score(model, gaps=(16,48,96,192,768,4096)),
                   stats=model.stats())
        if profile == "addressed":
            row["ablations"] = {mode: dependencies.score(model, gaps=(16,192,768), recall=mode)
                                for mode in ("no_read", "recency_only", "no_fixed_values", "no_learned_values")}
        # A fixed model must answer correctly on a continuous shuffled stream too.
        with model.inference_copy() as frozen:
            correct = 0
            for index in np.random.default_rng(991).permutation(256):
                sequence = dependencies.example(int(index % 2), (16,48,96,192)[index//2 % 4], int(index//8 % 8))
                for position, byte in enumerate(sequence):
                    logits = frozen.observe(byte)
                    if position == len(sequence)-3:
                        correct += int(np.argmax(logits) == sequence[-2])
            row["continuous_byte_accuracy"] = correct/256
        return row


def extra_task(task, seed, examples):
    rng = np.random.default_rng(seed)
    data = b"".join(task_example(rng, int(rng.integers(2)), int(rng.choice([16,48,96])),
                                int(rng.integers(4)), task, int(rng.integers(2)), int(rng.integers(2)))
                    for _ in range(examples))
    with Model(seed=seed) as model:
        start = time.perf_counter()
        model.train(data)
        model.finish()
        return dict(task=task, seed=seed, train_targets=len(data)-1,
                    train_sha256=hashlib.sha256(data).hexdigest(), seconds=time.perf_counter()-start,
                    scores=extra_score(model, task),
                    ablations={mode:extra_score(model, task, mode)
                               for mode in ("no_read", "recency_only")})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7,17,29])
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--examples", type=int, default=6000)
    parser.add_argument("--targets", type=int, default=1_000_000)
    parser.add_argument("--skip-language", action="store_true")
    parser.add_argument("--skip-recall", action="store_true")
    parser.add_argument("--extra-tasks", action="store_true")
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    if not args.skip_language and not args.data:
        parser.error("--data is required unless --skip-language is set")
    out = Path(args.out)
    result = dict(base="51030d6f7fdd8030c65cb6b2c3e5a10529fe90fe", python=platform.python_version(),
                  numpy=np.__version__, cpu=args.cpu, delayed=[], language=[], extra_tasks=[])
    def record():
        benchmark.atomic_json(out / "delayed_recall.json", result)
    for index, seed in enumerate(args.seeds):
        for profile in (["addressed_v3", "addressed"] if index % 2 == 0 else ["addressed", "addressed_v3"]):
            if not args.skip_recall:
                row = delayed(profile, seed, args.examples)
                result["delayed"].append(row)
                print(json.dumps({k:row[k] for k in ("profile","seed","scores","continuous_byte_accuracy")}), flush=True)
                record()
            if not args.skip_language:
                a = SimpleNamespace(data=args.data, out=str(out / f"{profile}_{seed}"), targets=args.targets,
                    seeds=[seed], levels=[4], parameters=1_000_000, lr=.08, aux=.1, cpu=args.cpu,
                    router="indexed", associations=False, checkpoints=False,
                    memory_profile=profile, reconstruction_weight=None)
                result["language"].extend(benchmark.run(a))
                record()
        if args.extra_tasks:
            for task in ("two_cues", "varied_distractors"):
                row = extra_task(task, seed, args.examples)
                result["extra_tasks"].append(row)
                print(json.dumps(row), flush=True)
                record()
    here = Path(__file__).parent
    result["source_sha256"] = {name:hashlib.sha256((here/name).read_bytes()).hexdigest()
                               for name in ("native.cpp", "native.hpp", "address_memory.hpp", "recall.hpp", "model.py")}
    record()


if __name__ == "__main__":
    main()
