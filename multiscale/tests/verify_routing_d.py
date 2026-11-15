"""Replay the selected default against recovered D source and FineWeb runs.

External inputs are intentional: datasets and trained checkpoints stay out of git.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np

from multiscale import Model


def chunks(documents, seed):
    segments = [(a, min(a+4096, d["stop"]-1)) for d in documents
                for a in range(d["start"], d["stop"]-1, 4096)]
    return [segments[i] for i in np.random.default_rng(seed).permutation(len(segments))]


def configure(model, done, requested):
    p = (done + requested/2) / 5_000_000
    scale = max(.01, p/.01) if p < .01 else .1 + .45*(1+math.cos(math.pi*min(1, (p-.01)/.99)))
    if done >= 5_000_000:
        scale *= .25
    model.configure(fast_rate=.008*scale, slow_rate=.08*scale, query_rate=.008*scale,
                    head_rate=.16*scale, route_rate=.002, auxiliary_weight=.1,
                    reconstruction_weight=.05, constituent_credit=1.)


def frozen_score(model, data, documents):
    losses = []
    with model.clone() as retired:
        retired.finish()
        with retired.inference_copy(cold=True, clear_memory=True) as frozen:
            for document in documents:
                frozen.reset_context(clear_memory=True)
                start, stop = document["start"], document["stop"]
                losses.append((frozen.run(data[start:stop-1], data[start+1:stop]), stop-start-1))
    return sum(loss*count for loss, count in losses) / sum(count for _, count in losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    spec = importlib.util.spec_from_file_location("historical_D", args.historical_model)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    arrays = {}
    for name in ("train", "development"):
        info = manifest["files"][name]
        raw = (args.data_dir / info["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == info["sha256"]
        arrays[name] = np.frombuffer(raw, np.uint8)
    train, dev = arrays["train"], arrays["development"]
    rows = []
    for seed in (1, 7, 19):
        with old.Model(seed=seed, router="scan", semantic_laces=False, routing_arm="D") as historical, Model(seed=seed) as current:
            assert current.dumps() == historical.dumps()
            done = 0
            for start, stop in chunks(manifest["files"]["train"]["documents"], seed):
                requested = min(4096, 100_000-done)
                end = min(stop, start+requested)
                for model in (historical, current):
                    configure(model, done, requested)
                    model.reset_context(clear_memory=True)
                assert historical.run(train[start:end], train[start+1:end+1]) == current.run(train[start:end], train[start+1:end+1])
                done += end-start
                if done == 100_000:
                    break
            assert done == 100_000 and historical.dumps() == current.dumps()
        folder = args.checkpoints / f"D_s{seed}_15m"
        checkpoint = folder / "latest.shoe"
        recorded = json.loads((folder / "results.json").read_text())["curve"][-1]["nll"]
        with Model.load(checkpoint) as current:
            observed = frozen_score(current, dev, manifest["files"]["development"]["documents"])
            assert observed == recorded
        with old.Model.load(checkpoint) as historical, Model.load(checkpoint) as current:
            assert historical.dumps() == current.dumps()
            assert historical.train(dev[:2001]) == current.train(dev[:2001])
            assert historical.dumps() == current.dumps()
        row = dict(seed=seed, fresh_targets=done, fresh_state_identical=True,
                   checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   development_nll=observed, recorded_nll_identical=True,
                   continued_targets=2000, continued_state_identical=True)
        rows.append(row)
        print(json.dumps(row), flush=True)
    source = Path(__file__).resolve().parents[1]
    names = ["native.cpp", "native.hpp", "address_memory.hpp", "recall.hpp",
             "semantic_laces.hpp", "semantic_laces_impl.hpp",
             "routing_geometry.hpp", "routing_geometry_impl.hpp"]
    hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names}
    for name in names:
        assert (source/name).read_bytes() == (args.historical_model.parent/name).read_bytes()
    result = dict(passed=True, seeds=rows, native_source_identical=True, native_sha256=hashes,
                  data_sha256={name: manifest["files"][name]["sha256"] for name in arrays})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
