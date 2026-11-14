"""Matched v2/v3 qualification, including the unresolved delayed-cue gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
from . import benchmark, dependencies

HERE = Path(__file__).resolve().parent


def diagnose(profile, examples, seed, compiler):
    rng = np.random.default_rng(seed)
    data = b"".join(dependencies.example(int(rng.integers(2)), int(rng.choice([16,48,96])),
                                        int(rng.integers(4))) for _ in range(examples))
    with tempfile.TemporaryDirectory(prefix="shoe-memory-diagnostic-") as temporary:
        tmp = Path(temporary)
        (tmp / "train.bin").write_bytes(data)
        old = profile == "addressed_v2"
        source = HERE / profile / "native.cpp"
        cmd = [compiler, "-std=c++17", "-O3", "-march=native", "-ffp-contract=off"]
        cmd.append("-DSHOE_ADDRESSED_V2" if old else "-DSHOE_ADDRESSED_V3")
        subprocess.run([*cmd, str(source), str(HERE / "memory_diagnostics.cpp"),
                        "-o", str(tmp / "diagnostics")], check=True)
        return json.loads(subprocess.check_output([str(tmp / "diagnostics"), str(tmp / "train.bin")], text=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--targets", type=int, default=1_000_000)
    p.add_argument("--examples", type=int, default=6000)
    p.add_argument("--seeds", type=int, nargs="+", default=[7,17,29])
    p.add_argument("--cpu", type=int, default=0)
    p.add_argument("--checkpoints", action="store_true")
    args = p.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    compiler = os.environ.get("CXX", "g++")
    out = Path(args.out)
    result = dict(base="4e52beede8ef800e5f7a52ecac612cd75ebb644b", python=platform.python_version(),
                  numpy=np.__version__, cpu=args.cpu, language=[], dependencies=[], diagnostics={})
    for index, seed in enumerate(args.seeds):
        profiles = ["addressed_v2", "addressed_v3"] if index % 2 == 0 else ["addressed_v3", "addressed_v2"]
        for profile in profiles:
            destination = out / f"{profile}_{seed}"
            a = SimpleNamespace(data=args.data, out=str(destination), targets=args.targets,
                seeds=[seed], levels=[4], parameters=1_000_000, lr=.08, aux=.1, cpu=args.cpu,
                router="indexed", associations=False, checkpoints=args.checkpoints,
                memory_profile=profile, reconstruction_weight=None)
            result["language"].extend(benchmark.run(a))
            a = SimpleNamespace(out=str(destination), examples=args.examples, seeds=[seed], levels=[4],
                                cpu=args.cpu, lr=.08, memory_profile=profile)
            result["dependencies"].extend(dependencies.main(a))
            benchmark.atomic_json(out / "memory_recall_repair.json", result)
    for profile in ("addressed_v2", "addressed_v3"):
        result["diagnostics"][profile] = diagnose(profile, args.examples, args.seeds[0], compiler)
    result["source_sha256"] = {str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [HERE / name for name in ("native.cpp", "native.hpp", "address_memory.hpp")]
                              + sorted((HERE / "addressed_v2").glob("*.?pp")) + sorted((HERE / "addressed_v3").glob("*.?pp"))}
    benchmark.atomic_json(out / "memory_recall_repair.json", result)
    print(json.dumps(result["diagnostics"], indent=2))


if __name__ == "__main__":
    main()
