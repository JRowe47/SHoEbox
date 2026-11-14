"""Matched-budget legacy/addressed comparisons, alternating order by seed."""
import argparse
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import benchmark, dependencies


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--targets", type=int, default=1_000_000)
    p.add_argument("--examples", type=int, default=6000)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    p.add_argument("--cpu", type=int, default=0)
    p.add_argument("--checkpoints", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    result = dict(python=platform.python_version(), numpy=np.__version__,
                  platform=platform.platform(), compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
                  cpu=args.cpu, language=[], dependencies=[])
    for index, seed in enumerate(args.seeds):
        profiles = ["legacy", "addressed"] if index % 2 == 0 else ["addressed", "legacy"]
        for profile in profiles:
            destination = out / f"{profile}_seed{seed}"
            language = SimpleNamespace(data=args.data, out=str(destination), cpu=args.cpu, targets=args.targets,
                seeds=[seed], levels=[4], parameters=1_000_000, lr=.08, aux=.1, router="indexed",
                associations=False, checkpoints=args.checkpoints, memory_profile=profile, reconstruction_weight=None)
            result["language"].extend(benchmark.run(language))
            dep = SimpleNamespace(out=str(destination), cpu=args.cpu, examples=args.examples, seeds=[seed],
                                  levels=[4], lr=.08, memory_profile=profile)
            result["dependencies"].extend(dependencies.main(dep))
            benchmark.atomic_json(out / "comparison.json", result)


if __name__ == "__main__":
    main()
