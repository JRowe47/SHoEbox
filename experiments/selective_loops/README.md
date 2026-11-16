# Selective multiscale loop screen

The completed-constituent loop implementation uses a tied selected expert and
64-value working state at the existing 4/16/64-byte scales. Preset extra passes
are 1/2/2, strength 0.25, and relative convergence tolerance 0.01. No additional
parameters, routing searches, memory admissions or stream-time ticks are added.
See [the architecture and API](../../multiscale/SELECTIVE_LOOPS.md).

The first short screen keeps corrected C as the default. Three matched seeds
(1, 7, 19), 999,615 parameters, and 1M unique Tiny Shakespeare training targets
per fit produce mean held-out NLL **1.861549 for loops vs 1.853721 for C**.
Paired changes are +0.022000, -0.005028, +0.006511 nats/byte. This screen does
not establish a learning improvement or measure long-horizon reasoning.

Serial median throughput on one pinned AMD EPYC 9V74 CPU core is **31,518 vs
32,851 training bytes/CPU-second** (-4.06%), and **49,744 vs 50,211 inference
bytes/CPU-second** (-0.93%). Three alternating 100K-target probes use the seed-1
fitted models. Compilation, loading and saving are outside these timers;
training reset and final retirement are included.

The 0.01 convergence threshold skips no passes in these fits. Stopping is
implemented and checked with early-exit cases; learned hardness selection and
learned halting remain separate future work.

Validation passes: 44 Python tests, 1,439 loop-specific native checks, and all
existing native regression harnesses, including ASan/UBSan runs. LeakSanitizer
was explicitly disabled on this host. The first full pass exposed extra stack
usage in checkpoint dispatch; splitting dispatch from the v18 loader fixed it,
and the full suite then completed successfully. All six trained checkpoints
also reproduce their measured held-out losses exactly with the final source.

Raw measurements and the precise split, hashes, schedule and timing boundaries
are in `results.json`. `compatibility.json` verifies byte-identical old/new
checkpoints and 4,097-target continuations for corrected C, historical D, and
route-local learning with language/recall credit enabled. Old checkpoints keep
loops off. Explicit zero-loop models also have identical base checkpoint bytes.

```bash
python -m multiscale.tests.run_all
python -m experiments.selective_loops.compare --data input.txt --out results.json
python -m experiments.selective_loops.report --results results.json --out output
python -m experiments.selective_loops.verify_compat --old /path/to/e2a2e93 --out compatibility.json
```

Datasets, weights and native binaries are excluded. The comparison is a small
implementation screen; the supplied runner accepts other training budgets that
fit the unique training prefix. The report renderer is for this recorded
1M-budget, three-seed screen.
