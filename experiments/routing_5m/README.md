# 5M routing diagnosis

The selected change keeps D's predictive cues and removes the continuing
starvation-bias term. Initial coverage warmup and subsequent bias decay remain.
Three matched seeds at 20M FineWeb training targets gave mean development NLL
1.752425 for historical D and 1.728077 for corrected D; audit NLL changed from
1.712747 to 1.687145. Reverting the predictor alone to C did not help.

Corrected C is a useful speed comparator: 31,282 versus 28,808 bytes/s, with
mean development NLL 1.731222 versus 1.728077 at the same 20M-byte budget.
At the common estimated serial time budget, corrected C is slightly ahead.
The upper L2/L3 expert-swap probes still show little specialization.

See `multiscale/results/routing_5m_selection.json` for exact measurements,
checkpoint digest, protocol, and verification summary. The downloadable
evidence package includes the report, PNG, CSV curves, per-document scores,
data splits, baseline runtime, and verification logs. Trained weights and data
are excluded from Git.

## Reproduce the comparisons

Requirements: Python, NumPy, Matplotlib, g++, and Linux CPU affinity support.
Run from the runtime root. Paths below are relative to the extracted evidence
package; choose a CPU available to your process.

```bash
python experiments/routing_5m/run.py --work ../rerun --data ../data/fineweb --build
python experiments/routing_5m/verify.py --work ../rerun --data ../data/fineweb
g++ -std=c++17 -O3 -fPIC -shared experiments/plateau_100m/accessors.cpp -o ../rerun/accessors.so
python experiments/routing_5m/run.py --work ../rerun --data ../data/fineweb --arm D_warmup_only --seed 7 --cpu 0 --targets 20000000
python experiments/routing_5m/probe.py --work ../rerun --data ../data/fineweb --audit-data ../data/audit/train.bin --audit-protocol ../data/audit/protocol.json --arm D_warmup_only --seed 7 --cpu 0
```

Repeat the last two commands for D, C, D_warmup_only, C_warmup_only and seeds
1, 7, 19. The other screen arms use seed 7 and 10M targets. `launch.py` schedules
up to eight processes on CPUs 0–7; invoke individual runs on smaller hosts.
`confirm_queue.py` and `launch_probes.py` record the original overlapping job
schedule and require its screening outputs.

The diagnostic controls are process globals and are reapplied by `install()`.
They are deliberately absent from experiment checkpoint serialization. Always
use the matching arm when resuming a diagnostic run. The supplied production
checkpoint is different: it persistently stores the corrected v10 policy and
requires no diagnostic controls.

To rebuild the chart/report from the included results without training:

```bash
python experiments/routing_5m/summarize.py --work ../evidence --out ../reports
python experiments/routing_5m/write_report.py --work ../evidence --out ../reports
```

The serial benchmark requires the regenerated per-arm checkpoints. Run it
after other CPU jobs stop, using `--selected-root . --original-root ../baseline`
and `--data ../data/timing`. The archive omits redundant trained checkpoints.
Its archived timing measurements remain available without retraining.

## Public runtime behavior

`Model()` creates D with warmup-only balancing. Explicit historical
`Model(routing_arm="C")` or `"D"` keeps v9; add
`balance_policy="warmup"` to select the correction. V8/v9 saved models retain
their old policy on load. Calling `set_balance_policy("warmup")` explicitly
changes future updates while preserving existing bias, weights and stream state.

The exported seed-7 checkpoint has 5,012,031 parameters and exactly 20M lifetime
training targets. `stats()["bytes"]` counts the current context, whereas
`routing_stats()["levels"][0]["events"]` records lifetime training selections.
The saved rates are fast/query 0.0002, slow 0.002, and head 0.004.
