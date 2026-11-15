# L2/L3 language credit investigation

Baseline: corrected C, exact search, coverage warmup only, 4,999,743 parameters,
expert banks 571 / 285 / 171 / 116. D is retained as an explicit historical
control and is not used as the baseline.

## Result and selected policy

Keep corrected C. New public API/CLI models select C with coverage-only
balancing; historical explicit C/D and checkpoint policies are preserved.
No tested credit candidate establishes useful L2/L3 specialization.

The decisive probe replaces changing L2/L3 outputs with fixed means calibrated
on training inputs only. Across three 20M-target C runs, removing L3 modulation
increases development NLL by 0.057543, while replacing it with its mean costs
only 0.000031. Replacing both upper levels' modulation and learned recall with
their means changes NLL by -0.000031. Upper masking was mostly measuring a
constant bias, not useful conditional information.

The three-seed recall-to-expert 4/16 candidate improves development NLL from
1.731222 to 1.728440, but audit NLL changes from 1.688454 to 1.688576 and upper
expert permutation penalties remain negligible. Centering, with or without
normalizing the slow readout, also fails to improve the seed-7 10M screen.
See `results.json` for compact measurements and the delivered evidence archive
for individual runs, per-document losses, means and test logs.

## Interventions

The first screen varies only the language VJP entering L2/L3 native states:
1/1, 4/4, 4/16, 16/16 and 16/64. Gain/bias output-head updates, auxiliary
objectives and their rates remain unchanged. This screens language credit
separately from an ordinary learning-rate increase.

The second screen tests retained-value producer credit: language errors accumulated
through learned recall values are differentiated into both producing content
constituents while their weights remain frozen in the current bounded epoch.
Retirement clears producer IDs. Older retained values remain available for
recall with detached producers. No replay or unbounded recurrent graph is added.

Screen arms `r1`, `r4`, `r4_16` use retained-value gains 1/1, 4/4 and 4/16 at
L2/L3. `gr1` combines 1/1 retained-value credit with 4/16 native-state language
gains. All other levels and the fixed address channel preserve their credit.

`specialize.py` amplifies only the selected expert matrix/bias's recall-derived
gradient (4/16 or 16/64), while `state_expert.py` isolates the primary language
gradient originating at each upper native scale. Shared/lower paths remain
unchanged by the extra expert update. `local_head.py` adds two direct byte
prediction heads (33,150 parameters); `local_expert.py` restricts their teaching
to expert compute weights. `centered.py` consumes deviations from the slow
state's running training mean, optionally RMS-normalized. `mean_probe.py` is
the frozen fixed-mean replacement diagnostic, not a training intervention.

## Reproduction

Use the FineWeb files and manifests from the previous 5M routing evidence
archive. Training uses 20M unique next-byte targets in document-contained
chunks of at most 4,096 targets, shuffled by seeds 1, 7 and 19. Development
contains 262,144 targets across 70 documents. The reserved audit contains 1M
targets across 310 documents, disjoint from training and development by both
document ID and full-text hash. The audit is not used to choose a screen arm.

The schedule is unchanged: the previous warmup/cosine over 0–5M targets, then
fast/query 0.0002, slow 0.002 and head 0.004. Prototype attraction 0.002,
contrastive 0.1 and reconstruction 0.05 remain fixed.

```bash
python experiments/upper_credit/screen.py --work WORK --data DATA/fineweb --build
g++ -std=c++17 -O3 -fPIC -shared experiments/upper_credit/accessors.cpp -o WORK/accessors.so
python experiments/upper_credit/screen.py --work WORK --data DATA/fineweb --arm r1 --seed 7 --cpu 0 --targets 10000000
python experiments/upper_credit/screen.py --work WORK --data DATA/fineweb --arm r1 --seed 7 --cpu 0 --targets 20000000 --resume
python experiments/upper_credit/probe.py --work WORK --data DATA/fineweb --arm r1 --seed 7 --cpu 0 --full --audit DATA/audit
```

Repeat confirmation with `C` and the selected arm for seeds 1, 7 and 19.
Active-expert swaps rotate computation matrices and biases among experts that
actually occur on the complete development set, while keeping routing keys
fixed. Three nonidentity rotations are scored per level. Their mean loss
penalty measures whether the learned expert assignment helps prediction.

`benchmark.py` runs serial paired repetitions after other CPU work stops.
`verify_compat.py` checks old/new checkpoint continuation in separate runtime
processes. Public language and recall gains are stored in v11/v12 checkpoints.
The first gain screen used equivalent private global gains; its saved v10
files alone do not identify that experiment's learning policy. The current
`screen.py` reproduces the operation using public saved gains. `specialize.py`
also requires its matching gain installer. The private local-head,
local-expert, state-expert and centered forks use v13, v14, v15 and v16,
respectively, and require their own installer. Never load them as ordinary
public checkpoints.

For other families, build and train with that family's Python driver, then
use `probe.py` with the same work directory and arm. `mean_probe.py --build`
builds its diagnostic runtime; pass a public C checkpoint and `--seed` to run
it. `summarize.py --baseline C_WORK --candidate EXPERT_WORK --arm e4_16
--means MEAN_WORK --timing TIMING_JSON --out OUT` produces the primary chart
and summary. `write_report.py --work WORK --out OUT` collects all completed
screens and the ordinary corrected-C checkpoint. It expects the family
directory names recorded in the exported evidence.

Trained weights, data, compiled binaries and temporary snapshots stay out of
Git. The numerical and lifecycle checks live in `multiscale/tests/`.
