# Cortical SHoEbox: wider transport and active capacity

`cortical_wide()` keeps the 18-region cortical graph and EC-only hippocampus,
while widening the interfaces and concentrating capacity at orders 1 and 2.
It has **14,976,882 parameters**. The original `cortical_10m()` configuration
and checkpoints remain supported.

| Capacity | Previous 10M | Wide profile |
| --- | --- | --- |
| Internal context / compact / primary port | 1 / 2 / 4 bytes | 4 / 8 / 16 bytes |
| A1 / B1 width; expert hidden width | 64; 16 | 128; 32 |
| L2 / S2 / R2 / F2 width; expert hidden width | 64; 16 | 96; 24 |
| Remaining regional widths | 64 | 64 |
| Selected experts per regional layer | 1 of 64 | 4 of 32 |
| HAT symbols: orders 1–2 / remaining | 64 / 64 | 256 / 128 |
| Active HAT symbols: orders 1–2 / remaining | 2 / 2 | 8 / 4 |
| Training gradient horizon | 2 observations | 32 observations |
| Learning rate | Short-run cosine schedule | Warm up for 256 targets, then 0.001 constant |

The first two orders own about 55% of the parameters. Reducing the number of
inactive experts frees capacity for wider active paths and output heads. Four
experts participate in each layer, drawn from two groups. Relative to the old
configuration, active expert matrix capacity grows 16x at order 1, 9x at order
2, and 4x elsewhere. Output ports are independent learned heads: a 4-byte
context code is not a slice of its producer's 16-byte primary code.

One 16-byte publication is one event's code, not sixteen observations of
history. Both recent complete occurrences remain available in the recipes.
Only the external observation and final emitted character remain single bytes.
The wider ports do not add a replay store or a regional long-term archive.

## Temporal learning and cold start

The run carries recurrent and episodic state across chunks, and detaches the
gradient every 32 observations. MinKron's internal 32-position computation
chunk is a separate mechanism: it does not detach gradients. Four streams
each consume 250 observations, producing exactly 1,000 scored byte targets
and eight optimizer updates (seven batches of 128 targets and one of 104).

Multiplying port widths by four also multiplies the number of internal HAT
byte steps. The new preset divides initial non-ingress HAT decay rates by four
to compensate. B1 retains its independently slower starting clock. These
rates remain trainable; the compensation is an initialization, not a fixed
memory controller. The learning rate warms up and then remains constant.

The final HAT level selects symbols only at the publication position. Every
incoming byte still writes the recurrent state, and gradients pass through
those writes. Earlier levels, if configured, still emit full sequences.
Symbol balance in the new preset therefore applies to published symbols.
This is an intentional objective-scope change from the original preset.

## Optimized execution

Use `model.set_execution_backend('optimized')`. The reference backend uses
the same wide configuration and objective, providing a numerical comparison.

- Project source, origin, position and boundary metadata before broadcasting;
  cache those small projections for one forward call.
- Cache normalized router/codebook keys and lace templates within that same
  forward call. No parameter-dependent cache survives an optimizer step.
- Compute all MinKron writes but only the final consumed HAT query/read.
- Skip generic shared-bank dispatch when every cortical bank is private, and
  skip the unconsumed second lace calculation and final refinement message.
- Reuse port softmax probabilities for diagnostics and straight-through bytes.
- At a TBPTT boundary, keep exact integer frames and remove their detached dense
  one-hot distributions. Live-window distributions retain their gradients.

The tests compare values, final recurrent states and gradients, including
multiple MinKron chunks and the heterogeneous cortical graph.

## Measurements

`PortMonitor` records the former 1-byte context ports (now 4-byte ports) and
the primary A1/B1/L2 ports (now 16-byte ports). It records final publications
once per external observation, rather than counting recurrent revisions twice.
All entropy values are **bits per byte**, with maximum 8:

- Conditional soft entropy: confidence of each output distribution.
- Marginal soft entropy and conditioning JS: variation of distributions with
  input/state, where JS = H(mean probabilities) − mean H(probabilities).
- Empirical hard-byte entropy, unique complete frames, largest frame share,
  and frame collision probability: actual transmitted code diversity.
- Selected expert diversity for each region and layer.

Near-uniform soft probabilities can still produce a constant argmax code;
high hard diversity alone can also be untrained noise. Both views are logged.
Counters reset at each training window. The initial/final probe uses identical
128 target positions and freshly reset state. It is a training-slice diagnostic.
The old 100k checkpoint is probed on the same slice only for port comparison.

The temporal diagnostic differentiates the last prediction's task loss alone
with respect to A1/B1/L2 latents at lags 0, 2, 8, 16 and 31 observations.
Its extra backward time is reported separately. Nonzero gradients establish
a path; their magnitudes show whether that path is materially strong.

Saved-tensor accounting deduplicates underlying storage. This is referenced
storage during the forward, not simultaneous peak RSS. Packed saved tensors
are detached aliases; returning the original tensor object would create
autograd reference cycles. The failed diagnostic attempt and corrected run
are recorded separately.

## Reproduce

Use the packaged `profile_train.bin`: 1,004 bytes including four unscored
stream prefixes. Run from the repository root into a fresh output directory:

```bash
python scripts/check_all.py
python -m shoebox.profile_wide --mode train --backend optimized \
  --data artifacts/cortical_wide_profile/profile_train.bin \
  --out artifacts/wide_reproduction --targets 1000 --batch 4 --context 32 \
  --threads 1 --lr 0.001 --warmup-targets 256

python -m shoebox.profile_wide --mode benchmark --backend reference \
  --data artifacts/cortical_wide_profile/profile_train.bin \
  --out artifacts/wide_reproduction
python -m shoebox.profile_wide --mode benchmark --backend optimized \
  --data artifacts/cortical_wide_profile/profile_train.bin \
  --out artifacts/wide_reproduction

python -m shoebox.profile_wide --mode operators --backend reference \
  --data artifacts/cortical_wide_profile/profile_train.bin \
  --out artifacts/wide_reproduction
python -m shoebox.profile_wide --mode operators --backend optimized \
  --data artifacts/cortical_wide_profile/profile_train.bin \
  --out artifacts/wide_reproduction
```

The benchmark uses the same weights, eight warmup observations, and a live
32-observation batch, in separate processes. Operator profiling uses one
observation per stream at the final checkpoint; its allocation counts are
cumulative traffic, not live memory. Extra benchmark and diagnostic passes
do not update weights and are outside the 1,000-target training budget.

The checkpoint includes weights, optimizer, carried state, cursor and RNG.
The profile runner intentionally refuses to overwrite a completed endpoint;
it does not resume a partially completed profile.

Measured values and interpretation are in `results.json`, the matched
benchmark files, operator profiles and the downloadable Markdown/PDF report.

The completed matched benchmark is **16.6% faster** (30.18 -> 25.89 seconds)
and reduces peak process RSS **23.4%** (5.21 -> 3.99 GiB). Corrected training
finishes all 1,000 targets in 194.62 seconds of forward/backward/update work;
the gradient audit adds 10.42 seconds. Forward RSS plateaus around 4.3 GiB.
All 17 tests and the endpoint reload/data/state audit pass. The fixed training
probe falls from 5.584954 to 5.060832 nats/byte. Port confidence remains near
uniform, despite varied hard codes; full measurements preserve that distinction.
