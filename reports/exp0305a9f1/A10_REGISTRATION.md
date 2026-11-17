# EXP03/05-A10: exact row-parallel affine execution

Registered after completed, outcome-locked A9-F1 and its frozen diagnosis.
**Not implemented or measured. Research fitting budget: zero bytes.**
Keep qualified E8, every original result/timing/cost and closed A1–A9 audits.
Read the A9-F1 report, diagnosis and RESUME together with this registration.

## Single change and reason

A9 reaches 88.281250% competing, 88.281250% corrected overwrite, 94.531250%
clean and 86.328125% diverse256. Its isolated throughput is 68.202933% native,
below the unchanged 80% gate. The registered selective credit improves record
identity, but accuracy and execution speed both fail independently.

This next experiment addresses only the execution cost. In the isolated A9
execution backend, evaluate independent affine output rows in parallel while
retaining the original accumulation order within each row. Apply the one
kernel to existing multirow query-key, record-key and value projections only;
leave scalar presence/applicability, selection, sorting, chronology, codebook,
nonlinearity, local gradient arithmetic and native core unchanged. No other
optimization or scientific-policy change is part of A10.

Reference definition is the original `affine` function and its existing call
sites in the generated A9 header. Preserve bias placement and the exact
sequence of float32 multiplications/additions for every row. A lane holds one
row accumulator. Visit input coordinates in precisely the scalar-reference
order; issue separate multiply then add with FMA disabled. Preserve scalar
`tanh` evaluation after each completed row. No cross-coordinate reduction,
reassociation, altered precision, approximation, cutoff or fast-math is allowed.

Row coefficient packing or transposition may use bounded per-call scratch
only. Do not persist it between calls or across an update. Do not add a model
field, saved cache, sequence state, optimizer state, label, inference argument,
parser or decoder. Record scratch bytes and packing CPU. Do not change the
source-vector function, candidate enumeration or conditional execution of a
projection. If the compiler cannot establish this kernel without changing
per-row arithmetic, record failure; do not substitute another optimization.

The original A9 backend remains the immutable reference. Use an explicitly
selected, separately compiled execution backend for the candidate. This is
the same A9 numerical policy, with identical magic, checkpoint bytes and cursor
schema; no implicit migration or public default change. The execution backend
identity and complete source/compiler closure are external locked metadata.

## Exactness gate before any performance measurement

Write and commit a source/control lock before timing. Preserve every failed
control and all CPU. Required controls:

1. Independent scalar versus row-parallel affine references for every supported
   row count/phase and zero/bias/finite extreme inputs, including signed zero.
   Require bitwise output equality, with no tolerance substitution.
2. Original 20-test A9 suite, historical/public Python and optimized/ASan/UBSan
   controls, protected policy isolation, lifecycle, causality, 256 normalized
   byte outputs, exhaustive selection/ties/fallback and fixed-code gradients.
   Run both reference and candidate where the execution path differs.
3. All 18 archived A9-F1 warmup/final model-cursor pairs: compare reference
   and candidate at the identical saved cursor for 1,000 steps, and compare
   uninterrupted versus 371/save/load/629. These are discarded correctness
   trajectories, never new research checkpoints or scores. Charge each
   executed input/update and compare every full probability vector, native/
   local row, admission/state/counter and saved byte. Preserve exact episode
   flags, labels, partial spacer/block state and disabled FMA.
4. A9 credit enabled must retain original A8 scalar parity. Disabled credit
   still skips all and only the registered record-key events; no hidden extra
   code comparisons. Generic calls remain unable to train local heads.
5. Every frozen checkpoint file remains hash-identical before and after tests.
   No A1–A9 audit world is instantiated or scored. No development-score
   regeneration or diagnostic-driven modification is authorized.

An exactness failure stops timing. Correct implementation bugs with preserved
failed receipts; no arithmetic tolerance or alternate mechanism may replace
the registered one. Correctness continuation is explicitly discarded and
separately accounted; research fitting remains **zero**.

## Fixed fresh timing worlds and workload

No new training, development or audit split is needed for this execution-only
experiment. Reserve a fresh **16-world timing-only** sample before any timing.
Enumerate `World('binding', ((key0,value0),(key1,value1)))` with key0=0..63,
key1=key0+1..63, value0/value1=0..2, skipping equal values, in exactly that
nested order. Remove the unchanged A9 `tasks.excluded` union and **all 896
selected A9 primary groups plus all 128 F1 binary-development groups**.
Shuffle the complete remaining list once with `random.Random(20261202)` and
select its first 16. No salt search, alternative sample or outcome-based choice.
If fewer than 16 remain, stop and register an allocation amendment.

Use the unchanged A9 fixture constructor. For conditions in this exact order:
`pad0, repeat16, diverse16, compete16, overwrite16, diverse256`, append each of
the 16 worlds in selected order, all four variants 0..3, zero offset/spacers,
and `updated=world_index % 2`. Concatenate complete episode input bytes in
that order, exactly as the existing benchmark does. Labels and expected answers
are never supplied to the benchmark. Timing measures full 256-byte output at
every input. Independently implement the enumerator and concatenation; lock
ordered world hashes, stream SHA-256, length, PAD/nuisance counts and exclusions
before measurement. Mark these 16 timing identities as opened for future
allocation exclusions, without reporting accuracy.

Timing arms are: fresh untrained nominal-500K native v19; archived seed-7 E8;
archived seed-7 scalar A9; and the same A9 checkpoint under the row-parallel
backend. No fit, inherited warmup for fitting, calibration loop or profiler run.
Compile and load before intervals; charge their process costs separately.

On one pinned CPU with all other local research jobs finished, run exactly
three interleaved repeats in the arm order above, rotating left by repeat
index 0/1/2. Each arm/repeat resets context, warms with 1,024 bytes, resets
again, then measures 65,536 bytes. Repeat the fixed workload cyclically as
needed. Preserve every raw duration, checksum, byte count and first measurement;
no favorable replacement run. Candidate/scalar A9 checksums must match exactly.

## Decision and bounds

Execution gate: all exactness controls pass, candidate throughput is **≥80% of
native**, and candidate is no slower than scalar A9 using median CPU time of
the three fixed repeats. All three conditions are required. Report absolute
times and ratios on this fresh workload separately from the original A9
68.202933% measurement; do not overwrite that result or compare unlike timing
streams as if paired. Preserve public E8 and all historical backends.

An execution pass is **not a development or capability pass**. A9's original
accuracy failure remains; no primary/seed/position/offset/order/delay/lesion/
binary threshold changes. A later accuracy experiment requires its own single
focused preregistration, fresh excluded folds and independent fitting budget.
Do not execute it in A10. Generic/capacity qualification is allowed only after
a candidate passes every development gate under a separate registration.

Account for every correctness byte/update, reference comparison, projection,
scratch allocation, failed/repeated control and whole-process CPU. There are
zero pilots, sweeps, research fits, replay fits, audit evaluations or additional
inference-policy experiments. If this execution experiment fails, retain the
failure and write one focused next preregistration, without further measurement.

Completed verified work, including failure, is explicitly authorized for commit
and push directly to main without confirmation. Preserve unrelated changes and
remote history; never force-push. Keep datasets, checkpoints and compiled
binaries outside git. Deliver ordinary downloadable Markdown, primary PNGs,
resumable archive, verified commit link and another copy/paste prompt preserving
this authorization. No Pages or Spaces; export and attach without confirmation.
