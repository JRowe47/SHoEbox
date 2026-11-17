# EXP03/05-A9: selective record-key credit (registered, not implemented or fitted)

Registered after the completed, outcome-locked A8 campaign and frozen diagnosis.
Keep qualified public E8, all original scores/timings/costs and closed A1–A8
audits. This document authorizes one future experiment; no A9 code or fit has run.

## Measured motivation and single change

A8 fails development: competing 80.859375%, corrected three-value overwrite
78.515625%, clean 87.890625%, diverse256 83.593750%, isolated throughput
68.587044% native. Fresh matched A6 scores are 87.109375%, 92.187500%,
92.968750%, 81.250000% and 67.443791%. Preserve the different original A6
and A7 campaign results; these are A8-world controls.

On the first-32-world, four-order, both-answer teacher-fed diagnosis, both A6
and A8 retain 100% of target signatures, with no target removals or true-record
phase false negatives. Diverse-pair false recognition falls 3.212851% to
1.857430%; competing nearest-key identity falls 92.968750% to 88.802083%.
Changed-key old-value predictions rise 3.125000% to 7.812500%. A8 record-key
reconstruction squared error is 5.170884, versus query error 0.220569 on the
competing trace. Wrong nearest keys differ by 3–8 code bits; this is not a
code collision. These subset measures are not full autoregressive outcomes.

A8 successfully supplies both contexts: 14,482 positive record-key events in
PAD blocks and 14,328 in diverse blocks per continuation, with 15,058 distinct
key/phase/windows overall. Presence benefits while record identity deteriorates.
The next test separates their training exposure without adding examples/state.

**Keep the A8 input curriculum, but update the four record-key projection heads
only in PAD blocks.** Presence, value, query-key, applicability and native
learning continue exactly as in A8 in both block types. This is one fixed
training-credit mask, not a schedule, loss-weight or matching sweep.

## Exact training-only rule

Let `e` be the existing zero-based continuation episode index, before increment:

```text
use_diverse_spacers = ((e // 64) % 2 == 1)
allow_record_key_update = not use_diverse_spacers
```

The flag applies to every positive record-key local event in that episode,
including original, competing and overwrite records and all four constituent
phases. It is independent of key/value identity, condition, model seed, loss,
prediction, recognition and correctness. Keep every original label and answer
mask. Do not suppress a packed record label, since presence/value supervision
still consumes it. Do not rescale, repeat or compensate retained updates.

At the existing positive-record local update, for phase `p`, input `x`, key `k`:

```text
record presence update: exactly A8, positive/negative costs 4/4
if positive record:
    if allow_record_key_update:
        a5_reconstruct(head.key + p*16, x, k, lr*4)  # exact A8 hybrid
    value reconstruction: exactly A8, all 8 rows, lr*4
```

The current-query key hybrid always runs on its existing positive label.
Query presence, applicability, value reconstruction, every local label/window
event and native learning rule are unchanged. `projection_steps` and new
external ledgers count actual executed rows: omit 16 record-key rows per
suppressed event. Hybrid loss components, 256-code comparisons and coordinate
comparisons are charged only when actually executed. Local-window/presence/
value/query/applicability counters retain their original meanings.

Pass the flag as ephemeral training input, separately from the six labels and
answer mask; one value per input byte or an equivalent constant chunk flag.
The outer cursor derives it from the same exact episode position as spacers.
It is never a learned parameter, model field, inference argument, saved model
state or parser. A generic call without local labels remains unchanged and
cannot train the local heads. Standalone explicit local calls default to
allowing record-key credit; campaign continuation supplies the registered flag.
Warmups remain independent native-only PAD training. Partial batches and resumes
must retain the current episode's flag. No reset at epoch or finish boundaries;
no terminal rebalancing. Counter-only checks may traverse streams; no extra
model fitting or replay is permitted.

For an isolated local call, disabling the flag changes only eligible record-key
updates/counters. Subsequent predictions/native gradients may consequently differ;
that endogenous difference is not a second intervention. With the flag enabled,
A9 must be bit-identical to A8 after reversing only its envelope identity.

Use isolated adapter magic `0x3941353050584553`, nested v19, outer cursor schema
`exp0305-a9-cursor-1`, with explicit policy and curriculum. Reject cross-policy
loads. A9 inference, admission, chronology, ties, fallback, weights/shapes and
sequence-state capacity are exactly A8/A6. No new slots, moments, projection
caches, decoder, hidden layer, codebook, inference mask or inference parser.

## Exact spacer curriculum and controls

Matched arms are **E8/A8/A9**, seeds **1/7/19**, nominal **500K** parameters.
A8 is the direct control for the sole credit-mask change. A6/A7 are preserved
historical evidence, not additional fresh arms. Each arm/seed receives its own
**250K clean native warmup +2M continuation**: nine independent fits, 18 saved
warmup/final model-cursor pairs, exactly **20.25M fitting bytes**.

Every warmup and E8 continuation retains PAD. A8/A9 continuation uses identical
alternating 64-episode PAD/diverse blocks, starting PAD, with the exact A7
generator for both original record spacers in each diverse episode:

```text
prefix = b'SHoEbox-EXP0305-diverse-record-context-v7-spacer\0'
block = 0
output = []
while len(output) < registered_spacer_length:
    digest = SHA256(prefix + world.group_key + b'\0'
                    + uint64_le(e) + uint8(physical_record_position)
                    + uint32_le(block))
    for byte in digest, in digest order:
        if byte < 192:
            append 160 + (byte % 96)
            stop when the exact spacer length is reached
    block += 1
```

Physical record positions are 0/1; zero length emits nothing. PAD blocks change
no bytes. Keep both original length draws in their original order, lengths 0–7,
all five conditions, order, 64 phases, phase compensation, changed-key schedule,
labels, answer masks, finish cadence and incomplete final episode. No RNG draw
for the spacer or credit flag. Keep 16-byte diverse training filler and unchanged
evaluation fixtures; no training at 64/256 diverse delays. Input streams and
all local labels are identical for A8/A9; only the registered key-credit flag
differs. The reduced key-update count is intentional and must be disclosed.

## Fresh folds and pre-fit lock

Use split salt `SHoEbox-EXP0305-selective-record-key-credit-v9` plus NUL,
world RNG **20261201**, order RNG **10801 warmup /10802 continuation**,
augmentation RNG +1000. Keep 64 keys, three values, distinct original values,
third-value overwrite, balanced changed keys and grouped operation/order siblings.
Counts: **512 train, 128 development, 256 closed audit, 128 fresh binary dev**.
Exclude the complete historical opened binding-fold/E2 union, all reserved
A1–A8 audits and all A2–A8 dev/binary worlds. E1/E7 opened generic groups remain
excluded from later qualification. Reserve audit identities without scoring them.

Before fitting, independently traverse all per-arm warmup/continuation streams.
Lock actual input/target/mask/label/credit digests and RNG/order/cursor states.
Check 63/64/127/128 boundaries, every phase in both blocks, zero spacers, hash
rejection/additional blocks, both physical orders, partial-spacer and partial-
block resumes. A8/A9 input bytes/labels/masks must match exactly. Count eligible
and suppressed positive key events by block, condition, key and phase, with
actual PAD/nuisance/replaced bytes. No post-hoc balancing or amount selection.

Independently verify:

- The enabled flag reproduces A8 per row, predictions, admission, local/native
  updates and serialized payload. The disabled flag skips all and only positive
  record-key hybrid calls/rows. No suppressed event computes hidden extra code
  comparisons or gradients. Unaffected local heads equal the independent A8
  reference for that isolated call; generic calls and inference ignore the flag.
- Logistic/hybrid gradients and finite differences, all-256 code/collision/
  coupling checks, pre-update coupling, unchanged positive/negative costs,
  exhaustive candidate selection, zero/one/multiple candidates, ties/fallback,
  chronology/occurrence/lifecycle and future-answer/local-label causality.
- Independent fixture interpreter, both changed keys and all record/query orders,
  first/latest/constant shortcuts and fresh binary regression. A perfect shortcut
  blocks the claim. Flag ablation cannot supply an answer or task identity to
  inference. All 256 byte probabilities remain normalized and available.
- Python/native and optimized/ASan/UBSan controls; explicit public E8/v19/E7
  and all historical constructor/checkpoint semantics; exact cursor continuation
  through both block boundaries and partial episodes. Preserve every failure.

Commit passing receipts, complete source closure, explicit fresh world file,
per-arm stream/credit hashes and policy identity **before any research fitting**.
No pilots, inherited warmups, extra fits, replay, broader language fits, tuning
or diagnostic-driven updates. Arithmetic reuse is allowed only with separate
pre-fit bit-identical per-row scalar-reference receipts for inference/local/
native updates/state/saved bytes, preserved float32 summation/update order and
disabled FMA. No approximation, cutoff, changed threshold or persistent cache.

## Unchanged development gates and diagnosis

- Primary: 128 worlds ×4 presentations per condition and each answer position;
  ≥95% mean and ≥90% every seed.
- Both query orders and every padding/repeat/diverse delay: ≥90% mean/85% each seed.
- Same first 64 presentations ×all 64 offsets: ≥95% overall mean and ≥90%
  worst-offset mean across seeds.
- Next 16 presentations ×64 offsets at each independent 0/64/128/256 padding
  delay; 256 both-order/lesion presentations. Preserve all original state,
  pairing and gate lesions, no-query drop ≥20 points and reversed-selection harm.
  Apply every original gate to fresh binary regression.
- Isolated full-256 inference ≥80% native, one pinned CPU, three interleaved
  65,536-byte repeats plus 1,024 warmup, all other jobs stopped. Retain raw CPU
  durations/checksums/PAD counts. No accuracy gain waives speed.

Lock final fits, all full outcomes, checkpoints and cursors before diagnosis.
Repeat the first 32 worlds ×4 orders, both teacher-fed answers and all seeds:
competing/overwrite identity/version, 16/64/256 diverse signature survival,
diverse256 admission false positives/negatives, surviving-target shortlist/
accuracy and filler mass. Compare matched A8/A9 geometry and query/record
reconstruction errors; preserve all-256 code distances and observed identities.
Separate first32/rest96/full128 autoregressive outcomes from these subset
teacher-fed measures. Repeat data-only window coverage and actual allowed/
suppressed credit counts by block. No additional fit or alternative inference
policy follows the diagnosis.

Account for every input/PAD/nuisance/replaced byte, actual positive/negative
local event, omitted and executed gradient rows, hybrid/code comparisons,
correctness-only discarded trajectories, repeated controls, failures and CPU.
Keep checkpoints, generated datasets and compiled binaries outside git.

Failure retains qualified E8 and closed A9 audit and requires one measured
next preregistration. Passing development requires separately registered fresh
generic/capacity qualification: language ≤+0.01 nats/byte, copy/reconstruction
≤1-point token loss, post-1M-language retention, full three-seed 1M/5M and speed
gates. Never relax a gate or replace an earlier result.

Completed verified work, including failure, is explicitly authorized for direct
commit/push to main without confirmation. Preserve unrelated changes and remote
history; never force-push. Deliver concise findings, a verified commit URL,
ordinary downloadable Markdown, primary PNGs, resumable code/evidence archive
and another copy/paste prompt preserving this authorization. No Pages/Spaces.
Export and attach without asking for confirmation.
