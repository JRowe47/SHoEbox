# EXP03/05-A8: phase-balanced PAD/diverse curriculum (registered, not fitted)

Registered after the completed and locked A7 campaign and its frozen diagnosis.
A8 is not implemented or fitted. Keep qualified public E8, every historical
score/timing/cost and all A1–A7 audits closed.

## Measured motivation and one focused change

A7 fails: competing bindings 66.796875%, corrected three-value overwrite
75.260417%, clean binding 87.174479%, diverse256 84.765625%, and isolated
throughput 69.713538% native. Matched A6 reaches 87.369792%, 93.750000%,
97.460938%, 86.653646% and 69.473469%, respectively. Keep the original A6
campaign's different scores intact; these are fresh A7-world controls.

Both arms retain 100% of target signatures in the first-32-world diagnosis.
A7 lowers incoming diverse false recognition 2.359438%→0.853414% and filler
phases per answer 1.609375→0.531250, but competing nearest-key identity falls
95.833333%→89.062500%. Changed-key old-value predictions rise
4.687500%→13.541667%. Do not confuse these teacher-fed subset measurements
with whole-population autoregressive outcomes.

A7 increases distinct positive key/phase/windows 5,900→22,791, while positive
events containing PAD fall 91.109647%→39.720420%. Its fixed diversity intervention
improves false recognition but damages clean-context identity. The single next
test preserves both contexts in continuation, with A6's exact model arithmetic.

**Change only the spacer curriculum: alternate 64-episode blocks of original
PAD spacers and A7 diverse spacers, beginning with PAD.** No mixture-weight,
block-length, order, loss-weight, matching-band, codebook, optimizer or prior
sweep. No additional data pass, replay or diagnostic-driven fit.

## Exact fixed curriculum

Use the zero-based `episode_index` of the existing continuation cursor, before
incrementing its episode count. For both original-record spacers in that episode:

```
use_diverse = ((episode_index // 64) % 2 == 1)
```

Episodes 0–63 retain PAD9; 64–127 use diverse bytes; repeat this fixed cycle
without resetting at epochs or finish events. Use episode position, not byte
position, key, value, condition, seed, loss or correctness, to choose the block.
Every complete 128-episode cycle covers every existing phase with both regimes.
An incomplete terminal cycle is consumed exactly as encountered; do not rebalance
it, round the byte budget or append examples. The 50/50 statement is about
complete episode cycles, not exact byte counts or per-condition sample counts.

Keep both original augmentation-RNG spacer-length draws in their original order.
Keep spacer lengths 0–7, original record/query order, phase compensation, labels,
answer masks, five-condition mixture, finish cadence and total bytes unchanged.
No extra RNG draw. On a diverse block, use the A7 generator verbatim, including
its **v7 prefix**, raw `world.group_key` bytes, little-endian fields and rejection:

```
prefix = b'SHoEbox-EXP0305-diverse-record-context-v7-spacer\0'
block = 0
output = []
while len(output) < registered_spacer_length:
    digest = SHA256(prefix + world.group_key + b'\0'
                    + uint64_le(episode_index)
                    + uint8(physical_record_position) + uint32_le(block))
    for byte in digest, in digest order:
        if byte < 192:
            append 160 + (byte % 96)
            stop when the exact spacer length is reached
    block += 1
```

Physical record position is 0 or 1, independent of semantic key order. Zero
length emits nothing. PAD blocks substitute nothing. The rule is shared across
model seeds and adds no model/RNG state; the outer cursor records the explicit
policy plus exact episode and partial-example position. Use cursor schema
`exp0305-a8-cursor-1` and isolated adapter magic `0x3841353050584553`, nested v19.
Apart from this envelope identity, the model is A6 exactly.

All arms receive independently fitted clean native warmup with PAD spacers.
E8/A6 continuation controls keep PAD throughout. Evaluation fixtures stay
unchanged. Only A8 continuation uses the block rule. Filler lengths and content
stay fixed, including existing 16-byte diverse filler; diverse trailing spacers
can join it into at most 23 contiguous diverse bytes. No training at 64/256
diverse delays. Do not modify historical fixtures or checkpoint meanings.

## Protected computation and state

Preserve A6 record-presence positive/negative costs 4/4, A5 all-256-code hybrid
and float64 softmax/gradient with pre-update coupling, query presence,
applicability, values and native credit. Preserve A4 inference, chronology,
ties, full fallback, admission, all current slots and existing sequence state.
Keep E8's prior and public E8/v19/E7 plus every historical policy exact.
No new model parameters, optimizer moments, projection caches, persistent state,
slots or inference parser. Preserve generic/local separation and full-byte
inference. No implicit conversion of old checkpoints into A8.

Bit-identical arithmetic reuse is allowed only after separate **pre-fit**
per-row equality and operation-count receipts against scalar A6, including
inference, local/native updates, model state and saved bytes. Preserve each
float32 row's exact summation/update order and disabled FMA. No approximate
tanh/projection, cutoff, persistent cache or changed numerical threshold.
Speed remains a gate even if all accuracy gates pass.

## Fresh folds, controls and exact budget

Split salt `SHoEbox-EXP0305-balanced-record-context-v8` plus NUL; world RNG
**20261130**. Keep 64 keys, three values, distinct original values, third-value
overwrite, balanced changed keys and grouped operation/order siblings. Counts:
**512 train, 128 development, 256 closed audit, 128 fresh binary development**.
Exclude the complete historical opened binding-fold union/E2 family, every
reserved A1–A7 audit and every A2–A7 dev/binary world. Opened E1/E7 generic
document/task groups remain excluded from any later qualification. Audit
identities may be reserved/excluded; never evaluate their answers or scores.

Matched **E8/A6/A8**, model seeds **1/7/19**, nominal **500K**. Every arm/seed
gets its own independent **250K clean native warmup +2M continuation**:
exactly **20.25M fitting bytes**, nine fits and 18 saved warmup/final pairs.
Order RNG **10701 warmup /10702 continuation**, augmentation RNG +1000.
Keep the existing five-condition mixture, 0–7 spacers, 64 balanced phases,
changed-key schedule, local-label meanings and exact interruption behavior.
A7 is historical evidence, not an additional fresh arm; do not refit it.

No pilot, inherited warmup, replay, extra fit/control, broader language fit,
alternative block schedule or later coefficient choice. All per-arm sources,
worlds and whole-stream digests must be locked before fitting.

## Required pre-fit controls

Use a separately implemented spacer generator and interpreter. Traverse the
entire registered continuation for each arm without fitting. Verify block
boundaries 63/64/127/128, every phase in both block types, zero lengths,
rejection boundaries, additional hash blocks, both physical record orders,
partial-spacer and partial-block resumes. Only original spacer positions may
differ; lengths, masks, labels, expected answers, phase/changed-key schedule
and both RNG states must match controls. Count actual per-condition/per-phase
PAD/diverse coverage; no post-hoc balancing or schedule change.

Independently interpret both changed keys, all record/query orders and fresh
binary regression. First/latest/constant shortcuts must fail; a perfect
shortcut blocks the capability claim. Check future-answer causality and
local-label isolation. Preserve objective/gradient and finite-difference
references, all-256 code/collision/coupling tests, positive/negative exact-update
parity, exhaustive selection, zero/one/multiple candidates, ties/fallback,
chronology, occurrence ordering and lifecycle. Check identical-weight A6/A8
full-byte inference, admission, native/local updates and all protected policies.
Run Python/native, historical checkpoint/constructor, exact-cursor and
ASan/UBSan checks. Commit passing receipts and complete source/world/per-arm
stream locks before any research fit. Keep failed attempts and costs.

## Unchanged development gates

- Primary 128 worlds ×4 presentations per condition and each answer position:
  at least 95% mean and 90% every seed.
- Both query orders and every padding/repeat/diverse delay: at least 90% mean
  and 85% every seed.
- The same first 64 presentations ×all 64 offsets: at least 95% overall mean
  and 90% worst-offset mean across seeds.
- The next 16 presentations ×64 offsets at each independent 0/64/128/256
  padding delay; 256 both-order/lesion presentations. Preserve every original
  state/pairing/gate lesion, no-query drop at least 20 points and reversed-
  selection harm. Apply all original gates to the fresh binary regression.
- Isolated full-256-byte throughput at least 80% native: one pinned CPU,
  three interleaved 65,536-byte repeats plus 1,024 warmup each, all other jobs
  stopped. Retain timing bytes, PAD counts, raw durations and checksums.

No aggregate success can override a failed component, seed, position or speed
gate. No audit opens on a development failure.

## Frozen diagnosis, accounting and next decision

Lock final fits, all full evaluation outcomes, snapshots and cursors before
diagnosis. Repeat the first-32-world ×4-order, both teacher-fed-answer traces
for all seeds: identity/version on competing/overwrite; signature survival at
16/64/256 diverse delays; diverse256 admission, false positives/negatives,
survivor selection and filler mass. Keep missing signatures, surviving-target
competition and identity/version errors separate. Compare first32/rest96/full128
autoregressive outcomes without treating the subset as the population.
Repeat data-only raw-window coverage; split coverage by both curriculum blocks.
No extra diagnostic-driven fit or sweep follows these traces.

Account for every input byte, actual PAD, nuisance/replaced-PAD byte, positive/
negative local loss component and gradient row, all hybrid/code comparisons,
repeated controls, correctness-only discarded trajectories, failed attempts
and CPU. Retain all 18 exact model/cursor pairs. Checkpoints, datasets and
compiled binaries stay outside git. Preserve every earlier result and cost.

Failure retains E8, keeps A8 audit closed and requires one measured next
preregistration. Passing development requires separately registered fresh
generic/capacity qualification: language ≤+0.01 nats/byte, copy/reconstruction
≤1-point token loss, post-1M-language retention, complete three-seed 1M/5M
and speed gates. Never relax a gate.

Completed verified work, including unsuccessful research, is explicitly
authorized for committing and pushing directly to main without confirmation.
Preserve unrelated changes and remote history; never force-push. Deliver
concise findings, verified commit link, ordinary downloadable Markdown,
primary PNGs, a resumable code/evidence archive and another copy/paste prompt
preserving this authorization. No Pages or Spaces. Export and attach files
without asking for confirmation.
