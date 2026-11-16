# EXP03/EXP05-A1: learned replacement priority, existing 16 slots

Registered after the fixed, corrected frozen diagnosis and before any A1 fit.
Starting main: 9e34d8075abbc495f966ab37d18ecfa532bb854d.

## Evidence and one intervention

On 64 fresh old-vocabulary development worlds (256 presentations), E8 gives
100% padding exactness, 15.625% at 64 diverse bytes and 7.8125% at 256. Required
L1 signatures are absent at both longer diverse delays. The externally timed
cache-freeze diagnostic restores 93.3594% and 92.1875%, with 100% signature
presence. At two competing writes, signatures survive but top-two inclusion
is only 61.7188% and exactness 47.6563%; overwrite exactness is 22.6563%.
These are separate eviction and read-selection failures.

A1 changes ONLY full-cache replacement in L1 (eight-byte retained pairs).
Use the existing learned four phase-presence logits on existing stable vectors.
At full capacity, choose the oldest unit whose maximum logit is <=0. If every
unit is positively recognized and the incoming pair is not, reject its cache
admission. If every unit and the new pair are recognized, evict the oldest as
before. Matching-signature refresh, duplicate suppression, clock rules on
actual admission, pair frontier and last-window updates remain unchanged.
Preserve current top-two record routing, key/value decoding, latest-query rule,
E8 prior, all local/native rates and objectives. Zero added fitted parameters,
slots, persistent metadata or optimizer moments. Four classifier dot products
per inspected old unit are charged; no parser or task symbols enter inference.

A1 has a distinct outer magic 0x3141353050584553. No existing checkpoint is
silently converted. Public E8 and all historical sources remain unchanged.
The mechanism is learned fixture-local admission, not generic learned memory
management. The delta-rule literature (Schlag et al., ICML 2021,
https://proceedings.mlr.press/v139/schlag21a.html) motivates selective updates;
A1 itself is our bounded-cache replacement experiment, not that paper's model.

## Locked folds, controls and budgets

Use tasks.py's fresh salt, 512 train /128 dev /256 audit initial-state groups,
four order siblings each. Keys 0..63; values 0/1. Previously opened binding
test folds on the historical 48-key vocabulary are excluded in full. Counterfactual
updates, distractors, spacers, phases and orders remain in the same base-world
fold. No generic documents or old generic-task fixtures are used in this screen;
all opened E1/E7 document/task groups remain excluded from later qualification.

Three arms, seeds 1/7/19: unchanged E8, A1, native v19. Nominal 500K each;
readout weights are deducted for E8/A1. Each arm gets a separately charged
250K native warmup (clean binding, answer mask), then 2M every-byte continuation.
Continuation mixes pad0, repeat16, diverse16, compete16, overwrite16 uniformly
through shuffled complete epochs. Local E8 labels supplied to E8/A1 only;
native has no local heads. Independent 0–7 record spacers and balanced 64 phases.
Order RNG 3701/3702, augmentation RNG +1000. All arms use identical data order.
Local extraction/applicability remain explicit prefix supervision, never target
answers or selected-record labels. No extra optimizer or prior candidates.

Total: 20.25M charged fitting bytes, all nine warmups and repeated controls
included. No preliminary fit or pilot. Each phase ends at its fixed budget,
including partial episodes. Finish at warmup 250K and each continuation 500K;
checkpoint outer cursors preserve RNGs, order, current world/condition/variant,
spacers, byte position, finish schedule, counts and rolling stream hash.

## Endpoints and unchanged thresholds

Primary development includes pad0, repeat16, diverse16, compete16, overwrite16,
512 presentations each, full-256-byte autoregressive exactness. Each condition
requires mean >=95% /each seed >=90%, and the same per answer position.
Both-query-order exactness >=90% mean /85% each. All 64 offsets on the same
first 64 presentations require >=95% mean /90% worst offset seed mean.
Independent padding delays 0/64/128/256 retain >=90% mean /85% each. Apply the
same retention gate to repeated/diverse filler at 64 and 256; these are 4x/16x
the trained filler length, reported independently. Query ablation drop >=20
points; reversed selection strictly hurts. Keep no-head/no-state/uniform/reverse/
unpair/gate-off/on interventions. Report overwritten and untouched keys separately.

After every arm finishes, lock all checkpoints and development results. If any
primary condition fails, leave the fresh audit closed, retain E8 and register
the next focused revision. Run extended diagnostic gates even on a failed
candidate where they explain the failure. A1's causal test is longer-filler
signature survival and matched native exactness, not just slot occupancy.

Promotion remains gated by ALL prior requirements, including fresh language
<=+0.01 nats/byte, copy/reconstruction token loss <=1 point, retained binding
through 1M generic targets, three-seed 1M/5M and isolated >=80% native full-byte
throughput. Those expensive qualifications require passing this screen and a
separate fresh-document registration; this file does not authorize reusing any
opened E7 generic groups. No gate is weakened by reporting a partial repair.

Run Python/native causality, lifecycle, historical checkpoint, interrupted
cursor and ASan/UBSan checks. Throughput: one pinned CPU, three interleaved
65,536-byte repeats plus 1,024 warmup, each full 256-byte output, all other jobs
stopped. Measure every new arm; compile and clone excluded from timed windows,
included in whole-process CPU. Record every input/padding/answer/objective byte,
local projection/presence/gate/window event, fit/evaluation CPU and failures.
