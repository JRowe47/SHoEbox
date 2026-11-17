# EXP03/05-A7: diverse record-context curriculum (registered, not fitted)

Registered after the locked A6 campaign and separate identity/admission/survivor
traces. A7 is not implemented or fitted. Keep qualified public E8 and A1–A6
audits closed. Preserve every original score, timing, failed attempt and cost.

## Measured failure and one focused revision

A6 raises matched diverse256 autoregressive exactness from 63.151042% to
84.375000%, but competing bindings stay 87.890625%, corrected overwrite stays
94.140625%, and isolated throughput is 67.583762% native (80% required).
Clean binding is 95.312500%. Full development fails.

On the registered first-32-world trace, original-signature survival improves
93.75%→100%; target removals drop 48→0 across seeds. Fully diverse incoming
false recognition falls 15.913655%→3.112450%; true-record phase false negatives
are zero in both arms. All surviving A6 targets are recognized/shortlisted,
and teacher-fed surviving-target accuracy is 96.875%. Recognized filler phases
fall from 11.15625 to 1.9375 per answer. This repairs the observed cache loss;
no extra slots or admission revision is justified by this trace.

Competing nearest-key identity stays 96.875% in A5 and A6. A6 query reconstruction
squared error averages 0.158972 versus 3.288515 for recognized record keys.
Same-key old/new squared distances span roughly 0.324–10.223 and overlap
different-key distances. The registered trace has zero changed-key old-value
predictions, but the full primary gate still fails; do not turn a subset result
into a complete version-selection claim. Diverse256 autoregressive exactness
is 93.75% on the first 32 worlds and 81.25% on the remaining 96, explaining why
the small teacher-fed trace is not representative of the complete endpoint.

Data-only traversal of the exact 2M continuation finds 28,856 positive record-key
events, of which 26,255 (90.9863%) contain padding in the eight-byte source window.
There are 5,997 distinct positive key/phase/window combinations. The 9,963 fully
diverse negative-window events cover only 3,252 distinct raw windows. This motivates
one controlled test of record-context diversity; it does not establish that
coverage alone causes the remaining errors.

**Change only continuation training's existing 0–7-byte spacers after the two
original records from PAD bytes to per-occurrence diverse bytes in [160,255].**
Keep their lengths, byte positions, record/query order, phase compensation,
labels, answer mask and total budget unchanged. Keep A6's complete model,
record positive/negative costs 4/4, A5 hybrid key objective, query presence,
applicability, values, native credit, A4 inference/chronology/ties/fallback,
admission, existing state and E8 prior. No matching-band, codebook, optimizer,
negative-weight, curriculum-mixture or E8-prior sweep.

The intervention trains the existing local key/value/presence heads against
varied surrounding bytes. It adds no model parameter, persistent sequence
state, optimizer moment, projection cache, slot or inference parser. The
training helper remains external. No nuisance byte contains a fixture marker,
value or key; inference still supports all 256 bytes. Filler conditions and
lengths stay fixed, including the existing 16-byte diverse condition. A trailing
spacer may join that filler into at most 23 contiguous diverse bytes; unseen
64/256-byte delay gates remain unchanged.

## Exact deterministic spacer rule

For A7 continuation only, retain original order RNG, augmentation RNG and both
spacer-length draws. Generate spacer content with SHA-256; do not consume either
existing RNG or alter a later length/order/phase. E8/A6 controls keep PAD spacers.
Warmup is independently fitted clean PAD-spacer native warmup for every arm.

Let `episode_index` be the zero-based number of episodes selected by this
continuation cursor before incrementing `episodes`. Let `record_position` be
0 or 1 in physical presentation order, not semantic key order. For each spacer:

```
prefix = b'SHoEbox-EXP0305-diverse-record-context-v7-spacer\0'
block = 0
output = []
while len(output) < registered_spacer_length:
    digest = SHA256(prefix + world.group_key + b'\0'
                    + uint64_le(episode_index)
                    + uint8(record_position) + uint32_le(block))
    for byte in digest, in digest order:
        if byte < 192:
            append 160 + (byte % 96)
            stop when the exact spacer length is reached
    block += 1
```

`world.group_key` is the existing exact bytes value, not a reformatted JSON
encoding. Zero length emits no bytes. The rejection mapping gives the existing
96-byte filler alphabet equal support. The rule is shared across model seeds;
no label, answer or audit value chooses a nuisance byte. It is pure data
augmentation, computed from the existing cursor/world. No additional model or
RNG state is required. The outer cursor records its A7 curriculum policy and
keeps the exact episode counter; use schema `exp0305-a7-cursor-1`.

Give the isolated research adapter outer magic `0x3741353050584553`, nested v19.
Apart from this explicit envelope identity its model arithmetic is A6 exactly.
Protected E8/v19/E7 and every older constructor/checkpoint keep their meanings;
reject implicit conversions. Never modify historical fixture helpers in place.

## Fresh folds, controls and exact budget

Salt `SHoEbox-EXP0305-diverse-record-context-v7` plus NUL; world RNG 20261129.
Keep 64 keys, three values, two distinct original values, third-value overwrite,
balanced changed keys and grouped operation/order siblings. Counts remain
512/128/256 train/dev/closed-audit worlds plus 128 fresh binary dev worlds.
Exclude the complete historical opened binding-fold union/E2-family, reserved
A1–A6 audits, and all A2–A6 development and binary-regression worlds. Opened
E1/E7 generic document/task groups remain excluded from later qualification.

Matched **E8/A6/A7**, seeds **1/7/19**, nominal 500K. Each arm/seed independently
receives 250K clean native warmup +2M continuation, exactly **20.25M bytes**.
Order RNG **9701/9702**, augmentation +1000. Preserve five-condition mixture,
spacer lengths 0–7, 64 balanced phases, changed-key schedule, local-label meanings,
finish cadence and exact interruption. E8 and A6 use original PAD spacers; A7
uses only the fixed substitution above. This is a matched-budget curriculum
comparison, not a claim that all arm input bytes are identical.

No pilot, inherited warmup, replay, extra fit/control, broader language fit or
alternative spacer distribution. Evaluation remains the unchanged original
fixtures on the newly locked worlds; do not replace test spacers or report
augmented training performance as qualification.

## Before fitting

Lock complete sources, worlds and per-arm generated-stream digests first.
Independent reference must reproduce every substituted byte, including rejection
mapping, zero lengths, both physical record orders and partial-spacer resume.
Compare control/candidate episodes: only original spacer positions may differ;
length, masks, expected answers, phase, order and every causal local label must
agree. Keep operation siblings grouped. Independently interpret both changed
keys/all orders and fresh binary regression; first/latest/constant shortcuts
must fail. A perfect shortcut blocks a capability claim.

Verify unchanged A6 source except magic, identical-weight full-byte inference,
admission, local/native updates, generic/local separation, all 256 codes,
float64 hybrid/pre-update coupling, record cost 4/4, query-presence/applicability,
values and native credit. Retain independent logistic/hybrid gradients, finite
differences, positive-update parity, exhaustive selection, zero/one/multiple
candidates, complete fallback/ties, chronology, occurrence-order, lifecycle,
protected-policy, historical and exact-cursor tests, plus ASan/UBSan.

Exact arithmetic reuse across independent projection rows is permitted only
with a separate pre-fit bit-identical equality/cost receipt against scalar A6,
preserving every float32 per-row summation/update order and disabled FMA. It
cannot change the mathematical policy or excuse the speed gate. No approximate
tanh/projection, heuristic cutoff, persistent cache or changed threshold.
Lock passing receipts and commit the pre-fit state before any research fit.

## Unchanged gates and diagnosis

- Primary 128 worlds ×4 orders per condition and each answer position:
  >=95% mean /90% every seed.
- Both-order and every padding/repeat/diverse delay: >=90% mean /85% every seed.
- Same first 64 presentations ×64 offsets: >=95% overall mean, >=90% worst-offset
  mean across seeds.
- Next 16 presentations ×64 offsets at each independent 0/64/128/256 padding
  delay; 256 both-order/lesion presentations. Keep every state/pairing/gate lesion,
  no-query drop >=20 points and reversed-selection harm. Apply all original
  gates to fresh binary canonical regression.
- Isolated full-distribution throughput >=80% native: one pinned CPU, three
  interleaved 65,536-byte repeats plus 1,024 warmup each, all other jobs stopped.

Lock final fits/outcomes/snapshots/cursors before diagnosis. Repeat the same
32-world identity/version, 16/64/256 signature, diverse256 admission and
survivor/filler-mass traces; measure false negatives and false positives. Retain
full-primary versus first-32-world outcomes separately. Repeat data-only raw-window
coverage accounting. No new diagnostic-driven fit or coefficient selection.

Charge all bytes including replaced padding, actual PAD counts, nuisance bytes,
positive/negative local loss components/gradient rows, hybrid/code comparisons,
repeated controls, correctness-only trajectories, failures and CPU. Keep all 18
warmup/final pairs and exact outer cursors. Checkpoints/data/binaries stay outside git.

Failure retains E8, keeps A7 audit closed, and triggers one measured next
preregistration. Passing development requires separately registered fresh
qualification: language <=+0.01 nats/byte, copy/reconstruction <=1-point token
loss, post-1M-language retention, complete three-seed 1M/5M and speed gates.
Partial memory, overwrite or throughput improvement cannot override another gate.

Completed verified work, including unsuccessful research, is explicitly
authorized for commit/push directly to main without another confirmation.
Preserve unrelated changes/history; never force-push. Deliver concise findings,
verified commit link, ordinary downloadable Markdown, primary PNGs, resumable
code/evidence archive and another copy/paste prompt preserving this authority.
No Pages or Spaces. Export and attach files without asking for confirmation.

## Grounding

The causal comparison is motivated by the measured A6 input coverage and frozen
key dispersion. Paired addressing remains grounded in [Miller et al., EMNLP
2016](https://aclanthology.org/D16-1147/), and multi-query recall measurement in
[Arora et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html).
Neither paper establishes the effect of this project's fixed spacer curriculum.
