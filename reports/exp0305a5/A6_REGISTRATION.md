# EXP03/05-A6: fixed record-presence false-positive cost (registered, not fitted)

Registered after locked A5 fits and frozen identity/admission/survivor traces.
A6 is not implemented or fitted. Keep qualified public E8 and all A1–A5 audits
closed. Preserve all earlier scores, costs, throughput and checkpoint meanings.

## Measured failure and one focused change

A5 improves fresh competing-binding exactness 88.671875% →90.234375% versus
matched A4; corrected overwrite stays 95.703125%, clean binding is 97.591146%,
and diverse256 is 63.802083%. It fails the original development gates.
On the matched 32-world competing trace, nearest-key identity is 97.65625%
for **both** A4 and A5. A5 reduces unrelated selected mass, but does not improve
that identity score. Record-key reconstruction remains much noisier than query
reconstruction; residual identity/version errors remain separately reported.

The separate long-diverse trace isolates a strong presence failure shared by
A4/A5: only 95.3125% of original signatures survive. Every one of 36 A5 removal
events occurs under all-slots-recognized pressure. Fully diverse incoming pairs
are falsely recognized 1336/7968 =16.767068% per seed. At diverse256 answer time,
there are 11.5625 positively recognized unannotated record-phase candidates per
answer on average. All surviving targets are recognized, but only 86.065574%
reach the shortlist. On wrong answers with a surviving target, roughly 95% of
selected mixture mass belongs to unannotated filler. This separates false
admission/readout competition from actual target loss and key reconstruction.

A6 changes only the existing **four record-presence classifier losses** so
false positives receive the same fixed cost as the existing positive events.
The fixed factor is four: the number of record-phase tests per window and the
already established positive weight. This is one preregistered objective, not
a search over weighting factors, thresholds, optimizers or priors.

## Exact numerical rule

For each existing local record-phase event (`bytes>=8 && bytes%4==0`), preserve
`x=source_vector(levels[1].retained.last)`, the existing phase row, and label
`y=(labels[phase+1]>=0)`. Compute in the existing float32 arithmetic:

```
z = affine(head.gate[phase], x)
p = sigmoid(z)
error = (p - y) * 4.f
rate = rate * auxiliary
```

Call the unchanged `affine_step` with this error/rate and its original
`1+energy(x)` normalization. The exact objective is four times binary logistic
loss for each of the four rows. Positive updates are bit-identical to A5;
negative updates change their multiplier from `1.f` to `4.f`. Keep the original
phase order and per-row float32 summation/update order; FMA remains disabled.
Count positive and negative events/losses/gradient rows separately. For reporting
stable losses, use softplus; do not change the actual registered update arithmetic.

**Everything else stays A5:** fifth/query presence classifier, applicability,
A5's exact all-256 fixed-code hybrid key objective and float64 pre-update
coupling, value reconstruction, native responsibility credit, original rates,
full-256-byte autoregression, A4 chronology/band/ties/fallback/paired values,
admission algorithm, 16 slots, E8 cold applicability prior and existing state.
The admission algorithm uses the newly learned presence scores naturally;
there is no new threshold, parser, hard key equality, label tape or oracle.

No extra parameters, persistent sequence state, optimizer moments, projection
cache or slots. A6 gets outer magic `0x3641353050584553`, nested v19, with explicit
cross-policy rejection. Protected public/historical constructors and loaders
retain their saved meanings. Generic byte objectives never supply local labels.
No E8 prior, matching-band, codebook, optimizer or negative-weight sweep.

Exact reuse/vectorization across independent projection rows is permissible
only with pre-fit bit-identical equality against scalar A5 arithmetic and
preserved float32 per-row summation order, with FMA disabled. No approximate
projection/tanh, heuristic early exit or throughput relaxation. Any such
implementation work requires its own before-fit equality and cost receipt.

## Fresh folds, controls and exact budget

Salt `SHoEbox-EXP0305-record-presence-cost-v6` plus NUL; world RNG 20261128.
Keep 64 keys, three distinct-value possibilities, third-value overwrite,
balanced changed keys and grouped operation/order variants. Counts remain
512/128/256 train/dev/closed-audit worlds plus 128 fresh binary dev worlds.
Exclude the entire historical opened binding-fold union including E2-family,
reserved A1/A2/A3/A4/**A5** audits, and **all A2/A3/A4/A5 dev and binary worlds**.
Opened E1/E7 generic document/task groups remain excluded from any qualification.
Pin every world and full source/fixture/objective dependency closure first.

Matched **E8/A5/A6**, seeds **1/7/19**, nominal 500K. Each arm/seed independently
receives charged 250K clean native warmup plus 2M mixed continuation, exactly
**20.25M bytes**. No pilot, inherited warmup, additional control/fit or replay.
Order RNG 8701/8702; augmentation +1000. Preserve the five-condition mixture,
0–7 spacers, 64 balanced phases, changed-key schedule, local label meanings,
finish cadence and exact interrupted outer cursor. Keep every warmup/final pair.

## Pre-fit controls and unchanged gates

Independent logistic loss/gradient references and finite differences;
bit-identical positive updates; exactly fourfold negative error at the same
pre-update row/input; finite extremes; all-phase negative and positive labels;
unchanged query-presence/applicability/value/key/native updates at identical
supplied inputs/errors. Verify same-weight inference/admission identity,
generic/local separation, causal labels, no hidden label/target in inference,
zero/one/multiple recognized candidates, complete fallback/ties, chronology,
occurrence reversal, exhaustive selection and exact outer cursor.
Retain A5's independent hybrid references/all-256 inclusion/collision disclosure
and pre-update coupling. Run Python/native, lifecycle, protected-policy,
historical and ASan/UBSan checks. Lock passing receipts, source and folds **before
fitting**. Keep the independent interpreter and first/latest/constant shortcuts
on both changed keys/all orders and fresh binary canonical regression.
A perfect shortcut blocks a capability claim.

Every original gate and sample remains:

- Primary 128 worlds x4 order presentations for each condition and each answer
  position: >=95% mean /90% every seed.
- Both-order and all padding/repeat/diverse delays: >=90% mean /85% every seed.
- Same first 64 presentations at all 64 offsets: >=95% overall mean and >=90%
  worst-offset mean across seeds.
- Next 16 presentations x64 offsets at each independent 0/64/128/256 padding
  delay; 256 presentations for both-order and every lesion control.
- No-query drop >=20 points; reversed selection hurts; retain all state,
  pairing and gate lesions. Apply every original gate to fresh binary regression.
- Isolated full-distribution throughput >=80% native: one pinned CPU, three
  interleaved 65,536-byte repeats plus 1,024 warmup each, all other jobs stopped.

Lock final outcomes/snapshots/cursors before diagnosis. Repeat the separate
32-world identity/version geometry, 16/64/256 signature survival, diverse256
admission and survivor/filler-mass traces. Measure target false negatives as
well as incoming false positives: rejecting true records is not a repair.
Report native full-byte outcomes, not teacher-fed traces, as capability gates.
Charge every byte/padding byte, positive/negative local loss/gradient event,
A5 code comparison, repeated control, failed attempt and CPU cost.

Failure retains public E8, keeps A6 audit closed and triggers one measured next
preregistration. Passing development requires separate fresh generic/capacity
qualification: language <=+0.01 nats/byte, copy/reconstruction <=1-point token
loss, post-1M-language retention, complete three-seed 1M/5M and speed gates.
Passing overwrite or reduced false recognition never overrides another gate.

Completed verified work, including failure, is explicitly authorized for
commit/push directly to main without asking again. Preserve unrelated changes
and remote history; never force-push. Keep checkpoints/data/binaries outside
git. Deliver concise findings, verified commit link, ordinary downloadable
Markdown, primary PNGs, resumable code/evidence archive and another copy/paste
prompt preserving this authorization. No Pages or Spaces.

## Grounding

[Elkan, IJCAI 2001, The Foundations of Cost-Sensitive Learning](https://cseweb.ucsd.edu/~elkan/rescale.pdf)
provides the general basis for assigning distinct error costs. The fixed
four-phase/positive-weight parity rule above is this project's prospective
choice, not a claimed result of that paper. The original paired addressing
and MQAR references remain in A5's report. No paper establishes A6's outcome.
