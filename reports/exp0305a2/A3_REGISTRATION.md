# EXP03/05-A3: continuous chronological matching (registered, not fitted)

Registered after A2's complete three-seed primary development failure and frozen
occurrence diagnosis. A2 raises competing-binding exactness from 37.890625% to
92.96875% and corrected overwrite from 26.171875% to 66.015625%. Its unchanged
95% primary gate fails, as do clean/diverse retention prerequisites. E8 remains
public. No A1 or A2 audit is opened; no A2 refit or prior search is registered.

## Measured failure and one next change

On 32 development worlds/four order siblings, all three A2 seeds retain and
shortlist both versions of every changed key. Old values are predicted for
29.6875% of changed-key answers in the teacher-fed trace. No old/new learned
query distance ties exactly, so A2's latest-occurrence tie-break never activates.
Record access and version selection remain distinct. A2 also fails isolated
throughput at 68.07% of native; the original >=80% gate remains mandatory. The trace is diagnostic;
full-256-byte autoregressive primary scores govern qualification.

Keep A2's existing 16-slot admission, current-query extraction, key/value
projections, generic byte calibration, local objectives, rates and 1/256 gate.
Change only the record-selection/mixture rule to a continuous latest-match scan:

1. Decode every positively recognized record-phase key as A2 does. Let
   `d_i = sum_r (query_r - key_ir)^2`.
2. Define matching probability `p_i = sigmoid(4 * (7.84 - d_i))`.
   The fixed midpoint 7.84 is `QRANK * 0.7^2`, half the expected squared
   distance between independent 16-dimensional bipolar reconstruction codes.
   The slope 4 is the existing attention slope. Neither constant is selected
   from A2 distance percentiles or searched; this is an explicit new A3 matching
   rule, not a reinterpretation of A2's no-tolerance registration.
3. Traverse recognized occurrences from newest `born` to oldest, using stable
   slot then phase order for equal `born`. Assign
   `weight_i = p_i * product_{j preceding i}(1 - p_j)`.
   Retain A2's float32 distance accumulation, then evaluate the scalar expression
   and log weights in float64 with literal 7.84 and slope 4, using
   log-sigmoid/log-one-minus-sigmoid; no
   unregistered clamp, threshold, recency coefficient or slot expansion.
4. Keep the two largest weights, stable slot/phase order for exact ties, and
   normalize those two weights for the common paired-value mixture. With one
   recognized candidate its value receives all mass. With no recognized
   candidates use A2's complete historical presence fallback and distance
   attention. Do not add a parser, discrete decoded-byte equality, source oracle,
   symbolic key store or history tape.

A numerically exact implementation may stop projecting older candidates only
after two candidates are retained and the remaining log mass is strictly below
the second-largest retained log weight: no subsequent weight can exceed that
upper bound. Cache projected keys only in per-call scratch and reuse the selected
projections. Require equality to the exhaustive scan before fitting; no heuristic
cutoff or persistent cache is authorized. Record actual speed even if it fails.

This replaces distance-only competition with recency conditional on continuous
query/key matching. It adds zero fitted weights, persistent sequence state or
optimizer moments. Bounded temporary candidate arrays are allowed and must be
measured. Give A3 its own outer checkpoint identity; all E8/A1/A2 loaders and
public/historical constructors retain their saved meanings. No A3 implementation
or fit has occurred in this delivery.

## Fresh folds, controls, budget and unchanged decision

Use salt `SHoEbox-EXP0305-chronological-match-v3` with a NUL separator and world
RNG 20261125. Retain the corrected three-value fixture, 64 keys, distinct initial
values and third-value writes, balanced changed key and grouped operation/order
variants. Use 512/128/256 train/dev/closed-audit worlds and 128 fresh binary dev
worlds. Exclude every historical opened audit/test group via A1's predicate,
all reserved A1/A2 audit worlds and all 128 A2 development worlds, conservatively
including the additional A2 binary-regression development worlds. Fresh generic
qualification must exclude all opened E1/E7 document/task groups.

Match E8, A2 and A3, seeds 1/7/19, nominal 500K. Independently charge 250K clean
native warmup plus 2M mixed continuation per arm/seed: exactly 20.25M bytes.
No preliminary fit, inherited warmup, pilot, optimizer/calibration sweep or
uncharged repeated control. Order RNG 5701/5702, augmentation RNG +1000; retain
A2's five-condition mixture, 0–7 spacers, balanced phases, changed-key alternation,
finish cadence, full local-target accounting and exact interrupted cursor.

Before any fit, lock source/data and implement Python/native parity, causal
labels, zero/one/multiple-match limits, competing-key controls, old/new order
counterfactuals, finite saturated logits, lifecycle, checkpoint separation,
all three-value first/last/constant shortcuts and independent interpreters.
A perfect shortcut blocks a capability claim. Run all historical/public and
ASan/UBSan checks; meter every discarded check, failed attempt and CPU cost.

Every A2/A1 threshold and sample count remains: primary and both answer positions
95% mean/90% each seed; both orders and every delay 90%/85%; all-offset mean
95%/worst-offset seed mean 90%; no-query drop >=20 points, reversed selection
harm, all pairing/state/gate lesions, and >=80% isolated native throughput.
Retain fresh binary canonical regression at all original gates. Score changed
and untouched keys separately; diagnose long-diverse signature loss separately.
Passing overwrite alone does not qualify A3.

If development fails, retain public E8, keep the A3 audit closed and preregister
the next measured revision. If it passes, register fresh generic data before
language <=+0.01 nats/byte, copy/reconstruction <=1-point loss, post-1M-language
retention, and three-seed 1M/5M qualification. Preserve all old failures/costs.
Verified completed work, including failed research, may be committed and pushed
directly to main under the user's authorization; never force-push.

Primary grounding: Miller et al. (EMNLP 2016), paired key/value addressing,
https://aclanthology.org/D16-1147/; Arora et al. (ICLR 2024), input-dependent
selection and multi-query associative recall,
https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html.
The chronological matching rule and constants above are this proposed design.
