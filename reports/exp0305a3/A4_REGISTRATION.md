# EXP03/05-A4: query-relative chronological matching (registered, not fitted)

Registered after A3's full fixed 20.25M-byte comparison and frozen occurrence
diagnosis. No A4 implementation, candidate evaluation or fit is performed here.
Qualified public E8 remains unchanged. A1/A2/A3 audits remain closed.

## Measured failure and one next change

A3 raises corrected three-value overwrite from matched A2's 76.8229% to
95.9635% but lowers competing bindings from 95.3125% to 85.8073%. Clean binding
is 94.6615%, and long-diverse retention also fails. Its frozen changed-key trace
has 100% old/new signature survival, 100% changed-key correctness and zero
old-value predictions. Chronology fixes this measured version conflict.

The same learned projections feed A2/A3, but A3's absolute matching probability
admits unrelated newer keys. In competing traces, mean selected mass on another
key rises from about 4.36% to 10.40%; 7.5521% of answers have a newer unrelated
candidate with matching probability above 0.5. Only 85.9375% of recognized target
occurrences are above that probability. These are teacher-fed development
diagnostics, not qualification scores. Long-diverse signature loss remains a
separate measured problem; changing the matcher cannot recover a missing unit.

Keep A3 admission, existing state, current-query extraction, local projections,
value pairing, generic byte calibration, objectives, rates, cold gate and all
thresholds. Replace only the absolute query-match midpoint with a query-relative
distance band, preserving continuous chronological weights:

1. Recognize the same record-phase candidates with strictly positive presence
   logit. Compute every candidate key and query distance with A2/A3's float32
   accumulation. Keep projections in bounded per-call scratch only.
2. Let `d_min` be the smallest recognized-candidate distance, including all
   candidates regardless of age. In float64 set
   `p_i = sigmoid(4 * (double(d_min) + 1.96 - double(d_i)))`.
   The literal band **1.96** is the squared distance of one full coordinate sign
   change in the existing ±0.7 reconstruction code: `(2 * 0.7)^2`.
   It is a single fixed design choice, not selected from A3 distance percentiles
   or a coefficient search. Slope remains the existing literal 4.
3. Visit newest `born` first, stable slot then phase for equal `born`, and use
   `weight_i = p_i * product_previous(1-p_j)` with float64 log-sigmoid arithmetic.
   Keep the two largest log weights, stable slot/phase for exact ties; normalize
   those two for the shared paired-value mixture. One recognized candidate gets
   all mass. None uses A2/A3's complete historical fallback and attention.
4. The minimum-distance pass must see every recognized candidate. Do not reuse
   A3's early projection stop before finding that minimum. Reuse projected keys;
   exact arithmetic/common-subexpression reuse is allowed, with pre-fit equality
   to an exhaustive independent reference. No heuristic cutoff, hard decoded-key
   equality, parser, source oracle, extra slot, persistent cache or history tape.

This tests whether A2's reliable relative query geometry can constrain A3's
chronological preference without an absolute reconstruction-distance threshold.
No fitted coefficient or optimizer moment is added; bounded scratch and actual
speed must be measured. Give A4 a distinct outer checkpoint identity; preserve
every historical/public loader and constructor. No A3 refit or E8 prior sweep.

## Fresh folds, controls and exact budget

Salt `SHoEbox-EXP0305-relative-chronological-v4` plus NUL; world RNG 20261126.
Keep 64 keys, three values, distinct initial values, third-value writes, balanced
changed key and grouped operation/order variants. Counts remain 512/128/256
train/dev/closed-audit worlds plus 128 fresh binary development worlds.

Use A1's complete historical opened-test-fold exclusion, including E2-family
binding, plus reserved A1/A2/A3 audits and all A2/A3 development and binary
regression worlds. No opened E1/E7 generic document/task group may enter later
qualification. Pin every generated world hash and fitting dependency before
any fit. Never group operation or order siblings into different folds.

Matched E8/A3/A4, seeds 1/7/19, nominal 500K: independently charged 250K clean
native warmup plus 2M mixed continuation per arm/seed, exactly **20.25M bytes**.
Order RNG 6701/6702, augmentation +1000. Preserve five-condition mixture,
0–7 spacers, balanced 64 phases, changed-key alternation, local target codes,
finish cadence and exact interrupted cursor. No pilot, inherited warmup,
uncharged repeated control, new objective or optimizer/calibration sweep.

## Controls and unchanged decision

Before fitting: Python/native and independent exhaustive parity, finite extreme
log weights, zero/one/multiple recognized candidates, competing-key and old/new
occurrence-order counterfactuals, stable ties, causal labels, generic/local
separation, lifecycle/checkpoint isolation and exact outer cursor. Independently
interpret both changed keys/all orders; first/last/constant shortcuts must fail.
A perfect shortcut blocks a capability claim. Run public/historical and
ASan/UBSan checks; lock source, data and passing receipts before fitting.

Preserve **every** A1/A2/A3 gate and sample count: 128 primary worlds ×4 order
variants; first 64 presentations at all 64 offsets; next 16 across all offsets
at each independent 0/64/128/256 padding delay; 256 both-order/lesion
presentations. Primary and each answer position >=95% mean /90% each seed;
both orders and every padding/repeat/diverse delay >=90% /85%; all-offset mean
>=95%, worst-offset seed mean >=90%; no-query drop >=20 points, reversed
selection harm, and every state/pairing/gate lesion. Apply all original gates
to fresh binary canonical regression; retain corrected three-value overwrite.
Score changed and untouched keys separately and trace long-diverse signature loss.

Isolated full-distribution throughput remains **>=80% native**: one pinned CPU,
three interleaved 65,536-byte repeats plus 1,024 warmup per repeat, all other jobs
stopped. The extra minimum-distance pass has a real cost; do not weaken this gate.
Account for every pilot, padding byte, local update, repeated control, failed
attempt and CPU cost. Full-256-byte autoregression remains the endpoint.

If development fails, retain E8, keep A4 audit closed, diagnose and preregister
one next measured revision. If it passes, separately register fresh generic
qualification: language <=+0.01 nats/byte, copy/reconstruction <=1-point token
loss, post-1M-language retention and complete three-seed 1M/5M qualification.
Passing overwrite alone is insufficient. Preserve all prior scores and costs.

Completed verified work, including failed research, may be committed and pushed
directly to main under the user's explicit authorization. Preserve unrelated
changes and remote history; never force-push. Keep checkpoints/data/binaries
outside git. Deliver ordinary downloadable Markdown, primary PNGs, resumable
code/evidence archive and a copy/paste prompt preserving this authority.

Grounding: Miller et al., EMNLP 2016, https://aclanthology.org/D16-1147/;
Arora et al., ICLR 2024,
https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html.
The query-relative band and exact rule above are this project's proposed design.
