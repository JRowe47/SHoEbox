# EXP03/05-A5: local discriminative key reconstruction (registered, not fitted)

Registered after A4's fixed 20.25M-byte development comparison and frozen
identity/version diagnosis. No A5 implementation, candidate evaluation or fit
has occurred. Qualified public E8 remains unchanged; A1/A2/A3/A4 audits stay
closed. Every previous result, throughput measurement and cost is preserved.

## Measured failure and focused change

A4 reaches 86.71875% competing-binding exactness, 94.79167% corrected overwrite,
95.89844% clean binding, 64.453125% diverse256 retention and 68.3469% native
throughput (80% required). On the 32-world
teacher-fed competing trace, required signatures survive/are recognized 100%,
but the nearest learned query/key candidate has the correct identity only
92.1875%. Overwrite nearest identity is 100%, while its latest-version accuracy
is 80.46875%. Old/new key vectors have mean squared separation 3.34195, and their
range overlaps different-key separations. A4 reduces unrelated selected mass
18.6088% →12.4083% versus matched A3, but old-value predictions return.

The measured ambiguity is in the learned query/key representation as well as
chronology. Another match-band constant search is not the next experiment.
A5 changes only the existing local query/key reconstruction objective, adding
fixed-code discrimination while retaining attraction to the original code.
This is explicit observed-window supervision, never a generic byte objective.

Separately, the diverse256 trace preserves 96.875% of original signatures.
All 24 A4 removal events across seeds occur when all 16 slots are classified
as records; 16.9679% of fully diverse incoming pairs are falsely recognized.
This measured admission problem remains separately reported. A5 does not claim
to repair it or add slots; every long-diverse endpoint remains required.

## Exact fixed local objective

For each existing positive query label or positive record-phase key label:

1. Keep the same 16 affine rows and input `x=source_vector(stable)`. Compute
   every pre-update `z_r=tanh(affine(row_r,x))` in the existing float32 arithmetic.
2. Keep the existing deterministic `byte_code(b,r)` ±0.7 codebook, all **256**
   byte labels, without changing any code or using the fixture key alphabet.
   Set `t_r=byte_code(observed_key_byte,r)`.
3. In float64 compute `d_b = sum_r (double(z_r)-double(code_b_r))^2` in increasing
   coordinate order, logits `s_b=-4*d_b`, and stable max-subtracted softmax
   `P_b=exp(s_b-max_s)/sum_j exp(s_j-max_s)` in increasing byte order.
   The literal 4 is the existing attention slope. No temperature search.
4. Compute `mu_r=sum_b P_b*double(code_b_r)` in increasing byte order. The
   objective is exactly
   `L = 0.25 * sum_r (z_r-t_r)^2 + (1/16) * (-log(P_observed))`.
   Its preactivation gradient is
   `e_r = 0.5 * (double(z_r)+mu_r-2*double(t_r)) * (1-double(z_r)*double(z_r))`.
   Cast each `e_r` to float32 once, then call the existing `affine_step` with the
   unchanged local rate `rate*auxiliary*4` and existing `1+energy(x)` normalization.
   All gradients use the same pre-update z/probabilities; do not update one row
   before computing another row's gradient.
5. Positive query and positive record-key events use this one objective.
   Negative presence, all five presence classifiers, applicability, **value**
   reconstruction, native responsibility credit and every other update remain
   numerically unchanged. No extra training labels, selected-record targets,
   retained label tape, optimizer moments or fitted coefficients are added.

The half weighting averages the original reconstruction gradient with the
softmax code-expectation gradient scaled by `1/(2*slope)`; it is a single fixed
design, not chosen from any A4 loss or distance percentile. Explicitly count
both local loss components, each 256-code comparison and all gradient events.
Codebook collisions, if any, must be disclosed before fitting; do not alter the
codebook or hide colliding byte labels to improve results.

Inference stays exactly A4: all recognized float32 distances, float64
`p_i=sigmoid(4*(double(d_min)+1.96-double(d_i)))`, chronology, stable ties,
top-two common paired-value mixture, one/zero-candidate fallback, unchanged
admission and existing state. No classification codebook is evaluated at
inference. Give A5 its own outer magic `0x3541353050584553`, nested v19;
protected loaders/constructors reject implicit conversion.

Training-only scratch may hold the fixed codebook/probability arrays. Inventory
its bytes separately from persistent sequence state and fitted parameters.
No new persistent projection cache. Exact arithmetic reuse or vectorization
across independent projection rows is allowed only with pre-fit bit-identical
native equality, preserving each row's float32 summation order and disabled FMA.
No heuristic early exit, approximate tanh, projection truncation or relaxed
throughput requirement is permitted. Measure actual isolated speed even on failure.

## Fresh folds, controls and exact budget

Salt `SHoEbox-EXP0305-discriminative-keys-v5` plus NUL; world RNG 20261127.
64 keys, three values, distinct initial values and third-value overwrite,
balanced changed key, grouped operation/order siblings. Counts remain
512/128/256 train/dev/closed-audit worlds plus 128 fresh binary dev worlds.
Exclude A1's complete opened-test-fold union including E2-family binding,
reserved A1/A2/A3/A4 audits, and **all A2/A3/A4 development and binary-regression
worlds**. No opened E1/E7 generic document/task group enters later qualification.
Pin every world and the full source/fixture/objective dependency closure first.

Matched **E8/A4/A5**, seeds **1/7/19**, nominal 500K: independent charged 250K
clean native warmup plus 2M mixed continuation per arm/seed, exactly **20.25M
bytes**. Order RNG 7701/7702; augmentation +1000. Preserve five-condition mix,
0–7 spacers, 64 balanced phases, changed-key schedule, local label codes,
finish cadence and exact interrupted cursor. No pilot, inherited warmup,
extra control, additional language fit, replay or objective/optimizer sweep.

## Pre-fit controls and unchanged decision

Verify Python/native and independent objective/gradient references, finite
extreme softmax/log losses, finite differences, descent on controlled positive
examples, all-256-code inclusion/collisions, pre-update coupling, causal labels,
generic/local separation and zero/one/multiple recognized candidates. Test
chronology/order counterfactuals, exhaustive selection parity, stable ties,
complete fallback, policy/lifecycle isolation and exact outer cursor. Confirm
unchanged value/presence/gate/native updates at identical supplied errors and
unchanged inference at identical weights. Run protected public/historical and
ASan/UBSan checks. Lock passing receipts/source/data before fitting.

Retain the independent interpreter and first/last/constant shortcuts on both
changed keys/all orders. A perfect shortcut blocks the capability claim.
Preserve full-256-byte autoregression and **every** original gate/sample:

- Primary 128 worlds ×4 order presentations per condition, and each answer
  position: >=95% mean /90% every seed.
- Both-order and all padding/repeat/diverse delays: >=90% mean /85% each seed.
- Same first 64 presentations at all 64 offsets: >=95% overall mean and >=90%
  worst-offset mean across seeds.
- Next 16 presentations ×64 offsets at each independent 0/64/128/256 padding
  delay; 256 presentations for both-order and all lesion controls.
- No-query drop >=20 points; reverse selection hurts; keep every state/pairing/
  gate lesion. Apply every original gate to fresh binary canonical regression.
- Isolated full-distribution throughput **>=80% native**, one pinned CPU, three
  interleaved 65,536-byte repeats plus 1,024 warmup each, all other jobs stopped.

Lock final outcomes/checkpoints before diagnosis. Trace identity/version
geometry and long-diverse signature loss separately. Account for every byte,
padding byte, repeated control, local loss/gradient/code comparison, failure and
CPU cost. Preserve all earlier failed costs and numerical/public policies.

On failure retain E8, keep A5 audit closed, diagnose and preregister one next
measured revision. On passing development separately preregister fresh generic
qualification: language <=+0.01 nats/byte, copy/reconstruction <=1-point token
loss, post-1M-language retention and complete three-seed 1M/5M qualification.
Passing one condition never overrides another failed gate. No E8 prior sweep.

Completed verified work, including failed research, is explicitly authorized
for commit/push directly to main without another confirmation. Preserve unrelated
changes and remote history; never force-push. Keep checkpoints/data/binaries
outside git. Deliver ordinary downloadable Markdown, primary PNGs, a resumable
code/evidence archive and a copy/paste prompt preserving this authorization.
No Pages or Spaces.

## Grounding

[Movshovitz-Attias et al., ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Movshovitz-Attias_No_Fuss_Distance_ICCV_2017_paper.html)
uses proxy-based distance objectives instead of enumerating training triplets.
Its proxies are learned; A5 uses this project's existing **fixed** byte codes
and the exact hybrid objective above, not an implementation of that paper.
[Miller et al., EMNLP 2016](https://aclanthology.org/D16-1147/) grounds preserved
paired addressing; [Arora et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html)
grounds multi-query recall measurement. No paper is claimed to establish A5's
performance or its chosen constants.
