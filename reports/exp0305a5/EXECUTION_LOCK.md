# A5 execution lock, fixed before fitting

Execute `reports/exp0305a4/A5_REGISTRATION.md` unchanged from
`116efa345aa435780044e3ad5705350b6268f1a6`. Isolated module:
`experiments/discriminative_keys`. Public E8 and all historical numerical
sources/results/costs remain unchanged. No prior, band, codebook or optimizer sweep.

Only positive query and record-key local reconstruction changes. Float32
pre-update tanh projections feed all 256 unchanged +/-0.7 codes. Distances,
stable softmax and coordinate means use float64 and increasing registered
summation orders, with FMA disabled. Loss is exactly 0.25 squared distance plus
1/16 negative log probability; gradient is exactly
`0.5*(z+mu-2*t)*(1-z*z)`, cast once to float32 before the original normalized
affine step at `rate*auxiliary*4`. All 16 errors exist before any row update.
Value/presence/applicability/native updates and A4 inference/admission are exact.
No inference codebook, extra weights, optimizer state, slots or projection cache.
A5 magic `0x3541353050584553`, nested v19; protected loaders reject conversion.

Pre-fit full codebook disclosure: 256 unique codes, zero collision pairs,
minimum Hamming distance one. Little-endian float32 table SHA-256:
`c8292df6bb4ea5e5a6c98daaf9796d008d03b0672107744637a13548a785559a`.
Training-only read-only codebook is 16,384 bytes. Objective scratch is 2,256 bytes and the
pre-update vector is 64 bytes, measured by native checks; neither enters model
state/checkpoints. A4 prediction/selection scratch is unchanged.

Fresh salt `SHoEbox-EXP0305-discriminative-keys-v5` plus NUL, world RNG 20261127.
512/128/256 train/dev/closed-audit worlds, plus 128 fresh binary dev worlds.
Exclude the complete old opened binding-fold union including E2-family, all
reserved A1/A2/A3/A4 audits, and all A2/A3/A4 dev/binary worlds. Group every
operation/order sibling. No generic data is used; opened E1/E7 generic groups
remain excluded from any future qualification.

Matched E8/A4/A5; seeds 1/7/19; independent 250K native warmup and 2M mixed
continuation per arm/seed; exactly 20,250,000 bytes. Order RNG 7701/7702,
augmentation +1000. Preserve five-condition mix, 0–7 spacers, 64 balanced phases,
changed-key schedule, labels and finish cadence. No pilot/inherited warmup/refit.
Cursor `exp0305-a5-cursor-1` retains RNGs, partial examples, shuffle epoch,
operation choices and finish schedule. Verify all 18 actual snapshot/cursor pairs.

Lock passing Python/native, independent float64 Python and long-double native
objective/gradient references, finite differences, descent, coupled updates,
all-byte codebook, source isolation, exhaustive A4 selection, causal labels,
shortcut/interpreter, lifecycle, protected policies, historical trajectories and
ASan/UBSan receipts with the complete fitting source/data closure before fitting.
Native selection still includes zero/one/multiple candidates, stable ties,
occurrence reversals, oldest-distance minimum and complete fallback.

Every original threshold/count remains: primary 128 worlds x4; first 64
presentations at 64 offsets; next 16 x64 offsets at each independent 0/64/128/256
padding delay; 256 both-order and lesion presentations. Fresh binary canonical
regression repeats all original gates. Full 256-byte autoregression. Mean/seed
primary and each-position >=95/90%; offsets >=95% mean/90% worst-offset mean;
both orders/all delays >=90/85%; no-query drop >=20 points; reverse hurts.
Isolated native/E8/A4/A5/public-E8 speed >=80% native: one CPU, three interleaved
65,536-byte repeats plus 1,024 warmup each, all other research jobs stopped.

Final outcomes/checkpoints lock before frozen diagnosis. Report identity/version
geometry separately from long-diverse signature/admission loss. Charge all
input/padding, local attraction/discrimination events, 256-code comparisons,
gradient rows, discarded correctness, repeated controls, failures and CPU.
Failure keeps A5 audit closed and E8 public, then preregisters one measured
revision. Passing development requires separately registered fresh generic and
three-seed 1M/5M qualification at all original gates.

Completed verified work, including failure, is authorized for direct-main
commit/push; preserve unrelated history and never force-push. Deliver Markdown,
primary PNGs, resumable archive and copy/paste prompt with this authorization.
