# EXP03/05-A11: fixed eight-row exact affine execution

Registered after A10's complete, outcome-locked failure. **Not implemented or
measured. Research fitting budget: zero bytes.** This is the sole next
experiment. Preserve public E8, scalar A9, A10, every original result and cost,
the original A9 allocation failure, and closed A1–A9 audits. The independent
A9-F1 20.25M-byte fitting budget remains exhausted.

## Single change

A10's four-row kernel preserves bitwise outputs and saved state. Its fixed
measurement reaches 76.020287% of native and 111.182846% of scalar A9
throughput, below the unchanged 80% native gate. This registration changes
only the number of independent affine rows executed together: four to eight.
It does not change A9's failed accuracy or numerical policy.

Use one fixed AVX2 eight-row kernel at the same six A10 call sites. For each
eight-row block and each eight-coordinate block, load eight contiguous float32
coefficients from each row and transpose the 8×8 block in registers using
unpack, shuffle and 128-bit-half permutation operations. One 256-bit vector
then holds the eight coefficients for one input coordinate. Initialize the
eight row accumulators to positive zero; visit coordinates in order 0–63,
performing a separate multiply and add for each coordinate. Add the original
bias after coordinate 63, in the original operand order. A 16-row projection
uses two such blocks; an eight-row projection uses one. Preserve scalar tanh.
No cross-coordinate reduction, FMA, fast-math, reassociation, approximate
nonlinearity, alternate lane width or kernel selection is permitted.

Use only bounded automatic/register scratch, with no persistent packing or
coefficient cache. Record source-level scratch bounds and generated stack
usage, including spills. Register transposition is part of the measured
kernel; there is no separate packing pass or packing microbenchmark. Retain
A10's existing 32-byte local-value result buffer. No additional model field,
saved byte, sequence state, optimizer state, candidate projection, argument,
label, parser, cutoff, source-vector operation or gradient change is allowed.
If AVX2 is unavailable or exact arithmetic cannot be established, stop and
record failure; do not substitute another mechanism.

Keep A9 scalar and A10 source immutable. Compile A11 into an explicitly
selected separate backend, preserving A9 magic, serialization and cursor
schema. Record execution identity, generated source, compiler and flags
externally. Use the same optimization flags and disabled FMA as A10; no
compiler-flag sweep or public default change.

## Controls and accounting before timing

Before any performance measurement, complete and commit a passing source,
control, workload and accounting lock. Retain all failures and repetitions.

1. Independently compare every row against original scalar affine, A10 and A11
   bitwise, covering both row counts, all five key/four value phases, zero,
   signed zero, bias, cancellation, subnormals, finite extremes and fixed
   random inputs. Keep the original A10 input generator/seed and trial count.
   Compare scalar tanh and inspect generated instructions for disabled FMA.
2. Run the original 20 A9 tests on all three execution paths; retain protected
   policy isolation, generic/local separation, enabled A8 parity, omitted-key
   masking, all-256 normalization, lifecycle, causality, exhaustive selection,
   fallback/ties, fixed-code gradients, Python/native and optimized/ASan/UBSan
   controls. Run public/runtime, 209 historical research, protected-loader
   and 14 historical checkpoint controls. v5/v6 remain unchanged even where
   the historical independent-constructor suite does not instantiate them.
3. For all 18 original A9-F1 warmup/final model-cursor pairs, compare scalar A9,
   A10 and A11 for 1,000 steps at the actual cursor and for the exact
   371/save/load/629 continuation. Compare every full output vector, native
   and local row, state/counter, saved byte and cursor. Discard all trajectories.
   Preserve all 36 frozen files hash-identically. No accuracy is scored.
4. Instrument correctness-only entry points before running their suites to
   record every input, target, local update, projection, comparison and scratch
   activation, including inherited microtests. Counters must not enter the
   timing library or saved state. Compare counted and uncounted outputs/saved
   bytes where needed, with every additional correctness trajectory charged.
   No inherited microtest event may silently be recorded as zero. Include
   complete child-process CPU, compiler CPU, failed controls and bookkeeping.

An exactness failure blocks timing. Implementation corrections may be tested
with failed receipts preserved, but may not change the registered mechanism.
No pilot, profiler, calibration run, accuracy rescoring, audit episode,
research fitting, replay fit or inference-policy experiment is permitted.

## Exact fresh timing allocation

Use precisely A10's independent nested enumeration: key0 0..63, key1
key0+1..63, value0/value1 0..2, unequal values, in that order. Exclude the
unchanged A9 `tasks.excluded` union, all 896 original A9 primary identities,
all 128 A9-F1 binary identities, and all 16 A10 timing identities in
`reports/exp0305a10/timing_worlds.lock.json`. Read identity manifests only;
never instantiate or score excluded audit worlds.

Shuffle the complete remaining list exactly once with
`random.Random(20261203)` and take its first 16 worlds. Stop for an allocation
amendment if fewer than 16 remain. No salt search or substitute sample.
Independently verify the enumerator, exclusion union and literal byte stream.
Lock ordered identities, hashes, stream length and PAD/nuisance counts before
timing; mark the 16 identities opened for future allocation exclusions.

Construct complete episodes with the unchanged A9 fixture in this order:
conditions `pad0, repeat16, diverse16, compete16, overwrite16, diverse256`;
then selected world order; then variants 0..3. Use zero offset and spacers,
`updated=world_index % 2`. Concatenate input bytes only. No labels or expected
answers reach any benchmark. Compute all 256 outputs at every input.

## Five fixed timing arms and decision

Arms, in this exact order:

1. Fresh untrained nominal-500K native v19, seed 7.
2. Original archived seed-7 final E8 checkpoint.
3. Original archived seed-7 final scalar A9 checkpoint.
4. That same A9 checkpoint through the immutable A10 backend.
5. That same A9 checkpoint through A11.

Compile and load outside measured intervals and charge process CPU. Finish
all other local research jobs, pin one CPU, then run exactly three interleaved
repeats, rotating the arm order left by repeat index 0/1/2. Per arm/repeat,
reset, warm 1,024 bytes, reset, measure 65,536 bytes. Cycle the fixed stream
only as needed. Retain every first raw duration and exact checksum; no retries
or favorable replacement measurement. This is 998,400 total inference bytes,
including 15,360 warmup bytes, and zero research fitting bytes.

All exactness controls and scalar/A10/A11 checksums must agree bitwise. A11
must reach **at least 80% of native throughput**, and its median CPU time must
be no larger than either A10's or scalar A9's median on this same fresh stream.
Preserve A10's original 76.020287% result separately; do not replace it or
treat different streams as paired. Every condition is mandatory.

A speed pass does not repair A9 accuracy, satisfy development, authorize audit
access or promote a public default. Keep every seed/position/offset/order/delay,
lesion/binary and generic/capacity gate unchanged. End with one focused next
preregistration, without executing another experiment.

Completed verified work, including failure, remains explicitly authorized for
commit and push directly to main without further confirmation. Preserve
unrelated changes and remote history; never force-push. Keep checkpoints,
datasets and compiled binaries outside git. Deliver findings, verified commit
link, ordinary downloadable Markdown, primary PNGs, resumable archive and a
copy/paste prompt preserving this authorization. No Pages or Spaces. Export
and attach without confirmation.
