# A11: fixed eight-row exact affine execution

The sole mechanism is the AVX2 8×8 register transpose preregistered in
`reports/exp0305a10/A11_REGISTRATION.md`. `backend.Model` explicitly selects it;
public E8, scalar A9 and A10 remain unchanged. There is no scientific policy,
state, serialization, gradient, accumulation-order or scalar-tanh change.
Only the same six affine call sites change execution. FMA is disabled.

`instrument.py` wraps the compiler only for correctness runs. It keeps a full
source-indexed ledger of dynamic project functions, loop iterations, comparison
expressions and automatic block-object activations, with explicit consumed-byte,
target, affine-row and local-gradient-row counters. The complete site maps and
raw counters are in the resumable archive; their joined ledger is committed as
compressed JSON lines. Standard-library internals and loop induction scalars
are not mislabeled as additional model scratch. Source-level SIMD storage and
generated stack frames, including spills, are reported separately.

`control.py` gives every command a new write-once directory, compiler cache,
counter table, Python API ledger and whole-child-tree CPU receipt. Counters do
not enter saved state or production binaries. Original correctness tests run
unchanged. `verify_checkpoints.py` compares three counted and three uncounted
paths at every original cursor, uninterrupted and 371/save/load/629. All work
is discarded. No research checkpoint is updated and no accuracy is rescored.

`prepare.py` compiles production libraries, rejects generated FMA instructions,
and records stack usage with identical machine-code text. It does not execute
model inputs. `lock_workload.py` verifies identity-only exclusion commitments,
allocates seed 20261203 once, and independently constructs the literal stream.

`pre_timing.py` requires passing controls, complete accounting and original
file hashes. Its lock must be committed before `benchmark.py` accepts the run.
The benchmark has exactly five arms and fifteen first measurements. A timing
marker prevents repetition. **Do not rerun a completed measurement.** Consult
the result report and its next registration for authorized future work.

Research fitting budget: zero. A9-F1's separate 20.25M fitting bytes are
exhausted. Exact execution or a speed pass never repairs A9's accuracy failure,
opens an audit, relaxes a development gate or changes the public default.
