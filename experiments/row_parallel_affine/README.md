# EXP03/05-A10: exact row-parallel affine execution

Completed result: exactness passes; execution qualification fails at
76.020287% native throughput, with an 11.182846% gain over paired scalar A9.
The twelve original measurements must not be rerun or replaced. See
`reports/exp0305a10/SHoEbox_EXP0305_A10_Report.md`, `ACCOUNTING.json` and
`RESUME.md`. The sole next A11 registration is unimplemented/unmeasured.

This is an explicitly selected execution backend for A9, with the same policy,
checkpoint magic and exact cursor. It is not a public default or an accuracy
experiment. Research fitting budget is zero. Original A9 and all older backends
remain unchanged. Read `reports/exp0305a9f1/A10_REGISTRATION.md` in full.

The only numerical execution change is four independent affine rows in SSE
lanes. Input coordinates remain in order 0–63, with a zero initial accumulator,
separate multiplication/addition and bias added afterward. A register-only 4×4
transpose supplies lane coefficients; there is no packed coefficient buffer,
heap allocation or persistent cache. Scalar `tanh`, local gradient/update order,
presence/applicability, source vectors, selection and the native core stay exact.
The local value projection adds a 32-byte automatic result buffer. Existing
query/key results are reused. Register shuffles are part of the same execution
interval; there is no separately timed packing pass or profiler.

`backend.Model(policy='a9')` selects row-parallel execution explicitly. Loading
E8/A8 through that helper dispatches to the unchanged historical libraries.
The model never serializes an execution choice. The benchmark locks that choice,
the generated source, compiler, flags and binary hashes externally.

Run required controls through the existing write-once process meter. Preserve
every failure and repetition. `check_python` executes all twenty original A9
tests for each backend; `check_native` runs independent bitwise affine cases and
the complete inherited numerical/mask/lifecycle/selection suites under optimized
and ASan/UBSan builds. Public/runtime and historical checks are also mandatory.

`verify_checkpoints` compares all 18 actual A9-F1 pairs for 1,000 input updates,
scalar versus candidate, both uninterrupted and 371/save/load/629. It compares
every full output vector and saved payload directly, plus resumed trace hashes
and final bytes. All trajectories are discarded; source snapshots stay intact.
`check_counts` uses test-only integer counters on separately charged trajectories
and must reproduce the uninstrumented probability and saved-byte hashes.

`lock_workload` selects the exact registered 16 fresh identities and independently
reconstructs the complete byte stream. It performs no model calls or scoring.
The committed `timing_worlds.lock.json` marks these groups opened for all future
allocations. Generated stream bytes and checkpoints stay outside Git.

`pre_timing` requires every passing receipt and pins the complete source closure.
Commit its lock before `benchmark`. The benchmark rejects a missing committed
lock or existing start marker, records every measurement immediately, and runs
only the four registered arms with three rotating repeats. No timing run may be
replaced. An execution pass leaves A9's accuracy failure and every development
gate intact. No A9 fitting, rescoring or audit access is authorized by this code.
