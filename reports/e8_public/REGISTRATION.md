# E8 public integration verification plan

Starting main: 28f3ec2f88c9b593528c81dcae4cdaec14d5964e.
This is an engineering integration of the qualified fixed E8 policy. No new
research fit, prior sweep, objective, state, budget or numerical revision.

Copy the qualified readout arithmetic into the public runtime, changing only
checkpoint identity. Keep the historical native source and E1–E8 experiment
sources intact. Expose `policy="e8"` and explicit `policy="v19"`. Historical
explicit architectural constructors (including `core="minimal"`) retain their
meaning. Change only unqualified new-model default selection after all checks.

Require exact initial weights, full 256-byte outputs, local/generic updates,
partial-boundary checkpoint continuation and frozen lifecycle. Explicitly import
qualified kind-4 minimal E8 checkpoints; reject E7 and internal controls.
Retain hashes of both formats and unchanged outer cursors and decoder arrays.
Verify every qualified kind-4 snapshot using 1,000 updates versus
371 + serialization +629 and against the research trajectory, on discarded
copies. Use pinned train/development data only; no audit selection or retuning.

Run historical Python/native suites, all E8 numerical and causality tests,
public lifecycle and ASan/UBSan. Isolated throughput: pinned CPU, three
interleaved 65,536-byte repeats, native full-256-byte output, 1,024-byte warmup,
compilation/cloning excluded. Compare public E8, exact research E8 and matched
native at 500K/1M/5M and retained 500K; unchanged >=80% native threshold.

Record all correctness-only bytes, padding, objective and local update events,
process CPU (including compilation/evaluation), and all failed checks. No
correctness trajectory becomes a selected fit. E2-family binding and E7 generic
audits remain OPENED. Any numerical revision requires fresh preregistration
and groups excluding opened audits. Preserve v19 if integration fails.
