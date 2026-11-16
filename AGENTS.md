# SHoEbox development

- Before new capability work, read `DEVELOPMENT_HANDOFF.md`,
  `docs/capabilities/SHoEbox_Minimal_Capability_Matrix.md` and `reports/exp01d/`.
  EXP01-D3 / EXP22 passes the binding/access development gates: 99.22% native
  two-answer exactness across three seeds, every boundary offset above 98%,
  and 100% through +256 padding bytes. Use the `paired_latest` research policy
  with short-spacing training variation; preserve D1/D2 policies and evidence.
  The next item is EXP01-E: joint native-core/readout integration and production
  regression qualification, including a learned residual-applicability gate.
  No default promotion yet. Minimal v19 remains the production control.
  New slots and optimizer sweeps remain gated on a measured state failure.
  Record the selected core explicitly; do not merge historical/full-C evidence
  with minimal-v19 or research-adapter qualifications.

- The selected CPU runtime is the minimal corrected-C core, exposed by `shoe_multiscale.py` and
  implemented in `multiscale/`. Preserve `shoebox_reference.py` as the historical
  cortical comparator. New public models use v19: expert banks at 1/4 bytes,
  ordered summaries at 16/64 bytes, exact C search and coverage-only warmup.
  Preserve bounded context, addressed/retained memory, reconstruction and child
  credit. Upper summaries have diagonal recurrence and constant learned output
  modulation, with no expert bank, router, forecast head or refinement.
  `core="full"` preserves the previous v10 architecture. Explicit C/D controls
  retain v9 unless warmup-only is selected. Explicit experimental options select
  full when core is omitted; they must not silently extend the minimal core.
- Implemented architectural repairs belong in the default reference path.
  Enable compatible reference features and full credit coverage by default.
  Throughput regressions do not justify leaving a completed repair opt-in.
- Preserve explicitly selected historical kernels and saved checkpoint policies
  for reproducible comparisons. Do not silently reinterpret old stream state.
- Report effective capabilities. Experimental controls without a current
  implementation must not be presented as enabled reference features.
- Upper language-credit gains and retained-value producer credit are experimental
  controls. Numerical correctness does not establish expert specialization;
  require matched loss and active-expert permutation evidence before promotion.
- The MAD v1 harness explicitly uses the full historical architecture so published
  comparisons remain reproducible. New architecture comparisons must name their
  core; the minimal qualification is in `experiments/minimal_core/`.
- Validate changes with the relevant native regression tests, checkpoint
  continuation checks and sanitizer harness. Record throughput when measured.
- Keep trained models, datasets, compiled binaries and temporary snapshots out
  of git. Record compact source, tests and measurement summaries.
