# SHoEbox development

- Before new capability work, read `DEVELOPMENT_HANDOFF.md`,
  `docs/capabilities/SHoEbox_Minimal_Capability_Matrix.md`, `reports/exp01e/`,
  `reports/exp01d/`, and the paired/joint readout READMEs.
  EXP01-E1 completed three-seed binding, generic and 1M/5M qualification:
  98.83% primary exactness, but 94.11% all-offset mean fails the 95% gate,
  and language continuation erases binding. E7's locally supervised request
  gate preserves 99.41% in its development retention pilot, but fresh generic
  development fails (language +0.519175 nats/byte; copy -1.042 points).
  Production remains minimal v19. The next focused item is EXP01-E8,
  `reports/exp01e/AMENDMENT_E8.md`: cold-start applicability prior followed
  by complete qualification. E2-family binding and E7 generic audits remain
  unopened. Do not tune against E1's opened audit or inherit pilot results.
  Preserve every D/E policy, checkpoint meaning, failed fit and original gate.
  No new slots or optimizer sweeps without measured state failure. Distinguish
  explicit fixture-local supervision from generic byte objectives.
  Verified research may be committed and pushed directly to main under the
  user's standing authorization, including failed promotion; never force-push.

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
