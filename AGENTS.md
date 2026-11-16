# SHoEbox development

- Before new capability work, read `DEVELOPMENT_HANDOFF.md`,
  `docs/capabilities/SHoEbox_Minimal_Capability_Matrix.md`, `reports/exp01e8/`,
  the preserved E1-E8 registrations in `reports/exp01e/`, and the
  paired/joint/request/cold-gate READMEs.
  EXP01-E8 passes every registered research qualification gate: 98.27%
  held-out binding retained through 1M language targets, 97.27% offset mean,
  93.75% worst offset, 96.88% both orders and 100% padding delays.
  Generic language is +0.006233 nats/byte; copy/reconstruction remain within
  the original margins. Three-seed 1M/5M and >=80% speed gates pass.
  E8 is now the qualified public default: read `reports/e8_public/` and
  `multiscale/e8/README.md`. Public format is e8-public-1; explicit
  `core="minimal"` / `policy="v19"` retain v19. All 29 eligible fitted imports
  and exact outer cursors pass; preserve the pinned numerical policy. The E2-family binding
  and E7 generic audits are now OPENED; never tune against them. Any numerical
  revision needs fresh preregistered folds excluding all opened audit groups.
  Preserve every D/E policy, checkpoint meaning, failed fit and original gate.
  No new slots or optimizer sweeps without measured state failure. Distinguish
  explicit fixture-local supervision from generic byte objectives.
  Verified research may be committed and pushed directly to main under the
  user's standing authorization, including failed promotion; never force-push.

- The selected CPU runtime is the minimal corrected-C core, exposed by `shoe_multiscale.py` and
  implemented in `multiscale/`. Preserve `shoebox_reference.py` as the historical
  cortical comparator. New unspecified public models use E8 over the minimal v19 core: expert banks at 1/4 bytes,
  ordered summaries at 16/64 bytes, exact C search and coverage-only warmup.
  Explicit historical architecture arguments preserve their old constructors.
  Generic byte objectives freeze the local heads; observed-window labels are
  optional training-only input. Preserve bounded context, addressed/retained
  memory, reconstruction and child credit. Upper summaries have diagonal recurrence and constant learned output
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
