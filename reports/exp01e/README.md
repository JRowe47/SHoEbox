# EXP01-E evidence index

- [Findings, literature, comparisons and complete fitting costs](SHoEbox_EXP01E_Report.md).
- [Original protocol](PROTOCOL.md), retention pilot, and E2–E8 amendments.
- [Next registered revision, not fitted](AMENDMENT_E8.md).
- `summary.json`: per-seed primary, position, counterfactual, offset, delay and intervention summaries.
- `accounting.json`: every warmup/fit, padding bytes, targets, local updates, CPU and fitted parameter/state counts.
- `verification.json`, `export_verification.json`, and `evidence/*tests.log`: engineering checks.
- `evidence/`: original timestamped registrations, selection locks, raw curves/audits, diagnostic grafts, corpus hashes and isolated timing.
- `SHoEbox_EXP01E_*.png`: primary charts; `experiments/joint_readout/report.py` reproduces them.

Production remains minimal v19. E1's complete audit fails boundary mean and
retention; E7's development retention passes but generic qualification fails.
E2-family binding and E7 generic audit folds remain unopened. No E8 fit is
claimed. The downloadable code/evidence archive is outside git and contains
selected checkpoints, exact cursors, pinned data, hashes and resume instructions.
