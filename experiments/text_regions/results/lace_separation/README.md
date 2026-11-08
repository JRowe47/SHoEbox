# Lace-separation pilot evidence

`summary.json` contains the fixed update-32 results for all four arms and the
retained parent: validation, final tests, unseen wording, same-weight controls,
mechanism measurements, cost, promotion gates and evaluation-file hashes.

`A_protocol.json` through `D_protocol.json` preserve configurations and the exact
223 source-file hashes enforced by checkpoint resume. The matching history files
contain all 32 updates, source positions, budgets and local-credit measurements.

The accompanying source/checkpoint archives include rendered cohort inputs,
per-case predictions, full resumable checkpoints, the exact corpus and SHA-256
manifests under `artifacts/lace_separation/`. Reproduction commands and the chosen
working checkpoint are in `docs/lace_separation.md` and `STATUS.md`.
