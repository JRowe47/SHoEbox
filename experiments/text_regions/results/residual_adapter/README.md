# Validation-scaled residual byte decoders

The study preserves each frozen raw linear head and fits a 16-unit correction.
Identity and GELU controls share initial tensors and parameter counts. All 72
corrections begin with exactly zero output. Validation selects one scalar strength
per head, including zero, before held-out and independent scoring.

`comparison.json` records all graph/initialization results and validation candidate
scores. `manifest.json` identifies graph checkpoints, frozen linear heads and the
previous feature caches. `artifact_inventory.json` identifies every downloadable
tensor, raw prediction file, selection lock and execution log by SHA-256.

The downloadable complete package restores all dependencies. From its root:

```bash
python -m experiments.text_regions.residual_compare \
  --source artifacts/residual_adapter_v1 \
  --out artifacts/reproduced_residual_comparison.json
```

The audit reconstructs synthetic cohorts, verifies source/checkpoint identities,
reproduces the validation choice and every evaluation prediction/loss from saved
weights, and checks exact agreement with the previous raw linear baseline.
Raw rows, tensors and feature caches stay outside the public repository.
