# Complete-episode gradient horizon

This phase compares two 32-update continuations from the regional-facts
control at update 100. Both see the same complete episodes and share the same
loss normalization, initial weights, AdamW moments and execution budgets.
The truncated arm detaches graph state every 16 events; the full arm retains
cross-chunk gradients until the complete episode loss is backpropagated.

See the [report](../../../../docs/SHoEbox_Gradient_Horizon_Report.md) and
[locked protocol](../../../../docs/SHoEbox_Gradient_Horizon_Protocol.md).

`selection.json` records validation-only checkpoint selection and verified
inference exports. `evaluation_audit.json` verifies held-out predictions,
generated targets, diagnostic arithmetic and matching intervention budgets.
`protocol_lock.json` binds the declared protocol and parent checkpoint.
Each arm's directory contains its protocol, per-update history, endpoint
summary, complete evaluation predictions and frozen local-head diagnostics.

Endpoint scores are always reported, even if selection prefers an earlier
checkpoint. A selected non-endpoint checkpoint has its own held-out evaluation.
Missing-source controls use an independent full-factorial cohort, preventing
hidden private answers from being inferred from a restricted factor split.

## Reproduce or resume

Extract the source and artifact ZIPs into the same directory. Both use the
`SHoEbox/` root. Checkpoints and logs live under `artifacts/gradient_horizon/`
and are intentionally outside git. From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/audit_gradient_horizon.py --evaluations
python scripts/check_all.py
```

For fresh runs, use new output directories:

```bash
python -m experiments.text_regions.horizon_train \
  --parent artifacts/gradient_horizon/parent.pt \
  --out artifacts/reproduced_horizon/truncated --detach-between
python -m experiments.text_regions.horizon_train \
  --parent artifacts/gradient_horizon/parent.pt \
  --out artifacts/reproduced_horizon/full
```

Append `--resume` with identical settings after an interruption. Every
checkpoint includes the optimizer, RNGs, complete stream cursors and phase
history. Runtime state resets between independent episodes in this phase.
The original streaming trainers and inference runtime remain unchanged.

To repeat one frozen endpoint evaluation:

```bash
python scripts/evaluate_regional_facts.py \
  artifacts/gradient_horizon/full/last.pt \
  --out reproduced_full_evaluation.json --count 512 --batch 256
```

Append `--resume` to continue the evaluation with the same checkpoint and
settings. The evaluator saves every completed stage atomically.
