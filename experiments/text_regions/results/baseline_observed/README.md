# Source-observed baseline supervision

Four matched comparisons test `baseline_observed_only`: three evidence-gated seeds and an original-policy control at seed 20261106. The original and observed arms retain encoder gradients and differ only in the target mask. Detached models are separately matched context for the earlier intervention. Utility is disabled; all view losses remain enabled.

Gated-policy mean held-out accuracy changes from **68.2% to 68.6%**, private accuracy from **57.9% to 61.0%**, and B-only shared recovery from **85.1% to 70.3%**. The option remains default-off. The [report](../../../../docs/SHoEbox_Source_Observability_Report.md) covers seed variation, the original-policy control and the next source-visibility experiment.

- `manifest.json`: portable original/observed/detached paths; prior results reference `../evidence_policy` and `../baseline_gradient`.
- `comparison.json`: recomputed scores, paired changes, separate policy aggregates, source coverage and independent causal supervision audits.
- `runs/`: 28 exact new raw files, including protocols, histories, audited summaries, invocations and evaluation results/predictions.
- `runtime_source/`: four exact launch-source snapshots, their manifest, and bitwise two-step optimizer/gradient/state parity evidence against the final implementation.
- `raw_files.json` and `source_provenance.json`: hashes and provenance for 34 copied files and 40 referenced prior files.
- `verification.json` and `verification.log`: complete 223-test local verification at implementation commit `d9361c82e906890aed52dd39c17728846fdde52b`.

The snapshot's learner differs from the final learner only in accepting all-ignored targets without a mask. Every measured training call supplies the mask, and bitwise parity verifies its path is unchanged. The snapshot and final hashes match training and evaluation invocations respectively. Normal reproduction uses the current implementation.

All evaluations use 512 fresh episodes per query, balanced answers, and exact predictions over all 256 bytes. Held-out all-view combinations and full-factorial information-removal controls remain separate. Within each seed, model initialization, episodes, source observations, optimizer settings, 96,000 external events and 8,064,000 expert calls match.

From the repository root:

```bash
python -m experiments.text_regions.observability_compare \
  --manifest experiments/text_regions/results/baseline_observed/manifest.json \
  --out artifacts/reproduction/observability_comparison.json
```

The checkpoint bundle restores twelve compared trained models, the common initializer and twelve source readouts. No tensor checkpoints are stored in Git.
