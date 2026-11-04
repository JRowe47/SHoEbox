# Baseline encoder-gradient ablation

Four matched pairs test whether the source-only baseline should update the shared encoder. Three pairs use evidence-gated training and inference; the fourth uses the original publication policy at seed 20261106. Only `baseline_detach_features` differs within each pair. The baseline head continues to train, all view losses remain enabled, and utility updates remain disabled.

Gated-policy mean held-out accuracy rises from **68.2% to 77.6%** and pooled private accuracy from **57.9% to 73.7%**. B-only shared recovery falls from **85.1% to 67.3%**, and seed 1 loses left-private accuracy. The switch remains default-off. The [report](../../../../docs/SHoEbox_Baseline_Gradient_Report.md) explains the results and the next source-observability ablation.

- `manifest.json`: portable paths for four attached/detached pairs; attached results reference the existing `../evidence_policy` package.
- `comparison.json`: scores recomputed from raw predictions, paired gains/losses, seed aggregates, and reconstructed training source coverage.
- `runs/`: 28 exact raw files for the four new detached runs, including protocols, summaries, histories, invocation provenance, and evaluation results/predictions.
- `raw_files.json` and `source_provenance.json`: sizes, SHA256 hashes, original locations, and runtime source hashes.
- `verification.json` and `verification.log`: artifact integrity and the full passing 204-test local transcript for code commit `d66b0ef353738a2be7fa19054209ffda22de4086`.

Every evaluation uses 512 fresh episodes per query (1,536 per control), with balanced four-way answers and exact predictions over all 256 bytes. Held-out all-view evaluation and the full-factorial information-removal controls are kept separate. These are paired fine-tuning seeds from a common pretrained graph, not independent graph pretraining runs.

From the repository root:

```bash
python -m experiments.text_regions.baseline_compare \
  --manifest experiments/text_regions/results/baseline_gradient/manifest.json \
  --out artifacts/reproduction/baseline_comparison.json
```

The separate checkpoint bundle restores all eight compared models, the common initializer, and eight source readouts. No tensor checkpoints are included in Git.
