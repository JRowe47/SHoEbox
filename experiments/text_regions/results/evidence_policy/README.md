# Evidence-policy comparison

Three matched fine-tuning seeds separate the effect of requiring evidence before publication at inference from learning with that policy. All runs start from a common correction-trained graph; each original/gated pair has identical initial tensors and training examples. A fourth pair disables the view objectives and is reported separately.

Mean B-only shared recovery rises from **35.1% to 85.1%**, while all-view held-out accuracy falls from **78.2% to 68.2%**. Private recovery regresses in every pair. The policy remains optional and default-off.

- [Report](../../../../docs/SHoEbox_Evidence_Policy_Report.md): matched budgets, seed ranges, diagnostics, and next ablation.
- `manifest.json`: portable paths for the four pairs and twelve evaluations.
- `comparison.json`: validated scores, paired prediction changes, seed aggregates, and reconstructed training source coverage.
- `source_references.json`: provenance and hashes of the copied raw artifacts.
- `verification.json` and `verification.log`: artifact integrity and the complete 189-test CI transcript for the implementation commit.
- `SUPPLEMENTAL_README.md` and `supplemental_manifest.json`: regional readouts and query-only interference diagnostics, with their distinct sample sizes and provenance.

Recompute the comparison from the repository root:

```bash
python -m experiments.text_regions.evidence_compare \
  --manifest experiments/text_regions/results/evidence_policy/manifest.json \
  --out artifacts/evidence/recomputed_comparison.json
```

Each main evaluation uses 512 fresh episodes per query (1,536 per control). The regional transfer diagnostic uses 192 per query; the query-only intervention uses 512 shared-query episodes. Their reported percentages are not interchangeable. Raw predictions and training histories are included; learned tensors are in the separately downloadable checkpoint bundle.
