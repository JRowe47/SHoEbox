# Decoder cohort-diversity reproduction

Twelve frozen graphs, nested 960/3840 training cohorts, 432 residual fits,
and unchanged validation/evaluation caches. `comparison.json` contains the
audited aggregate and per-head metrics. `manifest.json` records the dependencies.
Full tensors, features and synthetic prediction rows are in the downloadable
artifact package. `docs/SHoEbox_Cohort_Reproduction_Report.md` compares these
results with the surviving report of the missing experiment.

```bash
python -m experiments.text_regions.diversity_compare \
  --source artifacts/cohort_reproduction_v1 \
  --out artifacts/reproduced_cohort_comparison.json
```
