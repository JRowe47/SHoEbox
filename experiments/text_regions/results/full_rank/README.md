# Full-rank decoder phase

Seventy-two deterministic full-matrix byte decoder fits, across twelve frozen
graphs and the same nested 960/3,840 training cohorts. The comparison re-audits
all 432 inherited identity/GELU corrections. Aggregate accuracy results and CPU timings
are summarized in summary.json. Detailed per-head comparisons, timing repetitions, tensors,
caches and raw synthetic predictions are in the downloadable artifact package.
See `docs/SHoEbox_Full_Rank_Readout_Report.md`.

```bash
python -m experiments.text_regions.full_rank_compare \
  --out artifacts/reproduced_full_rank_comparison.json
```
