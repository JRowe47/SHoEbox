# Frozen readout diagnostic

Twelve frozen models cover three fine-tuning seeds, two publication policies, and two source schedules. Each model uses exactly the same diagnostic cases. Two common 256-byte decoders compare raw P/Q outputs with the current mixed/normalized latent; fixed factor probes measure decodability at the description boundary and final query.

GitHub contains the aggregate comparison, provenance, and file manifests. The complete downloadable package also restores `runs/` with every raw result, prediction row, and fitted-head inventory. Extract that package before running the saved-result comparator below. All episodes are generated synthetic byte tasks from `view_data.py`.

- `manifest.json` identifies the twelve checkpoints and portable result paths.
- `comparison.json` contains independently rescored answers, matched-policy/schedule aggregates, temporal probe summaries, input hashes, and exact expert budgets.
- `runs/*/results.json.gz` preserves each complete original diagnostic JSON losslessly.
- `runs/*/predictions.jsonl` contains exact answer bytes, targets, episode IDs, queries, and controls for independent rescoring.
- `runs/*/fit_inventory.json` records training identities, head dimensions, fitting metadata, and the tensor artifact hash.
- `readout_artifacts.json` identifies twelve fitted-head artifacts and their restore paths.
- `verification.json` and `verification.log` preserve experiment and automated-check evidence.
- `study_run_log.json` preserves the actual twelve invocations, checkpoint hashes, completion states, and timings.

Run from the repository root:

```bash
python -m experiments.text_regions.readout_compare \
  --manifest experiments/text_regions/results/readout_diagnostic/manifest.json \
  --out /tmp/SHoEbox_readout_comparison.json
```

The comparator reconstructs all input bytes and cases, checks disjoint fit/evaluation identities and fixed expert counts, verifies all six paired training protocols, and recomputes 1,080 answer accuracy records. Factor summary class counts and arithmetic are checked; the saved fixed heads support reproducing the factor diagnostics. The original graph weights are never optimized by this study.

Fitting uses 64 episodes for each supported query/source cell, 960 interventions across 192 unique base episode IDs, with equal loss mass per query. Evaluation uses 128 episodes per query/control. The test cohort contains 16 held-out fact combinations; independent controls exhaust all 64 combinations. All output predictions select among 256 bytes; fixed factor probes use four known answer classes.
