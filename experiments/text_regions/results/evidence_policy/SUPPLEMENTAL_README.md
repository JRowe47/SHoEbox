# Supplemental evidence-policy diagnostics

These files preserve two additional diagnostics alongside the main policy comparison. The seven recorded files were copied byte-for-byte from their completed runs. `supplemental_manifest.json` records their original locations, sizes and SHA256 hashes, plus this README's hash. It does not alter the main comparison's manifest or verification scope. No model or readout tensors are included here; restore them from the separate checkpoint bundle before reproduction.

## Query-only interference

`query_interference/results.json` and `predictions.jsonl` record 512 paired shared-query episodes from the independent full-factorial cohort. The same original-trained views checkpoint was evaluated with original and evidence-gated publication, under B-only observations, a final query marker added on A/C, and every query marker added on A/C. All six conditions used identical cases and 1,075,200 expert executions. Scores use argmax over all 256 output bytes. The raw file contains six condition records with episode IDs, answers, predictions and B-template IDs.

`recorded_driver.py` is the exact historical driver; its SHA256 matches `script_sha256` in the recorded result. Its default repository path was intended for the original scratch location. Run it from the repository root with **`--repo .`** and a separate output directory:

```bash
python experiments/text_regions/results/evidence_policy/query_interference/recorded_driver.py \
  --repo . --checkpoint artifacts/views/views/checkpoint.pt \
  --count 512 --batch 128 --threads 2 \
  --out artifacts/reproduction/query_interference_recorded
```

The recorded run used **batch 128**, rather than the reusable CLI's default 64. Preserve that batch size when comparing the recorded predictions: batch-dependent floating-point execution can affect tied or near-tied decisions. The driver uses target-presence masks only to locate the fixed query schedule; no target values or factor metadata enter the model call. The maintained CLI instead locates query markers directly from input bytes and validates that no factual values are exposed:

```bash
python -m experiments.text_regions.query_interference artifacts/views/views/checkpoint.pt \
  --policy both --count 512 --batch 128 --threads 2 \
  --out artifacts/reproduction/query_interference
```

Both commands inherit seed, gap and query interval from the checkpoint. The maintained CLI uses a newer output schema and condition names; the copied historical files remain the record of this run.

## Frozen regional transfer readouts

`transfer/old_gated/` contains the original-trained views checkpoint's diagnostic under evidence-gated inference. `transfer/trained_gated/` contains the evidence-gated-trained views checkpoint's diagnostic under the same inference policy. Each directory preserves `results.json` and `invocation.json`; the latter records the exact original command and successful exit status.

Each run fits frozen linear readouts on 192 training episodes per control, then evaluates 192 episodes per query type in held-out and independent cohorts. Results include pre-query and final regional ports, final source ingress, valid-state coverage, pooled and query-matched fits, direct model answers, equal compute checks and unchanged model fingerprints. Reproduce from the repository root:

```bash
python -m experiments.text_regions.transfer_eval artifacts/views/views/checkpoint.pt \
  --publication-policy evidence --fit-count 192 --count-per-query 192 \
  --batch 64 --threads 2 --out artifacts/reproduction/transfer_old_gated
python -m experiments.text_regions.transfer_eval artifacts/evidence/views/checkpoint.pt \
  --publication-policy evidence --fit-count 192 --count-per-query 192 \
  --batch 64 --threads 2 --out artifacts/reproduction/transfer_trained_gated
```

The corresponding `transfer_readouts.pt` files are in the checkpoint bundle at their original `artifacts/evidence/transfer_*` paths. Recorded absolute scratch paths and timings are preserved as provenance; portable commands above write new outputs without overwriting the record.
