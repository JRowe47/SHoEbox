# Balanced whole-source availability

Six matched comparisons cover three fine-tuning seeds under both original and evidence-gated publication. Each compares the existing dropout schedule (`dropout=0.25`) with balanced supported whole-source subsets (`source_schedule=balanced`, `dropout=0`). Shared queries use all seven nonempty subsets; left-private queries retain A, and right-private queries retain C. Episode facts, identifiers and renderings remain identical within each pair.

Both baseline interventions remain disabled: baseline gradients are attached and the original eligible targets contribute to its loss. View objectives are enabled; routing utility is disabled. Every run has 240 optimizer steps, 96,000 aligned external events and 8,064,000 expert executions. Source observations and auxiliary-label eligibility intentionally differ and are audited separately.

The three seeds share a common correction-trained graph initializer. Each matched pair has identical initial tensors, model size, ordered training cases and expert budget. Evaluations contain 512 examples per query type: held-out test tuples plus the full-factorial independent cohort with all six source/reset controls. Answers are argmax predictions over all 256 bytes.

## Files

- `manifest.json` uses portable paths to six new runs and the six existing controls in `../evidence_policy/`.
- `comparison.json` contains recomputed prediction scores, both three-seed policy groups, three paired policy interactions, source-exposure audits, baseline-supervision audits and hashed inputs.
- `runs/` preserves 42 exact raw files: protocol, summary, training history, both invocations, evaluation results and predictions for each new run.
- `default_training_parity.json` records bitwise equality of the unchanged default dropout path with the previous implementation through both actual batch-32 chunks, including all 691 parameter gradients, optimizer updates, carried state and RNG.
- `raw_files.json` hashes those 43 copied files and 30 existing referenced files.
- `source_provenance.json` binds launch/evaluation source hashes to the tested code revision. The original launch commit remains in each invocation; the later tested commit includes the comparison tooling with identical runtime code.
- `verification.json` and `verification.log` record the successful 245-test check: 60 preserved baseline tests and 185 graph tests.

Reproduce the comparison from the repository root:

```bash
python -m experiments.text_regions.source_schedule_compare \
  --manifest experiments/text_regions/results/source_schedule/manifest.json \
  --out /tmp/SHoEbox_source_schedule_comparison.json
```

The comparator reconstructs source observations, supported subsets, answer/tuple coverage and auxiliary eligibility independently. It verifies fixed compute budgets while allowing the intended source-count changes. Model checkpoints and frozen source-readout heads are supplied separately in `SHoEbox_source_schedule_checkpoints.zip`.
