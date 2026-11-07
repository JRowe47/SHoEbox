# SHoEbox handoff

## Completed architecture: learnable recall workspace

- Base `1cd759a`; adds `RecallWorkspace` in `task_recall.py`.
- Configurable item-or-null slots, learned overlap penalties, role-specific
  composer input maps, composer-to-request feedback and usage-to-write feedback.
- The new continuation preset enables two distinct slots. Zero slots preserves
  the legacy reader, common initialization and caller RNG. Seven slots are supported.
- All new controls receive live task gradients. Existing conditional delayed
  credit reaches publication admission through the reads. Historical request and
  recurrent-feedback derivatives are not reconstructed across detached windows.
- Query audits include pooled-layout and feedback-off interventions, with complete
  per-slot selections alongside aggregate mean read mass.
- All 520 checks across five suites passed, including eight new recall checks.
  The live-output example also passed.

## Completed training checks

- Two matched 160-update continuations from the same 600-update parent, with all
  weights trainable and fresh optimizer/runtime state. Each has 564,519 parameters,
  81,920 training bytes and 4,915,200 expert executions. Initialization seed 59,
  data seed 259; selected checkpoint at continuation step 120 for both.
- Distinct versus pooled exact-answer NLL: selected test 1.1609 vs 1.1382;
  final test 1.2560 vs 1.1374; selected long-delay 1.1579 vs 1.1301;
  final long-delay 1.2159 vs 1.1206. The pooled arm wins this short continuation.
- Distinct slot input maps specialize (deviation from their mean reaches 0.591);
  pooled maps remain identical. Feedback learns and affects answers, with mixed
  intervention effects. All eight checkpoint/split audits reproduce live answers;
  maximum logit drift 4.7684e-6, zero argmax changes.
- Seven-slot/nine-owner integration: 600,354 parameters, 32 fresh updates,
  4,096 bytes, 491,520 expert executions. Producer-state holds and delayed payload
  credit are enabled together. Final validation averages 5.73 active slots and
  4.13 distinct selected owners; all three feedback projections learn.
- [Report](docs/SHoEbox_Recall_Report.md), [PDF](docs/SHoEbox_Recall_Report.pdf),
  [metrics](experiments/text_regions/results/task_recall/summary.json).
- Source, common parent, selected/final and seven-slot stage checkpoints,
  optimizer/runtime state, raw traces, audits, validation logs and hashes are in
  `SHoEbox_Recall_Artifacts.zip`.

## Development direction and next target

Retain well-motivated mechanisms with trainable controls and explicit ablations.
Small probe losses are not feature-removal gates. Code comments mark distinct
roles and feedback as potentially training-threshold-dependent features. At fixed
selected content, distinct role maps can preserve an ordering that early pooling
erases; this capability is verified directly.

Next, reuse the workspace for ordinary region-to-region recall and train longer
on mixed language and memory tasks. Carry state holding, payload credit, role
separation and feedback forward as independently controllable mechanisms. Use
broader ablations after specialization to decide which mechanisms earn their cost.
No replay store is added; recall slots are current views of owned publications.

## Reproduction

```bash
python scripts/run_recall.py --parent /path/to/parent.pt --out artifacts/recall/study
python scripts/summarize_recall.py --root artifacts/recall \
  --out experiments/text_regions/results/task_recall/summary.json
python scripts/check_all.py
```

The paired runner resumes existing checkpoints. Use the archived seven-slot
invocation for its separate integration check. `--recall-layout pooled`,
`--no-recall-feedback`, and `--no-recall-hard` expose later ablations.

Prior passes: [state commits](docs/SHoEbox_State_Commit_Report.md) (`483eb11`
implementation, `1cd759a` results) and [payload credit](docs/SHoEbox_Payload_Credit_Report.md)
(`a7a14ab` implementation, `150c8a0` results).
