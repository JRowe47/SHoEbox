# SHoEbox: publication threshold calibration

Locked November 5, 2026, after the first two learned runs finished and before any test evaluation. The original selective-publication protocol and its zero-threshold results remain unchanged.

Seed 20261111 reaches 90.6-91.4% validation task accuracy with 99.2-99.6% write recall but only 80.9-82.5% write precision. Balanced write/hold BCE trains the gate with equal class weighting, while actual writes are rare. The follow-up calibrates its decision threshold using validation only, then tests the selected bias adjustment on the untouched test split.

For each validation-selected learned checkpoint, evaluate offsets 0, 0.5, 1, 1.5, 2, 2.5 and 3. An offset is subtracted from the gate bias; the runtime still uses its ordinary zero threshold. No expert, recurrent, ingress or decoder weight changes.

Use the same fixed 128 validation episodes per task (recall, correction, withheld correction, gap 16). Rank candidates by: all three task accuracies at least 90% and every task's write precision/recall at least 95%; minimum task accuracy; minimum write precision/recall; mean task accuracy; lower mean NLL; smaller offset. Save every candidate score. Test data never chooses the bias.

The isolated producer has no feedback edges, so its candidate latents and write logits do not depend on the retained publication. Cache these computed validation sequences once per task and simulate each threshold's publication replacements. Verify the selected threshold against a fresh complete engine run before exporting it.

Evaluate the calibrated export on the same nine 512-episode test conditions as the original learned checkpoint. Report the original zero-threshold model, calibrated model, and unconditional publication control separately. Preserve the source checkpoint and original inference export. The calibration export records the source checkpoint hash, bias offset and validation choice. Audit regeneration must reproduce the choice and all test predictions.
