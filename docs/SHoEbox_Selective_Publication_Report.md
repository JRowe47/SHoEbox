# SHoEbox: learned publication retention

Completed November 5, 2026. Three paired seeds, extracted sparse producer, common byte decoder.

## Result

The task-accuracy target passes in 3/3 learned-publication runs; the write precision/recall target passes in 2/3.

The main gain appears beyond the training delay. At gap 64, learned publication reaches **99.09% recall versus 68.55%** for unconditional publication. When a correction is withheld, it retains the old value at **99.35% versus 71.09%**. All three seed pairs improve on each longer-delay task.

At the training-range gap 16, mean accuracy across recall, correction and withheld correction is **98.98% with learned publication** and **99.76% with unconditional publication**. Each run uses 240 updates and 7,680 fresh training episodes.

The mechanism adds a learned write/hold decision to a node's existing output. On hold, the runtime preserves its payload, semantic header, validity, version and provenance. The node still executes its contextualizer, routers, sparse experts and recurrent-state update. The decision reads only the computed latent.

| Seed | Publication | Selected update | Recall | Correction | Withheld: old |
| --- | --- | --- | --- | --- | --- |
| 20261110 | always | 240 | 100.00% | 100.00% | 100.00% |
| 20261110 | learned | 240 | 100.00% | 100.00% | 100.00% |
| 20261111 | always | 180 | 100.00% | 99.61% | 100.00% |
| 20261111 | learned | 240 | 97.85% | 94.92% | 98.24% |
| 20261112 | always | 180 | 100.00% | 98.24% | 100.00% |
| 20261112 | learned | 240 | 99.80% | 100.00% | 100.00% |

Each cell uses 512 fresh test episodes at gap 16. Selection uses validation only: minimum accuracy, then mean accuracy, then lower NLL. The 90% task threshold applies independently to each condition and seed. Final-endpoint results are also saved, regardless of selection.

The producer has **71,867 parameters**, including a **33-parameter gate**, 96 physical experts, and 12 expert executions per byte. Both paired arms have the same gate and losses; unconditional publication overrides only the write decision. Both inherit the same source-A computation from the selected gradient-horizon checkpoint and initialize a fresh 256-byte decoder. This isolates publication behavior within the producer task.

---

## Controls and learned decisions

The following accuracies are means of the three seeds. The last column changes only the selected learned model's runtime publication policy; weights remain fixed.

| Task, gap 16 | Trained always | Learned | Learned weights, forced always |
| --- | --- | --- | --- |
| Recall | 100.00% | 99.22% | 31.90% |
| Correction | 99.28% | 98.31% | 35.87% |
| Withheld update: old value | 100.00% | 99.41% | 32.81% |

| Seed | Minimum write precision | Minimum write recall | 95% target |
| --- | --- | --- | --- |
| 20261110 | 99.90% | 100.00% | pass |
| 20261111 | 80.57% | 100.00% | fail |
| 20261112 | 99.61% | 100.00% | pass |

Write scores above take the minimum over recall, correction and withheld correction at gap 16. The data contains bare A-D distractors, invalid '=x' strings, and independently randomized write timing. The model must recognize an observed write in context. Classification labels are never passed to inference.

| Control / extrapolation | Trained always | Learned |
| --- | --- | --- |
| Recall, gap 64 | 68.55% | 99.09% |
| Correction, gap 64 | 93.03% | 99.28% |
| Withheld update: old value, gap 64 | 71.09% | 99.35% |
| Reset before query | 24.35% | 23.63% |
| All writes withheld | 24.28% | 23.57% |
| Source absent | 25.00% | 23.63% |

Gap 64 exceeds the largest training gap, 16. Withheld corrections are scored against the old observed value. Reset clears the complete graph state before the final query. Source absence masks every input occurrence. Four-value chance accuracy is 25%.

All **816,261 measured hold transitions** across the carried/withheld gap-16 and gap-64 conditions preserve the latent payload bit-for-bit. Unit tests separately verify header and provenance retention, inactive rows, resets and serialization.

---

## Validation-only threshold follow-up

Seed 20261111 exposed false writes on distractors despite high training accuracy. Before viewing any test results, a separate protocol locked a small threshold grid: offsets 0, 0.5, 1, 1.5, 2, 2.5 and 3 subtracted from the trained gate bias. All expert, recurrent, ingress and decoder weights remain unchanged.

Validation ranks whether the declared task and decision thresholds pass, then minimum task accuracy, minimum decision precision/recall, mean task accuracy, lower NLL and smaller offset. The original fixed-budget results on the preceding pages are unchanged.

| Seed | Bias offset | Recall | Correction | Withheld: old | Write precision / recall |
| --- | --- | --- | --- | --- | --- |
| 20261110 | 0.0 | 100.00% | 100.00% | 100.00% | 99.90% / 100.00% |
| 20261111 | 2.5 | 98.05% | 95.51% | 98.44% | 84.07% / 100.00% |
| 20261112 | 0.0 | 99.80% | 100.00% | 100.00% | 99.81% / 100.00% |

These are test scores after validation-only selection. **2/3 calibrated runs** meet all three task thresholds and all three conditions' decision thresholds. The full nine-condition evaluations are included in the raw results. Calibration does not add parameters or an inference pass.

The selected offset is checked against the full engine. Because this isolated producer has no feedback edges, candidate latents and write logits are independent of retained publications. Validation computes those sequences once and checks the seven offsets by applying ordinary publication replacements. The audit independently regenerates the selection and exported tensors.

This follow-up distinguishes a misplaced decision threshold from an incorrect learned write classification. Highly confident false writes can survive a threshold change; improving that distinction requires further producer learning.

---

## Training, verification and scope

Each update uses a fresh batch. Loss combines candidate-value decoding at writes, retained-value decoding after observation, balanced write/hold classification, and the existing router regularizers. All retained labels follow the actual observed stream. Gradients detach every 16 events; held publication values persist. There is no straight-through gradient through the hard write decision.

| Arm | Events per run | Expert calls per run | Mean training minutes |
| --- | --- | --- | --- |
| always | 318,720 | 3,824,640 | 17.56 |
| learned | 318,720 | 3,824,640 | 16.66 |

Training time is wall time during concurrent CPU runs, excluding validation and audits. Gate computation and parameter counts are matched; expert counts do not imply equal wall time.

All **373 regression tests pass**. The audit reconstructs all 1,440 training batches, verifies matched initial weights and exposure in each pair, recomputes validation selection, regenerates **108 test prediction groups**, and compares inference exports against their selected checkpoints. Resume tests reproduce the next optimizer update and moments exactly.

The original seven-node checkpoint still achieves **100% two-round composition on 592 held-out cases**. Running it through the old and new engine gives identical logits, carried state tensors and execution counts. Its weights are unchanged, and selective publication remains disabled by default.

This phase implements and tests one producer's write/retain interface, including seed sensitivity. It does not combine the trained producer with the seven-node composition model. **M7 remains open.** The next work is to stabilize write discrimination across seeds and then integrate retained fact publications with current query signals. A single held publication cannot simultaneously serve as an unchanged fact and a newly arriving query signal; that interface distinction needs to be explicit in the integration experiment.

## Reproduce

Extract both ZIPs into the same directory; each uses the SHoEbox/ root. Install requirements.txt, then run:

```bash
python scripts/audit_selective_publication.py --evaluations
python scripts/check_all.py
```

For fresh runs, use the locked protocol and scripts/run_selective_publication.py. The artifact ZIP includes the parent, six complete optimizer checkpoints and selected models, inference exports, histories, predictions, audits, compatibility evidence and the separate development pilot. The source ZIP contains the exact published Git tree.
