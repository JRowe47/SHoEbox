# SHoEbox: gradient horizon

Completed November 5, 2026. Matched complete-episode training from the regional-facts control checkpoint.

## Result

Removing the 16-event gradient cut did not solve integrated memory or fact learning. The full-horizon endpoint preserves 100.00% composition, compared with 98.82% for the truncated arm. Shared-fact accuracy rises from 30.47% to 34.57%, while delayed correction falls from 39.26% to 29.69%.

| Held-out task | Truncated | Full episode | Change |
| --- | --- | --- | --- |
| Composition, K=2 | 98.82% | 100.00% | +1.18 pp |
| Recall, gap 16 | 39.26% | 38.09% | -1.17 pp |
| Correction, gap 16 | 39.26% | 29.69% | -9.57 pp |
| Shared fact | 30.47% | 34.57% | +4.10 pp |
| Left-private fact | 26.76% | 29.30% | +2.54 pp |
| Right-private fact | 29.69% | 27.54% | -2.15 pp |
| A-only shared | 32.81% | 34.77% | +1.95 pp |
| A-only left-private | 35.94% | 34.96% | -0.98 pp |
| B-only shared | 26.17% | 23.83% | -2.34 pp |
| C-only shared | 26.17% | 26.95% | +0.78 pp |
| C-only right-private | 26.17% | 25.98% | -0.20 pp |

Both arms complete 32 optimizer updates, with 2,048 composition cases and 512 complete episodes of each stream task. The sole paired intervention is whether gradients cross the 16-event chunk boundary. Each held-out stream condition/query uses 512 fresh episodes; composition uses all 592 held-out cases. Predictions use the common 256-byte output head. Four-value chance accuracy is 25%.

Validation selects **full/best.pt at phase update 32**. Endpoint scores above are reported regardless of checkpoint selection. Test scores never select a checkpoint.

---

## Validation and composition

| Arm | Update | Composition | All composition gates | Weakest task |
| --- | --- | --- | --- | --- |
| truncated | 0 | 100.00% | pass | 20.31% |
| truncated | 16 | 99.62% | pass | 21.88% |
| truncated | 32 | 98.67% | pass | 15.62% |
| full | 0 | 100.00% | pass | 20.31% |
| full | 16 | 100.00% | pass | 20.31% |
| full | 32 | 100.00% | pass | 23.44% |

Selection ranks all capability gates, composition preservation, minimum task accuracy, macro accuracy and lower macro NLL, in that order. Composition preservation includes both local lookups and exact table retention, as well as the final answer. A high answer score alone does not pass all gates.

| Composition control | Truncated | Full episode |
| --- | --- | --- |
| One round | 25.00% | 25.00% |
| Two rounds | 98.82% | 100.00% |
| Four rounds | 43.24% | 40.54% |
| Exchange disabled | 25.00% | 25.00% |
| Changed table, original target | 27.20% | 27.03% |
| Changed table, changed target | 98.99% | 100.00% |
| State reset before query | 22.80% | 25.17% |

Both arms preserve the seven-node graph, sparse expert banks, recurrent state format and common byte decoder. They inherit identical weights, optimizer moments and composition RNG. New stream episodes start from reset state with seed 20261109; partial parent stream states are intentionally discarded in both arms.

This pair also changes episode loss weighting and optimizer frequency relative to the preceding phase. Those changes are shared by both arms. The paired comparison isolates gradient detachment; comparisons to the earlier phase do not isolate it.

---

## Delayed memory and write retention

| Memory condition | Truncated | Full episode |
| --- | --- | --- |
| recall, 16, carried | 39.26% | 38.09% |
| recall, 16, reset | 24.80% | 24.80% |
| correction, 16, carried | 39.26% | 29.69% |
| correction, 16, reset | 24.61% | 24.61% |
| withheld correction, 16, old value | 35.55% | 31.25% |
| recall, 32, carried | 33.20% | 30.08% |
| recall, 32, reset | 24.80% | 24.80% |
| correction, 32, carried | 27.15% | 28.32% |
| correction, 32, reset | 24.61% | 24.61% |
| withheld correction, 32, old value | 33.59% | 28.32% |

Gap 16 is the training setting; gap 32 tests a longer delay. Query reset removes carried state immediately before the answer. Withheld corrections are scored against the old observed value here; raw records also include agreement with the unseen new value.

| Local memory head | Truncated: write / query | Full: write / query |
| --- | --- | --- |
| recall: P | 17.19% / 29.69% | 60.94% / 28.12% |
| recall: Q | 48.44% / 35.94% | 28.12% / 29.69% |
| correction: P | 35.94% / 28.12% | 37.50% / 29.69% |
| correction: Q | 35.94% / 32.81% | 26.56% / 31.25% |

Local diagnostics use frozen training-only heads on 64 held-out episodes per task after selection. They measure whether regional publications contain readable values at the observed write and the final query. They do not supply inference answers.

---

## Fact acquisition and verification

| Local fact head | Truncated: write / query | Full: write / query |
| --- | --- | --- |
| P: shared | 31.25% / 31.25% | 23.44% / 20.31% |
| Q: shared | 28.12% / 31.25% | 18.75% / 20.31% |
| P: left | 37.50% / 26.56% | 43.75% / 28.12% |
| Q: right | 23.44% / 25.00% | 26.56% / 32.81% |

| Training budget | Truncated | Full episode |
| --- | --- | --- |
| Expert executions | 6,561,792 | 6,561,792 |
| External events | 46,080 | 46,080 |
| Observed source occurrences | 68,536 | 68,536 |
| Composition cases | 2,048 | 2,048 |
| Complete correction episodes | 512 | 512 |
| Complete fact episodes | 512 | 512 |
| Training time, minutes | 28.67 | 28.66 |

All **360 regression tests** pass. Tests cover forward/state/cost equality, full-gradient equivalence to an unsliced episode in FP64, final-query credit reaching an earlier write only with the full horizon, loss-only labels and exact interrupted/resumed updates. FP64 avoids cancellation in the untrained fixture's large normalization gradients; training uses FP32.

The audit reconstructs each training batch from the generators, compares all data hashes and budgets, verifies parent weights and optimizer moments, and recomputes validation selection. It also verifies 76 held-out prediction groups and local diagnostic arithmetic. Exports remove all four auxiliary heads and reproduce checked inference logits and states exactly; the inference graph has 252,478 parameters.

## Next development step

Focus on fact acquisition and selective publication inside the regional experts: teach a producer to recognize an observed write, preserve its payload while processing distractors, and replace it when a correction arrives. Test that learned write/retain decision with the existing reset, withheld-update and source-retention controls, while preserving successful two-round composition.

M7 remains open. The integrated milestone also requires multi-seed replication and the natural-text experiment.

## Reproduce

Extract the source and artifact ZIPs into the same directory. Both use the SHoEbox/ root. Install requirements.txt, then run:

```bash
python scripts/audit_gradient_horizon.py --evaluations
```

The locked protocol contains fresh-run commands. Append --resume to the matching training command to recover a partial run. The artifact archive includes the parent, complete optimizer checkpoints, validation endpoints, inference exports, per-update histories, predictions, diagnostics and a SHA-256 manifest identifying the published source commit and tree.
