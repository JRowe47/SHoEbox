# SHoEbox: retained facts drive live composition

Completed November 5, 2026. Three seeds, one eight-node graph, separate retained memory and query workspace.

## Result

The integrated graph answers new two-step relation queries from a retained fact. At gap 16, mean accuracy is **99.54% for recall**, **98.31% after correction**, and **99.35% when a correction is withheld and the old fact remains authoritative**. The 90% task target passes in **3/3 seeds**.

Two query rounds reach **98.31%**, compared with **24.22%** for one round and **24.22%** for two rounds without delayed exchange. The retained fact supplies the query value; the existing regional experts perform both relation lookups.

| Training seed | Producer seed | Selected update | Recall | Correction | Withheld: old |
| --- | --- | --- | --- | --- | --- |
| 20261120 | 20261110 | 160 | 100.00% | 100.00% | 100.00% |
| 20261121 | 20261111 | 160 | 98.63% | 95.31% | 98.44% |
| 20261122 | 20261112 | 160 | 100.00% | 99.61% | 99.61% |

Each cell uses 512 fresh fact episodes with independently selected held-out relation-table pairs. Checkpoint selection uses validation only. The source producer checkpoints are the three previously published, uncalibrated selections; no producer seed is excluded.

## What changed

An eighth sparse node, F, owns its byte contextualizer and expert sharing bank. Its publication connects to A through an ordinary latent edge. The seven existing composition nodes, their relation pathways, and their common 256-byte decoder remain in the same engine. The runtime now supports independently owned source encoders and banks without changing the default shared configuration.

| Input | Information supplied | Role during a query |
| --- | --- | --- |
| F | Earlier fact observations and corrections | Retained latent supplies x; F is inactive |
| B | Current four-entry relation table f | First learned lookup |
| C | Current four-entry relation table g | Second learned lookup |
| P/Q | Regional latent publications | Existing common byte head emits g(f(x)) |

Only the new edge's **1,024 weights** are fitted. All producer and composer weights remain frozen. No fact is decoded to an intermediate byte at inference, and no fact label or task identifier enters the query. A fresh query clears the seven-node workspace while preserving F's contextual and recurrent state and its retained publication.

---

## Delays and causal controls

| Condition | Mean accuracy |
| --- | --- |
| Recall, gap 64 | 99.54% |
| Correction, gap 64 | 98.70% |
| Withheld correction, gap 64 | 99.35% |
| Correction, K=1 | 24.22% |
| Correction, K=2 | 98.31% |
| Correction, K=4 | 39.91% |
| K=2 without delayed exchange | 24.22% |
| F-to-A edge blocked | 26.37% |
| Memory reset before query | 26.37% |
| Memory shuffled: original target | 25.39% |
| Same weights, unconditional publication | 29.17% |
| Four consecutive live queries | 98.31% |

Shuffled memory scores **98.24%** against the answer implied by the substituted memory. Its entire latent publication and header are moved together. Blocked-edge, reset, shuffled, no-exchange and unconditional-publication controls have the same expert execution count as the intact K=2 condition. Four-value chance accuracy is 25%.

Four rounds reduce accuracy to **39.91%**. The combined model preserves the parent's learned two-round schedule; additional rounds are not yet a reliable refinement mechanism.

Gap 64 is four times the largest training gap. Across four consecutive queries, F's contextualizer state, expert recurrent state, payload, semantic header, version and provenance remain bit-for-bit unchanged. Each query receives fresh B/C tables; labels never enter its forward computation.

## Corrections after live queries

This sequence observes a fact, asks three independently drawn table queries, then continues the same producer state with a correction stream. A matched branch withholds the correction. Both branches receive the original fact again before the optional correction; this tests continued acquisition after queries, rather than a bare single-byte update.

| Sequence stage | Mean accuracy |
| --- | --- |
| Three queries before update | 99.54% |
| Query after correction | 98.11% |
| Query with correction withheld | 99.28% |

---

## Training and verification

Each seed receives 160 updates and 5,120 fresh fact episodes. Training alternates recall and correction over gaps 0, 4, 8 and 16. The objective combines final byte cross-entropy with local alignment of the edge output to the original query ingress representation of the observed fact byte. The alignment target comes from this project's own procedural observations and frozen ingress; it is used only in training.

| Seed | External events | Source occurrences | Expert executions | Training minutes |
| --- | --- | --- | --- | --- |
| 20261120 | 238,080 | 263,680 | 3,901,440 | 5.59 |
| 20261121 | 238,080 | 263,680 | 3,901,440 | 5.63 |
| 20261122 | 238,080 | 263,680 | 3,901,440 | 5.33 |

The graph has **317,098 unique parameters**, **480 physical experts**, eight nodes and four regions. A fact byte executes 12 experts; a fresh table pair plus a two-round query executes 264 experts. Producer computation is skipped during queries. Training time is concurrent CPU wall time excluding validation and audits.

All **381 regression tests pass**. The audit reconstructs all 480 training batches and table choices, checks checkpoint selection, verifies that only the new edge changed, and regenerates **54 held-out prediction groups**. Exported inference weights reproduce logits and complete state exactly. Optimizer continuation is tested bit-for-bit.

The original seven-node model still achieves **100% two-round composition on all 592 held-out cases**, with exact agreement in logits, state tensors and execution counts against the original source checkout.

## Scope and next work

This phase connects retained facts to live regional composition and measures the resulting dependence on memory, learned communication and recurrence. The fixed component weights make this an interface-integration experiment. It does not establish joint learning of all components, shared/private fact recovery, or natural-text learning in this combined graph; M7 remains open. The next step is to improve the weakest producer/connection pair through joint local learning while keeping these memory and composition controls.

## Reproduce

Extract both ZIPs into the same directory; each uses the SHoEbox/ root. Install requirements.txt, then run:

```bash
python scripts/audit_retained_composition.py --evaluations
python scripts/check_all.py
```

The artifact ZIP includes all four parent checkpoints, three resumable runs, selected models, inference exports, predictions, audits and the separate development pilot. The source ZIP is the exact published Git tree. Fresh training uses experiments.text_regions.retained_composition_train with the included parent checkpoints and the locked protocol.
