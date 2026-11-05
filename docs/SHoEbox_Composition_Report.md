# SHoEbox: regional composition development

November 5, 2026

**The new regional composition experiment produces an initial positive result: K=2 reaches 33.78% held-out accuracy versus 26.35% at K=1.** Disabling delayed regional feedback reduces accuracy to 26.69%; shuffling the source relation reduces original-answer accuracy to 25.51%. The source tables remain fully recoverable before and after query processing. This phase trains the regional network itself with a simple common byte readout. Overall accuracy remains low, and the integrated M7 milestone remains open.

## What changed

The new task is `g(f(x))`: B observes a four-entry relation table f, C observes a separate table g, and A later receives query x. During encoding, only B and C execute. At query time all seven nodes execute; B and C receive a constant read cue, and only A receives x. P can see x and f in the first round. Q can see f and g but receives query-dependent information only in the second round.

The experiment retains the seven-node, three-region topology, sparse experts, recurrent state and persistent publications. An optional common affine head reads the full P/Q publications and predicts all 256 byte values. Older experiments retain their existing default readout. There is no symbolic answer path or frozen decoder fit in this experiment.

New code provides deterministic table-pair splits, graph training, K=1/2/4 evaluation from the same prefix state, source and feedback interventions, exact checkpoint continuation, and eleven automated checks. A training-only source reconstruction head can additionally teach B and C to retain their own observed input bytes; it is not used to produce answers at inference.

## Completed experiments

Both formal runs use seed 20261105, 1,000 optimizer steps, batches of 64, AdamW, LR 0.003, a 20-step LR warmup, gradient clipping at 1, and fresh graph weights. Each performs 64,000 training episodes and 16,896,000 expert executions. Validation selects the lowest answer NLL at K=2 after the curriculum. The held-out test set is used only after selection.

The base run uses 200 lookup warmup steps followed by 800 full-composition steps. The follow-up uses 100 copy steps, 200 lookup steps, then 700 full-composition steps, with source reconstruction weighted at 0.5. The follow-up changes both curriculum and local supervision; this is a development follow-up, not an isolated causal ablation.

Whole table pairs are held out: 296 training pairs, 132 validation pairs and 148 test pairs. All four queries for a pair stay together. Every individual table occurs in every split. The 592 original test queries are unique and answer-balanced. The meaningful task baseline is 25%, although the answer head scores all 256 bytes.

| Evaluation | Base run | Copy + reconstruction |
| --- | ---: | ---: |
| K=1 | 23.82% (141/592) | 26.35% (156/592) |
| K=2 | 24.66% (146/592) | 33.78% (200/592) |
| K=4 | 24.66% (146/592) | 29.22% (173/592) |
| K=2, delayed exchange off | 23.82% (141/592) | 26.69% (158/592) |
| K=2, state reset | 24.66% (146/592) | 24.32% (144/592) |
| K=2, shuffled B / original answers | 24.66% (146/592) | 25.51% (151/592) |
| K=2, shuffled B / new answers | 23.99% (142/592) | 31.25% (185/592) |

The selected checkpoints are step 1000 for the base run and step 1000 for the follow-up. K=2 answer NLL is 1.3878 and 1.2998 nats respectively; an exactly uniform four-answer distribution is 1.3863 nats. The follow-up gains 7.43 percentage points from K=1 to K=2, exceeding the specification's initial five-point development target. K=4 falls to 29.22%, so further rounds are not automatically helpful. This model was trained at K=2. The result is from one formal seed; the wider one-pass comparison and replications remain unfinished.

The shuffled control transforms the same 592 paired cases, producing 524 distinct replacement combinations. B actually changes in 552 cases and the implied answer changes in 432. Both original and counterfactual answers are scored. These transformed cases are not counted as additional independent test examples. In the base run, resetting memory or shuffling B changes none of the K=2 predicted bytes. In the follow-up, those interventions change 406 and 292 predicted bytes respectively. The feedback, shuffle and reset controls support a real contribution from retained source information and regional exchange. Counterfactual scoring rises to 31.25% for the new relation, consistent with partial adaptation to that change. The task is still far from mastered.

## What did learn

The follow-up's existing local reconstruction head was scored without refitting on all test cases, before and after query processing. Each source table contains four entries. The entry and whole-table scores below measure retained source information, not successful composition. The individual tables are familiar; their combinations are held out.

| Source state | Initial entry accuracy | Selected entry accuracy | Selected whole-table accuracy |
| --- | ---: | ---: | ---: |
| B, after prefix | 21.96% | 100.00% | 100.00% |
| C, after prefix | 25.17% | 100.00% | 100.00% |
| B, after query | 22.80% | 100.00% | 100.00% |
| C, after query | 26.18% | 100.00% | 100.00% |

An exploratory copy curriculum reached 100% training-batch accuracy, but the later lookup and composition stages fell back toward chance. A separate exploratory reconstruction run improved local retention and lookup learning, motivating the formal follow-up. Those exploratory runs do not count as independent replications of the formal result. The intermediate copy checkpoint was not retained, so the copy result is a training-curve observation only.

The concrete remaining gap is reliable query-driven use of the retained relations. The common answer path now benefits from regional exchange, but it does not yet compose consistently. On the 48 held-out cases where one table is the identity, the follow-up answers 21 correctly (43.75%). A strong single-lookup capability is therefore still an appropriate next training target.

## Verification and execution accounting

All **328 tests pass** across all five experiment suites, including eleven composition checks. They verify split separation, no answer leakage into input tensors, Q's first-round information boundary, second-round query dependence, immutable shared prefix state, gradients through source encoding and delayed edges, actual expert counts, and exact model/optimizer/RNG continuation including the auxiliary head. Saved predictions independently reproduce all fourteen reported accuracy groups.

| Query rounds | Prefix expert calls/case | Query calls/case | Total calls/case |
| --- | ---: | ---: | ---: |
| K=1 | 96 | 84 | 180 |
| K=2 | 96 | 168 | 264 |
| K=4 | 96 | 336 | 432 |

External input is eleven bytes per case at every K. The inference graph has 252,478 parameters and 384 physical experts; local reconstruction adds 528 training-only parameters. These counts describe work performed. This phase does not establish a speed advantage or optimize kernels.

Two additional recurrent seeds and a wider one-pass control were stopped at checkpoints 200, 50 and 250 respectively when the initial learning failure became clear. They are preserved as partial runs and are excluded from completed comparisons. The wider control implementation matches K=2 query expert calls but uses more prefix work; it remains part of the next acceptance work.

## Next development target

First establish reliable single-table lookup through the same regional publications and common answer head. Use an explicit learning threshold before advancing the curriculum, then teach two-table composition while checking that lookup remains intact. The present fixed 200-step lookup stage advanced before lookup was mastered. Retain the same held-out table-pair protocol and K/feedback/source controls.

As the composition signal strengthens, complete the paired wider one-pass comparison and independent seeds, then integrate composition with the existing memory/correction and shared/private tasks. Profiling and streamlining follow that functional result. The current evidence supports keeping the architecture and focusing development on learned relational use.

## Reproduction and contents

The repository's `docs/SHoEbox_Composition_Protocol.md` gives the complete training and evaluation commands. Results, raw predictions, source hashes, histories and the retention diagnostic are under `experiments/text_regions/results/composition`. The attached source archive contains the current implementation. The artifact archive contains initial, validation-selected and final tensors for both formal runs, partial-run checkpoints, exploratory histories/scripts, test logs and a source snapshot for the older base runtime. No earlier phase's large checkpoint collection is claimed to be included.

The formal base runtime is GitHub commit `01a6c492307b8ab4349a2815cd88c93cec519929`; the reconstruction runtime is `cd12b0c3682a62ff52d9f43d4d44fd9b7e4aaa5b`. Each checkpoint embeds its protocol and runtime file hashes. Training used PyTorch 2.15.0+cpu with two Torch threads per run. Wall times in the histories include contention from concurrent tests and experiments and are not performance benchmarks.
