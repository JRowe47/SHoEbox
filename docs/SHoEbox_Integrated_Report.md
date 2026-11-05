# SHoEbox: integrated capability training

November 5, 2026 | Completed development round | 349 automated checks passed

The existing regional composition model now has a resumable joint trainer for composition, delayed correction, and shared/private facts. The controlled gradient follow-up improves held-out composition to **99.83% (591/592)**, versus 94.43% at the same training budget. It preserves all composition validation gates at updates 50 and 100. **The new memory and fact tasks remain near chance; joint capability mastery is still open.**

The validation-only selection rule chooses the longer baseline at update 200 as the integrated candidate. It scores **97.97% (580/592)** on composition. Its weakest validation capability is higher than the balanced model's, although neither meets the new capability gates. Both endpoints and the matched baseline update 100 are retained and evaluated.

## What was built

One seven-node graph shares all learned parameters and the common 256-byte output head across three objectives. The graph retains 252,478 inference parameters and 384 physical experts. Each task has its own runtime state, while the optimizer updates the shared graph. Composition episodes, a correction stream segment, and a factorized-view segment contribute to every update.

Atomic checkpoints save weights, AdamW moments, both carried stream states and cursors, sampler/Torch RNGs, validation history, counters and source hashes. Exact continuation is tested for both gradient rules. Checkpoint selection protects the existing composition gates and then prioritizes the weakest remaining capability. The evaluator runs the same selected graph through all tasks and causal controls.

The composition source-reconstruction and regional-supervision heads remain training-only. Removing both leaves identical inference logits and recurrent states on the export parity checks. The shared inference graph receives no additional task-specific head or capacity in this round.

## Experiment and matched intervention

Both runs start from the selected second-lookup checkpoint, which previously scored 95.78% on the 592-case composition test. Its weights, optimizer moments and RNG state are preserved. The baseline averages three task losses before the final norm-1 gradient clip. It completes 200 updates at constant learning rate 0.0005.

A next-batch diagnostic at baseline update 25 measures gradient norms of 14.09 for composition, 813.70 for correction and 111.14 for views. This motivates one controlled change: clip each unweighted task gradient to norm 1 before averaging, retaining the final global clip. The follow-up runs for 100 updates and is compared with the preserved baseline update-100 checkpoint. It was declared from training/validation evidence before held-out test scoring.

Each update uses 64 composition cases and one 16-event segment from each of two batch-16 streams. Correction uses gap 16 and intermediate query markers every four bytes; views use gap 8 and balanced supported source subsets. All query/stream processing uses K=2. Composition independently encodes B and C at K=1 before its K=2 query. Stream state carries across segments and detaches for truncated backpropagation.

| Training exposure | Each 100-update arm | Baseline 200 |
|---|---:|---:|
| Expert executions | 9,084,288 | 18,168,576 |
| Aligned external events | 76,016 | 152,032 |
| Available source occurrences | 130,402 | 260,312 |
| Composition cases | 6,400 | 12,800 |
| Correction final queries | 528 | 1,056 |
| View final queries | 800 | 1,600 |
| Correction supervised queries, including intermediate | 4,784 | 9,584 |
| View supervised queries, including intermediate | 4,000 | 8,000 |

The comparison audit verifies identical initialization, first-step losses, sampled examples, stream cursors, RNGs and every per-update execution budget across the two 100-update arms. The 200-update endpoint uses additional compute. Logged training-update time is 62.93 minutes for the baseline and 29.59 minutes for the follow-up; the runs overlapped, so these are execution records rather than an isolated speed comparison.

## Composition results

The test contains all 592 queries from held-out whole pairs of four-entry relation tables. B stores f, C stores g, and A supplies x; the required answer is g(f(x)). The local lookup subsets contain one identity relation. P's first lookup and Q's second lookup are scored through the existing training-only diagnostic head. Ordinary answers always use the largest of all 256 output logits.

| Held-out score | Baseline 100 | Balanced 100 | Baseline 200 |
|---|---:|---:|---:|
| Overall composition benchmark | 94.43% | 99.83% | 97.97% |
| Two nonidentity lookups | 94.49% | 99.82% | 97.79% |
| B-only lookup subset | 85.00% | 100.00% | 100.00% |
| C-only lookup subset | 100.00% | 100.00% | 100.00% |
| P first lookup | 97.80% | 100.00% | 98.99% |
| Q second lookup | 95.10% | 99.83% | 98.31% |

At the matched 100-update budget, per-task gradient clipping improves overall composition by **5.41 percentage points**. The balanced model's validation composition rises from 99.24% at update 50 to 100.00% at update 100, with every local-lookup and exact-retention gate passing at both checks. This establishes consecutive composition preservation under joint training; the combined-task stability gate remains unmet.

| Exact complete-table retention | Baseline 100 | Balanced 100 | Baseline 200 |
|---|---:|---:|---:|
| B before query | 100.00% | 100.00% | 100.00% |
| C before query | 100.00% | 100.00% | 100.00% |
| B after query | 99.83% | 100.00% | 100.00% |
| C after query | 100.00% | 100.00% | 100.00% |

These retention scores require the entire four-entry table to be reconstructed correctly in each case, before and after the query. They are stricter than individual-entry accuracy.

## New capability results

Recall/correction use 192 fresh test episodes per task. Shared/private all-source tests use 192 episodes per query on held-out fact combinations. Single-source tests use a separate complete-factorial cohort of 192 episodes per query, preventing absent facts from being inferred from the restricted test split. The four-answer reference rate is 25%; prediction still considers all 256 bytes.

| Held-out task | Baseline 100 | Balanced 100 | Baseline 200 |
|---|---:|---:|---:|
| Recall, gap 16 | 24.48% | 19.79% | 30.73% |
| Correction, gap 16 | 25.00% | 20.83% | 32.29% |
| Shared fact, all sources | 25.00% | 26.56% | 30.21% |
| Left-private fact, all sources | 29.69% | 26.56% | 20.31% |
| Right-private fact, all sources | 22.40% | 27.08% | 20.83% |
| A only: shared | 23.96% | 23.44% | 20.83% |
| A only: left private | 20.31% | 25.00% | 28.12% |
| B only: shared | 27.08% | 26.04% | 25.00% |
| C only: shared | 22.92% | 21.88% | 29.17% |
| C only: right private | 27.60% | 18.75% | 28.12% |

The current joint objective has not taught reliable delayed correction or factorized fact recovery. Gradient balancing protects composition but does not unlock these tasks at this budget. The earlier successful specialist probes used separate trained checkpoints; those weights were not merged into this composition parent.

## Causal controls

Every composition row uses the same 592 cases. The shuffled-B intervention changes the observed first table; its counterfactual target is recomputed from that changed table and the original C table. The same predictions are scored against both targets.

| Composition intervention | Baseline 100 | Balanced 100 | Baseline 200 |
|---|---:|---:|---:|
| K=1 | 25.00% | 25.00% | 25.00% |
| K=2 | 94.43% | 99.83% | 97.97% |
| K=4 | 43.07% | 47.47% | 42.40% |
| K=2, delayed exchange off | 25.00% | 25.00% | 24.83% |
| K=2, state reset | 22.47% | 25.00% | 23.31% |
| Shuffled B, original target | 26.52% | 27.03% | 26.35% |
| Shuffled B, counterfactual target | 95.27% | 100.00% | 98.31% |

The balanced model falls from 99.83% to 25.00% when delayed exchange is disabled, and reaches 100.00% against the changed-table counterfactual targets. Its answers therefore depend on the exchanged source evidence. Shuffling can produce table pairs from any split. Increasing to K=4 drops accuracy to 47.47%, so extra-round stability remains unresolved.

K=2 uses 264 expert executions per composition case: 96 for independent source encoding and 168 for the query. K=1 uses 180 and K=4 uses 432. Removing exchange, resetting memory, and shuffling B preserve the K=2 execution budget. Raw records retain predictions, targets, case identities and counts for every intervention.

| Selected baseline 200: memory | Carried | Query reset | No update: old value |
|---|---:|---:|---:|
| Recall, gap 16 | 30.73% | 28.12% | n/a |
| Correction, gap 16 | 32.29% | 22.40% | 25.52% |
| Recall, gap 32 | 33.85% | 28.12% | n/a |
| Correction, gap 32 | 29.17% | 22.40% | 28.65% |

Withholding an update is scored against the old value in the final column. Memory controls and longer-delay scores do not establish useful correction behavior. Full independent-view controls include all sources, A/B/C only, no private facts and query reset, with identical expert budgets.

## Selection, verification and artifacts

Validation covers all 528 composition cases and fixed 64-episode cohorts for each new task. Full qualification requires 90% answer/lookup/new-capability accuracy and 100% complete source retention. Selection ranks full qualification first, then composition preservation, then minimum capability accuracy, macro accuracy and macro answer NLL. No checkpoint passes full qualification. Baseline 200 wins the declared cross-run selection because its minimum validation accuracy is 23.44%, versus 18.75% for balanced 100. Held-out results never choose the checkpoint.

All 349 automated checks pass, including exact interrupted/resumed training and identical first-update observations, costs and losses across gradient policies. Training audits reproduce parent initialization, source hashes, update counters and checkpoint selection. Inference exports match composition logits on 64 cases and logits/state on eight correction and eight view cases per model. The evaluation audit independently reconstructs **114 prediction groups**, checks generated targets and identities, and verifies matched intervention costs.

The downloadable source archive matches the published Git tree. The artifact archive includes the parent, initial/final/selected checkpoints, matched update-100 snapshot, diagnostic update-25 checkpoint, optimizer/runtime states, inference exports, complete raw evaluations, protocols, histories, audits and a SHA-256 manifest. Both archives extract under the same SHoEbox directory.

| Artifact path | Purpose |
|---|---|
| artifacts/integrated/selected.pt | Validation-selected integrated candidate, baseline 200 |
| artifacts/integrated/joint/inference.pt | Its inference-only graph |
| artifacts/integrated/balanced/inference.pt | Balanced 100 inference-only graph |
| artifacts/integrated/baseline_step100.pt | Matched baseline endpoint |
| artifacts/integrated/selection.json | Validation selection and comparison audit |
| artifacts/integrated/evaluation_audit.json | Raw prediction, label and cost audit |

The parent checkpoint SHA-256 is `f0abd8d038d5c7dc047d2d3dd4d7a81794bec003212727f92407caa379f80ed8`. Exact per-file hashes and the published commit/tree are included in the archives. Training and evaluation used Python 3.12.14, PyTorch 2.15.0 CPU and one Torch thread per process.

## Next development target

M7 remains open. The next focused change is task-local regional supervision for retained facts and corrections, using the successful composition gates as retention constraints. Composition already benefits from explicit learning signals at the regions performing its intermediate work; the new tasks currently depend on query losses plus routing auxiliaries. This round supplies the matched trainer, controls and resumable checkpoints needed to test that change directly.

Reproduction commands are in docs/SHoEbox_Integrated_Protocol.md and experiments/text_regions/results/integrated/README.md. Repository: [JRowe47/SHoEbox](https://github.com/JRowe47/SHoEbox).
