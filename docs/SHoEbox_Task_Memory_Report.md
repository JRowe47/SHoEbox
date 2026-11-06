# SHoEbox: task-driven memory study

November 6, 2026. Source base `7707348`. Three paired seeds (17, 18, 19), 548,195 parameters, 4 owned publications, 16-byte gradient chunks. The integrated arm improves mean selected held-out task NLL versus both training controls.

## Integrated behavior

Ordinary bytes drive recurrent SHoE producers with separately owned state. A dense soft read over current publications plus a null option feeds the composer’s existing sparse routing and experts. Reads precede committed writes. The stream tests conflicting values, distractors, corrections and context-only questions; answers and memory-slot IDs never enter model input.

Task loss trains payloads, read heads and gates within chunks. Bounded conditional straight-through sensitivities carry later publication gradients across detach boundaries to the write-policy head; historical candidate/feature gradients remain truncated. Read heads learn at retrieval events. Replacement removes old responsibility; holds decay it by 0.995. This is not sampled policy-gradient estimation. Matched read-blocked forwards measure causal gain separately.

## Three-seed held-out means

| Arm | NLL | Accuracy | Exact | Long NLL | Long exact |
| --- | --- | --- | --- | --- | --- |
| Integrated | 0.524 | 78.9% | 69.9% | 0.557 | 66.2% |
| No delayed credit | 0.531 | 75.5% | 69.5% | 0.666 | 61.8% |
| No memory | 0.644 | 70.3% | 62.0% | 0.792 | 54.1% |

| Seed/data | Memory NLL gain | Delayed-credit NLL gain | Blocked delta |
| --- | --- | --- | --- |
| 17/123 | 0.281 | -0.041 | 0.444 |
| 18/124 | 0.076 | -0.023 | 1.147 |
| 19/125 | 0.002 | 0.086 | 2.246 |
| Mean | 0.120 | 0.007 | 1.279 |

Positive gains favor integrated training. Memory/credit gains are positive in 3/3 and 1/3 pairs. Final-budget mean NLL (integrated/no-credit/no-memory): 0.517/0.605/0.609. Each arm receives 600 updates and 153,600 bytes. Means weight seeds equally; the JSON preserves selected/final budgets and every per-seed result.

## Seed 17 pilot detail

| Arm | NLL | Exact | Category | Revised exact | Long exact | Blocked delta |
| --- | --- | --- | --- | --- | --- | --- |
| Integrated | 0.489 | 75.3% | 80.0% | 100.0% | 70.9% | 0.444 |
| No delayed credit | 0.448 | 74.1% | 73.3% | 100.0% | 64.0% | 1.593 |
| No memory | 0.770 | 49.4% | 66.7% | 75.0% | 46.5% | n/a |

NLL is nats per answer byte; accuracies use their respective query subsets. Each test arm scores 128 answers from 16 fresh independent streams. Positive blocked delta means memory helps. Interventions fork identical pre-query state at matched expert count. Long-delay evaluation multiplies declared delays by 3; new entity/attribute combinations are not measured by this pilot.

At the final matched budget, NLL is 0.489 integrated, 0.507 without delayed credit, and 0.643 without memory. Selected checkpoints minimize validation answer NLL. Integrated shuffled/stale deltas are 0.875/1.274 nats. Ordinary next-byte NLL is 0.869.

| Arm | Steps sel/final | Bytes sel/final | Expert calls | Seconds |
| --- | --- | --- | --- | --- |
| Integrated | 600/600 | 153,600/153,600 | 9,216,000 | 372.7 |
| No delayed credit | 500/600 | 128,000/153,600 | 9,216,000 | 375.6 |
| No memory | 300/600 | 76,800/153,600 | 9,216,000 | 376.6 |

Expert counts and time cover the final budget, including periodic validation time. Full curves and selected budgets are in summary.json.

---

## Seed 17 selectivity and correction

| Arm | Terminal null: exact | Terminal null: context | Writes | Read mass |
| --- | --- | --- | --- | --- |
| Integrated | 91.2% | 3.4% | 91.3% | 85.7% |
| No delayed credit | 81.7% | 1.4% | 96.2% | 71.2% |
| No memory | 21.1% | 17.0% | 51.0% | 0.0% |

Null mass averages the terminal-token no-read probability; writes count commits per owner-byte. Retrieval can occur during entity/attribute prefixes, so these terminal rates do not establish query-window relevance selection; detailed context-selective retrieval remains unestablished here. A positive shuffle penalty separately shows that publication content matters.

Seed 17 intact/single-query-blocked/full-blocked accuracy: 82.0%/66.4%/57.8%. Query publications average 1.03 bytes old with credit and 1.01 without: useful recurrent summaries are rewritten frequently. Long observation delays do not imply old persistent publications.

For integrated exact correction tasks, 16/16 revised fact versions receive a correct answer at a measured later query; 0 remain unresolved. Among resolved versions, the first success occurs at mean query opportunity 1.00 and 6.0 bytes after the correction value. This is sampled correction lag: unobserved recovery times are not inferred.

## Causal trace excerpts

First two eligible revisions, ordered by document/position; positions are zero-based bytes.

- Stream 0, A.x: observe '2' at byte 4; query at 32 predicts '1'. Correct to '3' at 52; query at 58 predicts '3' (reads blocked: '3').
- Stream 1, A.x: observe '2' at byte 4; query at 32 predicts '2'. Correct to '0' at 57; query at 63 predicts '0' (reads blocked: '0').

## Separate seed 17 write-cost followup

Observed 97.4% writing at update 100 triggered two additional paired arms with `0.02 * mean(write_probability)` added to the objective. Seeds, bytes, model and final compute match the primary cost-zero pilot.

| Arm | Step | NLL | Exact | Writes | Query age |
| --- | --- | --- | --- | --- | --- |
| Primary (cost 0) | 600 | 0.489 | 75.3% | 91.3% | 1.0 |
| Cost + credit | 400 | 0.585 | 61.7% | 37.9% | 1.0 |
| Cost, no credit | 600 | 0.514 | 71.6% | 39.9% | 1.3 |

Final-budget cost+credit/cost-without-credit NLL is 0.614/0.514; writes 42.2%/39.9%; query ages 1.0/1.3 bytes. Age is read-weighted time since publication, conditional on nonnull reads. Selected owner write ranges are 0.0%–71.8% and 33.4%–52.5%; occupancy/read fractions are in JSON. Lower writing alone does not establish useful selective retention; primary phase exits remain unchanged.

## Phase exits (three-seed means)

- Directional evidence (3/3 paired seeds support): Useful retained experience versus no-memory training and blocked reads.
- Unestablished: Detailed context-selective retrieval.
- Directional evidence (1/3 paired seeds support): Revision response above the no-memory control with positive blocked-read gain.
- Directional evidence (1/3 paired seeds support): Delayed credit improves selected task NLL over credit-disabled training.
- Directional evidence (3/3 paired seeds support): Memory improves longer-delay task NLL over no-memory training.
- Complete: three paired seeds; this does not imply three positive results. The 24.9M scale-up remains open.

Integrated state occupies 86,464 bytes for the full evaluation batch; eligibility traces occupy 240,640 bytes for the training batch. Seed 17 evaluation including intervention forwards took 2.83 seconds for 1,532 valid bytes (1,680 live bytes including padding; diagnostic execution counts excluded). Resumable checkpoints accompany the downloadable artifacts; JSON evidence is committed separately.

## Validation

All 5 regression suites passed (452 checks in that run). The final 23 task-memory checks also pass, including exact optimizer/stream/credit continuation and the conditional detached-gradient fixture at 1e-12 tolerance.
