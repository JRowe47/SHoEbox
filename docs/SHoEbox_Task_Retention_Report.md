# SHoEbox: persistent-publication retention

November 6, 2026. Source base `fd4d1ac`; seeds 29/229. Trained fresh recurrence performs better on exact questions: 1.257990 NLL versus 1.275528 competitive publications. The intended retention advantage is not established. Scratch and common-parent continuation cohorts are reported separately.

## Changes and comparison

The write policy can condition on the current owned publication, validity, age and candidate similarity. Competitive writes choose one owner or none; cross-owner delayed derivatives follow that competition. Generic startup fills each of four invalid owners once. These forced commits receive no learned gate credit. Common initialization is preserved; conditioned arms have 549,059 parameters, and legacy has 864 fewer.

Four recurring entities receive changing values, questions, corrections and updates irrelevant to the current question but useful later. Delays are 2/48 bytes; longer-delay evaluation triples them. Inputs contain no answers, slot IDs or task labels. The fresh arm is trained separately to read current expert candidates, while retaining producer/composer recurrence and the same sparse execution budget.

## Scratch cohort: selected and final tasks

| Arm | Selected step | Exact NLL | Exact accuracy | Final exact NLL | Long exact NLL |
| --- | --- | --- | --- | --- | --- |
| Legacy | 700 | 1.368 | 33.6% | 1.351 | 1.379 |
| Conditioned | 800 | 1.359 | 40.7% | 1.359 | 1.301 |
| Competitive | 800 | 1.356 | 33.6% | 1.356 | 1.359 |
| No delayed credit | 800 | 1.397 | 35.4% | 1.397 | 1.404 |
| Trained fresh | 800 | 1.361 | 26.5% | 1.361 | 1.381 |

NLL is nats per scored answer byte. Checkpoints minimize validation all-answer NLL; exact-value NLL is the declared retention comparison. Semantic/context scores and both complete checkpoints’ evaluations are in JSON. Positive gains below favor stored publications.

| Stored arm | Selected exact gain vs fresh | Final exact gain vs fresh |
| --- | --- | --- |
| Legacy | -0.007969 | 0.009965 |
| Conditioned | 0.002003 | 0.002003 |
| Competitive | 0.004999 | 0.004999 |
| No delayed credit | -0.036044 | -0.036044 |

## Whole-query causal checks

| Arm | Normal-write shuffle delta | Held-content gain | Held-shuffle delta |
| --- | --- | --- | --- |
| Legacy | -0.002354 | 0.006 | -0.008 |
| Conditioned | 0.000383 | 0.933 | -0.460 |
| Competitive | 0.001267 | -1.397 | -1.682 |
| No delayed credit | 0.000022 | -0.190 | -3.134 |

Normal-write shuffle replaces pre-query publication content while allowing ordinary query computation; positive exact-NLL delta supports content use. Held-content gain is hold+blocked minus hold; held-shuffle delta is hold+shuffled minus hold. Holding every publication also removes fast query-time computation, so these are separate diagnostics, not mandatory proof. Forks preserve earlier recurrent state and isolate direct pre-query content use, not all historical information in recurrence. 

Competitive age-filtered shuffling changed 165 older slots across 91 exact queries. Exact NLL changed by -0.000671 across all queries and -0.000833 on the matched changed-query subset. Older-publication usefulness remains unmet. The intact read mass on changed older versions was 11.0%; their age and consumption did not by themselves establish usefulness.

The earlier three-seed basic-stream audit had exact NLL 0.735 intact, 0.888 query-blocked, 1.933 held, and 0.888 held+blocked. It motivated separating continuing recurrent computation from useful held publications; it is a different stream, not a paired quality baseline.

## Retention behavior and cost

| Arm | Forced commits | Learned commits/1k bytes | Query age | Old-query mass | Revised exact |
| --- | --- | --- | --- | --- | --- |
| Legacy | 0 | 3467.1 | 1.0 | 0.0% | 34.6% |
| Conditioned | 64 | 3594.7 | 1.0 | 0.0% | 43.6% |
| Competitive | 64 | 992.4 | 11.8 | 14.6% | 39.7% |
| No delayed credit | 64 | 987.4 | 9.3 | 7.2% | 35.9% |
| Trained fresh | 64 | 993.3 | 0.0 | 0.0% | 28.2% |

Commit rate excludes startup events and counts all owners. Query age is read-mass-weighted bytes since publication over exact query windows; old mass counts publications older than 16 bytes. Fact-to-query delay is separate. Lower writes or older occupied state alone do not establish useful retention.

Competitive corrections resolve 31/66 queried revised versions; 35 remain unresolved. Among resolved versions, first success averages 1.10 query opportunities and 27.9 bytes. These are sampled opportunities, not continuous recovery timing.

Each arm trains 800 updates, 204,800 bytes and 12,288,000 sparse expert calls; batches/chunks are 16/16. Training seconds (legacy/conditioned/competitive/no-credit/fresh): 541.1/546.3/561.0/561.9/552.3.

Competitive state/eligibility storage is 103,360/959,488 bytes for evaluation/training batches. Evaluation/audit took 5.78/7.05 seconds including diagnostic forwards; live execution counts exclude those extra forwards. Selected budgets and owner utilization are in JSON. Full raw query traces ship in the attached archive; the repository contains summaries and four deterministic query samples per arm.

## Scratch outcome

Small direction-only signs favor Conditioned, Competitive; a stable retention advance remains unestablished. Competitive delayed-credit exact NLL gain over credit-disabled training is 0.041 selected and 0.041 final. The selected exact-loss credit criterion is met in this pilot. Three-seed retention replication and 24.9M scaling remain open. No age or null-mass threshold is substituted for task usefulness.

## Separate common-parent continuation

All continuation arms start from the same previously trained seed-17 checkpoint (parent step 600; SHA256 `750c9978e77a`). All weights remain trainable; optimizer, runtime and stream state restart. Each inherits 153,600 earlier training bytes and receives 204,800 new retention bytes (800 updates). Results remain separate from scratch initialization.

| Arm | Selected step | Exact NLL | Final exact NLL | Gain vs fresh | Shuffle delta |
| --- | --- | --- | --- | --- | --- |
| Conditioned | 100 | 1.297 | 1.452 | -0.038893 | 0.000416 |
| Competitive | 600 | 1.276 | 1.341 | -0.017539 | 0.001824 |
| Trained fresh | 800 | 1.258 | 1.258 | reference | n/a |

The continuation does not meet the combined trained-fresh advantage and normal-write content-sensitivity criterion. Full selected/final, age-filtered shuffle diagnostics when available, read ages, correction and resource metrics remain separate in JSON. Three-seed confirmation remains open.

Competitive continuation age-filtered shuffling changes 65 old slots across 55 exact questions; paired changed-query NLL delta is 0.002490. This causal effect is reported separately from whether stored publications improve performance over the trained fresh control.

## Validation and next target

All five regression suites pass (483 checks); 59 final focused checks pass. Logs and hashes accompany the evidence.

Next, train acquisition and payload representations at short write/read intervals inside an active gradient chunk, then increase delay and interference. Current detached traces update write-policy heads only; historical payload/feature learning remains truncated. Keep the trained-fresh, normal-content-shuffle and age-filtered causal controls. Establish a reproducible task gain before scaling.
