# SHoEbox hippocampus implementation

November 8, 2026 (UTC) | Completed implementation and controlled pilot

The explicit hippocampus is implemented: one shared sparse associative service,
owner-local capsules, a dedicated cue lane, conditional regional readers and
causal next-round commits. The extension defaults off. The shared-memory arm does not pass the declared validation promotion gates. Keep the stronger retention parent as the working model; retain the implemented extension behind its switch.

## Fixed-budget results

All trained arms use the same strong parent, 64 updates, source schedule, AdamW
settings and task/language objectives. The unchanged control's completed endpoint
was reused after 30 updates reproduced its losses, gradient norms and
source ledgers exactly. Final cohorts are new: seed 1201, document start 80000.
There are 192 conflicting histories per paired cohort; both questions must be
correct about the same unchanged history.

| Model | Unique parameters | Both +0 bytes | Both +32 | Both +64 |
| --- | --- | --- | --- | --- |
| Retained parent | 312,440 | 0/192 (0.0%) | 0/192 (0.0%) | 0/192 (0.0%) |
| A - paired control | 312,440 | 0/192 (0.0%) | 2/192 (1.0%) | 4/192 (2.1%) |
| B - local anchors | 391,174 | 1/192 (0.5%) | 0/192 (0.0%) | 1/192 (0.5%) |
| C - shared hippocampus | 391,174 | 5/192 (2.6%) | 2/192 (1.0%) | 3/192 (1.6%) |

| Model | Single +64 | Correction +64 | Natural-text NLL |
| --- | --- | --- | --- |
| Retained parent | 100.0% | 100.0% | 3.132340 |
| A - paired control | 88.3% | 71.1% | 3.134060 |
| B - local anchors | 85.9% | 72.3% | 2.981199 |
| C - shared hippocampus | 99.2% | 88.7% | 3.009036 |

Single-fact tests use 32 templates; two-fact/correction tests use 16. Natural-text NLL uses 1,024 bytes. Eight documents per ordinary and long-delay stream are included in the machine-readable results.

Changing only the query switches C's answer in 7/192, 3/192 and 3/192 conflicting histories at +0/+32/+64. Source-value interventions and each query direction are reported separately below.

| Gap | Older correct | Newer correct | Older source switches | Newer source switches |
| --- | --- | --- | --- | --- |
| 0 | 30.5% | 95.7% | 14/64 | 64/64 |
| 32 | 28.1% | 96.9% | 8/64 | 64/64 |
| 64 | 36.3% | 89.5% | 16/64 | 64/64 |

The shared arm still predominantly answers with the newer value. The outstanding behavioral exit is reliable query-dependent selection between the retained facts.

## Validation and promotion

Validation uses seed 683/start 3000 and eight templates per cohort. The primary comparison is the fixed update-64 endpoint. B/C validation-selected checkpoints are retained separately.

| Arm | Selected update | Selected paired NLL | Pair gate | Retention gate | Correction gate | Language gate |
| --- | --- | --- | --- | --- | --- | --- |
| B | 64 | 0.851335 | Fail | Fail | Fail | Pass |
| C | 48 | 0.930073 | Fail | Pass | Fail | Pass |

Gate columns describe update 64, separately from the validation-selected candidate. Promotion requires at least 50% both-correct validation pairs at gaps 0 and 32, no more than a 2-point single-fact decrease, no more than a 3-point correction decrease, and at most +0.05 natural-text NLL versus the parent on matching validation cohorts. Control A uses its verified fixed endpoint; its intermediate validation selection is not compared.

| Arm | Extra state capacity | Retrieval execution |
| --- | --- | --- |
| A | None | Original local publication recall. |
| B | 64 capsules per region; no global bindings | Dedicated cues and conditional reads within the requesting region. |
| C | 64 capsules per region plus 2 x 128 shared bindings | Bounded global directory/postings and targeted owner reads. |

B and C instantiate the same optional parameter schema; B bypasses shared association routing. Unique parameter count therefore matches, while their allocated fast state and executed service work differ.

## What was implemented

| Component | Behavior |
| --- | --- |
| Regional interface | A shared compact trunk with one independent input-driven cue trace per region; no candidate pooling in the initial cue. |
| Owner anchors | 64 fixed-width capsules per region, multiplexing its nodes; generation/version/namespace validation and immutable snapshots. |
| DG / CA3 | 8-of-256 sparse separation, binary shard directory, bounded postings, associative links and at most two completion steps. |
| Conditional readers / CA1 | Owner-local capsule/cue interaction; distinct returned roles; learned accept/null/refine and gated expert-input reinstatement. |
| Admission / revision / feedback | Separate write/erase policies, bounded replacement, provenance, apply-once contributor receipts and attributed publication feedback. |
| Runtime / checkpoints | Read-before-write scheduling, isolated counterfactual forks, named optimizer migration, row reset, detach/device transfer and serializable queues. |

Regional experts, HAT encoding, native recurrence and ordinary publications remain in their existing owners. The service stores compact associations and handles, while experts interpret the retained capsules. No replay buffer, answer dictionary, task parser, frozen backbone or extra answer-supervised binding loss was introduced.

## Structural and gradient exits

51 targeted checks pass. Disabled behavior and the enabled zero-residual warm start preserve logits, RNG and clocks. Tests cover source/write/query/link/reader gradients, stale descriptor masking, correction/eviction, duplicate delivery, reset isolation, counterfactual branches, exact checkpoint/optimizer resume, two independent owners read by a third region, and owner-name permutation.

Third-region, masked-owner and permutation checks are controlled wiring fixtures. The learned comparison uses the native three-region byte task.

Training retains full live credit across recomputed segments and detaches at the optimizer boundary. Sparse recomputation leaves unused owner/page gradients absent, preserving optimizer behavior. The actual update-14 checkpoint reproduces 28-byte logits and loss exactly; the maximum gradient difference across 828 parameter tensors is 3.58e-7. A masked FP64 probe agrees to 1.14e-15 relative gradient error; FP32 accumulation differs by 1.17e-6 relative L2. Historical content gradients are not synthesized from detached handles.

The shared arm resumed from update 14 after host-memory failures. The previously failing update completed at approximately 1.2 GiB process memory with 16-byte recomputation segments. Short calls use direct execution; the execution history records the switch. This changes activation storage, not the learning window or update/data budget.

## Lateral scaling smoke

The same sparse service was exercised with 64-512 mock regional owners and 4-32 shards. This is a single-process CPU actor simulation. It measures bounded lookup, paging and transport behavior; physical multi-device/GPU execution remains the later H5 milestone. Latencies were sampled while other training was active.

| Regions | Shards | p50 ms | p95 ms | Max candidates | Max posting reads | Max readbacks |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 4 | 2.168 | 2.914 | 32 | 51 | 2 |
| 128 | 8 | 2.073 | 2.622 | 32 | 49 | 2 |
| 256 | 16 | 2.078 | 2.374 | 32 | 57 | 2 |
| 512 | 32 | 3.136 | 3.362 | 32 | 59 | 2 |

Each request is capped at two shards, 128 actual posting entries, 32 fine candidates and eight target readbacks, including association expansion. The directory also passes a zero-cue tie test without scanning every shard. Queues have explicit overload/expiry and fair service; future messages cannot block unrelated regions.

| Regions | Burst requests | Requests/s | Visited owners max | Defer events | Dropped / expired |
| --- | --- | --- | --- | --- | --- |
| 64 | 8 | 481.9 | 2 | 0 | 0 / 0 |
| 128 | 16 | 486.3 | 2 | 0 | 0 / 0 |
| 256 | 32 | 505.2 | 2 | 16 | 0 / 0 |
| 512 | 64 | 365.4 | 2 | 96 | 0 / 0 |

The queue holds 64 requests. Defer counts are service-cycle events and may count one request more than once. Each smoke allocates one anchor/binding per mock owner; it does not saturate total capacity. Returned evidence plus cue occupies at most 256 bytes per request in this probe.

| Regions | Persistent tensors KiB | Python metadata KiB | Saved tensors/query KiB | Gradient bytes/query KiB | Owners with gradient |
| --- | --- | --- | --- | --- | --- |
| 64 | 88.8 | 92.5 | 157.1 | 134.7 | 2 |
| 128 | 177.5 | 187.8 | 161.7 | 134.7 | 2 |
| 256 | 355.0 | 365.6 | 158.6 | 134.7 | 2 |
| 512 | 710.0 | 723.1 | 157.1 | 134.7 | 2 |

Cold capacity uses lazily allocated immutable one-record pages. Python metadata is an object-size estimate. Saved-tensor and gradient allocations above belong to one bounded service query; they are separate from full language-model training activations.

## Inference ablations

Every intervention uses the same final C weights and held-out histories. Shuffles only exchange causally valid selected anchors; no unavailable or future content is inserted.

| C intervention | Both +0 | Both +32 | Both +64 | Paired NLL +32 |
| --- | --- | --- | --- | --- |
| intact | 5/192 (2.6%) | 2/192 (1.0%) | 3/192 (1.6%) | 1.038091 |
| blocked | 4/192 (2.1%) | 1/192 (0.5%) | 2/192 (1.0%) | 1.036522 |
| shuffled | 5/192 (2.6%) | 1/192 (0.5%) | 3/192 (1.6%) | 1.037169 |
| pooled | 5/192 (2.6%) | 1/192 (0.5%) | 2/192 (1.0%) | 1.037234 |
| no_refinement | 5/192 (2.6%) | 1/192 (0.5%) | 3/192 (1.6%) | 1.037180 |
| no_feedback | 5/192 (2.6%) | 2/192 (1.0%) | 3/192 (1.6%) | 1.037113 |

## Reproducibility and delivery

The source archive contains the implementation, tests, original specification and experiment commands. The checkpoint archive contains the retained parent, A/B/C endpoint and validation-selected checkpoints, exact corpus, full histories/evaluations, RNG and optimizer state, and SHA-256 checksums. Training source trajectories agree across all arms.

| Arm | Weight KiB | Optimizer KiB | Carried state KiB |
| --- | --- | --- | --- |
| Retained parent | 1220.5 | 2443.9 | 35.9 |
| A - paired control | 1220.5 | 2443.9 | 35.5 |
| B - local anchors | 1528.0 | 3055.5 | 583.0 |
| C - shared hippocampus | 1528.0 | 3058.8 | 3359.3 |

| Arm | Saved training tensors MiB | Gradient MiB | Gradient tensors |
| --- | --- | --- | --- |
| Retained parent | 27.729 | 1.192 | 757 |
| A - paired control | 27.729 | 1.192 | 757 |
| B - local anchors | 41.551 | 1.427 | 808 |
| C - shared hippocampus | 44.348 | 1.425 | 810 |

The training-storage probe uses the same 56 natural-byte positions per model (batch 2, context 28), cold state, direct autograd and no optimizer update. Saved tensor storage excludes model weights and inputs; it includes newly created fast state. Python graph metadata and allocator overhead are excluded. This is a declared representative call, not the peak of a full task update. Full-run process high-water marks are retained in the histories.

Weights and storage are counted uniquely; state-dict aliases do not inflate these totals. H0-H2 structural exits pass, H3 completes the fixed-budget comparison and ablations, and H4 completes the bounded actor smoke. The validation gate determines promotion independently of implementation completion.
