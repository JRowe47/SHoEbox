# SHoEbox: first-class memory integration

November 8, 2026 (Mountain Time)

The architecture-review refactor is implemented: distinct memory slots now feed ordinary sparse computation and final composition. The integration includes event-based storage, direct content paths, independent working state and bounded retrieval. **All 94 checks pass**, and exact resume reproduces weights, optimizer state and carried memory.

The four-update continuation lowers paired-answer loss and language loss, but does not solve the conflicting-question evaluation. The experimental checkpoint is available for continuation; the retained parent remains the working model.

## Implemented changes

| Review recommendation | Delivered behavior |
| --- | --- |
| First-class memory context | Up to eight distinct raw-content slots per consumer, with learned keys/roles, validity, versioned identity and event age. The question has its own reserved lane. |
| Regional requests | Every node can request evidence using fresh ingress, its regional cue and its committed working state. |
| Sparse and late access | A shared conditional reader feeds routing/experts at blocks 0 and 2. A final composer reads evidence beside complete P/Q query features. |
| Preserve content | Owner reads combine a direct content projection with a contextual adjustment. Query and stored key stay separate. Cached reads deliver the refinement that bounded CA1 accepts. |
| Separate memory jobs | Working state advances while publications hold. Durable anchors remain separately addressable. The legacy processor-hold option is retained. |
| Dedicated graph paths | Raw publication slots provide protected forward content lanes alongside the existing transformed graph connections. |
| Multiple detail levels | Source and abstract anchors share a regional event binding and link budget. Repeated rounds refresh or version the same pages. |
| Demand and caching | Version-valid slots are reused. A fair scheduler admits at most four directory requests per stream per round, also respecting the service budget. Same-world requests share capsule fetches. |
| Causal provenance | Mixed anchors record recalled handles and direct publication versions available to their computation. Ungrounded recalled activity cannot create a new event offer. |
| Ordinary learning | Task and language losses train the new path inside the live credit window. Extra lace-credit rules are excluded. |

The identity/version fields control transport and freshness; no answer database, task parser or replay store is introduced. Slot selection regulates consumption without multiplying away durable content.

<!-- PAGE -->

## Compact continuation results

All three models see the same locked inputs: one paired template at each delay with the full 4x4 value crossing, four single-fact answers and sixteen correction answers at +64 bytes, plus 128 language targets. Counts below exclude equal-answer pairs. This compact integration bank is separate from the existing promotion banks.

| Measurement | Retained parent | New interface, initial | After 4 updates |
| --- | ---: | ---: | ---: |
| Both conflicting questions, +0 | 0/12 | 0/12 | 0/12 |
| Both conflicting questions, +32 | 0/12 | 0/12 | 0/12 |
| Both conflicting questions, +64 | 0/12 | 0/12 | 0/12 |
| Single fact, +64 | 4/4 | 4/4 | 4/4 |
| Correction, +64 | 16/16 | 16/16 | 15/16 |
| Paired-answer NLL, +64 | 1.6471 | 1.6573 | 1.2978 |
| Language NLL | 3.4470 | 3.4797 | 3.3627 |

The paired-answer probabilities improve while both-correct accuracy stays at zero. Correction retention loses one answer. No model is promoted from this run.

## Validation and cost

The 94 checks cover legacy compatibility, ordinary gradients, slot identity and stale-version rejection, causal chunking, fork/reset, source masking, event deduplication, active working state, runtime parity and activation recomputation. A controlled two-fact reader intervention changes the requested source's output while preserving the other query. In the trained model, blocking source A only in the new context lane changes logits by up to 0.6120 on the same history; it changes none of the 32 answer predictions.

Exact optimizer/state resume passes, including all 757 inherited optimizer entries. All 227 guarded source files match the delivered implementation. A new continuation from the experimental endpoint also restores successfully.

The model has 458,669 unique parameters and keeps the existing 384 physical experts and 84 expert calls per round per external event. The four updates consume 96 memory-answer targets, 2,224 executed task positions and 512 language targets. Training takes 315.1 seconds with peak process RSS 2,771.6 MiB in this CPU run with concurrent checks/evaluation.

On the +64 paired bank, 8,038 of 31,732 selected capsule reads reuse another same-world fetch (25.3%). There are 4,525 context-reuse decisions and 6,955 deferred requests. These are bounded reference-runtime counters, not a distributed throughput result.

## Recovery

The source archive contains the implementation, checks, protocol and measured results. Extract the checkpoint archive over the source directory to restore the retained parent, new initial/final checkpoints, optimizer/RNG/carried state, exact corpus and locked inputs.

The final endpoint is `artifacts/memory_context/pilot/last.pt`, at update 4 and next document 667. Start a new continuation with this endpoint as `--parent` and a fresh output directory. Use `--resume` with the original parent and unchanged step budget for an interrupted run. Commands and controls are documented in `docs/memory_context.md`.
