# SHoE: a simpler multiscale rebuild

Architecture proposal | 14 November 2026 | Prepared for Joshua Rowe

**Recommendation:** keep sparse expert selection, local learning, route reinforcement and compact associative state. Replace the cortical region graph with a small hierarchy of increasingly infrequent computations, one shared output head and one indexed association store. Make each level responsible for predicting a different scale of events.

## 1. What the rebuild should solve

The current reference at commit `5dc81667cde1ad6eef2a89b91f34f60f9e430aca` has 18 regions. Its content experts, context state, forecasts, completion records, feedback and delayed credit have separate execution and retention requirements. The selected event policy already reduces old ancestry traversal, but still maintains local frontiers, credit predictors, replay and cross-publication teaching. A smaller implementation of those same obligations would retain much of the structural expense.

The latest retrieved 1M-target Tiny Shakespeare report shows something worth preserving: SHoEbox reaches 1.97691 nats/byte versus nanoGPT's 2.27537. Training throughput is 1,702.8 versus 13,141.7 bytes/s. These are the report's matched-data results, not a matched-time quality comparison. They establish a useful learning-per-byte advantage in that run, alongside expensive training.

My architectural diagnosis is that useful sparse learning has been embedded in too much coordination. Correct constituent identities and causal teaching make the system coherent, but do not by themselves make the representations increasingly abstract. A higher region can still learn another overlapping predictor of the same stream.

The rebuild should separate three jobs: learn reusable transformations in expert weights; maintain recent predictive context in compact recurrent state; retrieve particular associations from an indexed store. Each has its own execution rate and storage budget.

| Preserve | Replace |
| --- | --- |
| Sparse selected experts and task-informed routing | The fixed map of named cortical regions |
| Local expert learning and route reinforcement | Historical publication graphs used to transport credit |
| Ordered composition of learned content | Multiple overlapping pathways that repeatedly encode the same evidence |
| Compact recurrent associative state | Repeated context reconstruction and feedback-conditioned expert passes |
| Content-addressed recall | Separate recall regions and multi-hop delivery |

This document proposes a new learning contract and architecture. It does not report a trained replacement.

<!-- pagebreak -->

## 2. The forward architecture

Start with four scales, width 64 and approximately 1M total fitted parameters. Each scale has its own expert bank and compact recurrent state. Keep one selected expert per event initially; increase stored expert capacity before increasing width or active expert count.

| Scale | Initial update rate | Input and job |
| --- | --- | --- |
| Byte core | Every byte | Immediate transitions, spelling and short context |
| Level 1 | Every 4 bytes | Compose four ordered byte-level content vectors |
| Level 2 | Every 4 Level-1 events, or 16 bytes | Compose four completed lower-level units |
| Level 3 | Every 4 Level-2 events, or 64 bytes | Model longer sequences of completed units |

These rates are scheduling defaults, not claims that a four-byte group is a word or that a 64-byte group is a sentence. After the hierarchy works, replace fixed groups with causal learned boundaries averaging about four children, constrained to a declared minimum and maximum group size.

Each slow event performs one order-sensitive composition and one selected expert transformation. Its input contains the child content vectors, positions and lengths. Use a small shared role-sensitive input adapter; do not use an unordered average. The expert produces content for the next scale. A cheap recurrent update separately supplies context for prediction.

Keep content and context distinct without executing a second full expert. The exported content path uses the completed group's observed children; the recurrent path carries surrounding history. For the initial implementation, route the canonical content transform from the group's content and roles. A route ID records which expert ran; it is never substituted for the group's meaning.

Every scale feeds one shared byte decoder. Slow states supply cached gain and bias vectors that modulate the fast state:

`decoder_input = nonlinearity(fast_state * (1 + cached_gain) + cached_bias)`

The cached values combine the latest completed slow states and recalled information. The decoder can therefore use long-context information immediately on every byte, without rerunning the slow experts. Gains and biases permit context-dependent interactions with the fast state; a fixed logit bias alone would be too restrictive.

When a group closes, update the affected scales once in increasing order. An output prediction may use only groups completed from its observed prefix. Predictions within an unfinished group use the previous slow state plus the current fast state. There is no iterative settling, same-step cycle or return message through the hierarchy.

For other modalities, adapters can emit typed content vectors with position or source roles into the same composition interface. The first implementation remains byte-based so its speed and learning can be compared directly.

<!-- pagebreak -->

## 3. Give the hierarchy a reason to learn deeply

Slower updates alone are insufficient. Each level needs a distinct prediction problem and direct evidence that its representation helps the actual output.

**Native-scale prediction.** From the completed prefix at a level, predict the content of the next completed unit at that level. A Level-2 forecast is about the next 16-byte unit in the fixed-rate baseline, not just the next byte. Train a small contrastive prediction head against the observed future unit and a bounded set of other same-scale units. Preserve order in the targets. The target vectors are detached, immutable copies of the vectors actually emitted; the prediction is recorded before the target exists.

This avoids a vocabulary sized to the entire expert bank. It also avoids assuming that regressing two freely moving embeddings toward each other will produce useful representations. Direct byte supervision and contrastive alternatives give the representation concrete discrimination work. Start with one future unit; add a second horizon only if it improves held-out quality per unit time.

**Direct task credit.** Every scale influences the shared decoder, so the ordinary next-byte loss gives each active state a direct teaching signal. Accumulate that state's gain/bias derivatives during its lifetime. Differentiate its local computation once when processing its learning block; do not reexecute it or traverse its historical ancestors on every byte.

**One explicit initial update policy.** Use blocks of eight native events per level. Hold that level's expert, state and projection parameters fixed during its block; retain the eight selected local computations and their accumulated output adjoints. When the final event's output retires, apply one local update, then start the next block. The byte readout has its own declared update schedule. Derivatives reaching a cached state are computed using the readout weights that made the corresponding prediction.

Keep gradients within that level's eight-event computation. Treat published child vectors, recalled records and the entering recurrent state as constants at the boundary. Use SHoE's existing reinforcement rule for hard route choices, alongside local derivatives through the selected continuous operations. This is deliberately local training, not a claim of full end-to-end or unlimited temporal gradients.

The block policy avoids maintaining a family of old parameter versions for overlapping regional jobs. It requires a finite event ring and selected-bank gradient buffers. It changes update timing, so the local learning rate is tuned for that timing rather than copied blindly from the old byte-wise schedule.

At 64 bytes per event, eight local events span 512 input bytes; a later 256-byte scale would span 2,048. Those are local training spans, not guaranteed memory lengths. The recurrent state can persist beyond a block, while its credit is explicitly truncated.

The central learning question is whether those coarse states retain information needed by future predictions when the recent bytes are insufficient. The acceptance tests below isolate exactly that question.

<!-- pagebreak -->

## 4. Fast associations and a bounded cost

Use compact MinKron-style recurrence for running context. Keep an indexed key-value store for specific associations that must survive context changes. A small recurrent matrix and a persistent record store perform different jobs; neither replaces the other.

Initially query the store once per Level-1 event, using a query derived from the current content and slower context. Examine a bounded candidate set, return at most two records and cache their contribution to the decoder. Admit records after their key and value are observed. Preserve payload handles for facts or sequences requiring exact recovery rather than expecting a small latent vector to reproduce an arbitrary string.

Start with a single store and one delivery path. Route recall directly to decoder conditioning. Keep its reads, writes, capacity and candidate count explicit. Test the hierarchy without this store first, then require a gain on delayed association and exact-recall tasks before enabling it in the selected profile.

Four scales at rates 1, 1/4, 1/16 and 1/64 execute:

`1 + 1/4 + 1/16 + 1/64 = 1.328125 selected expert events per byte`

That is an event-count budget, not a measured speedup. It excludes grouping, routing, recurrent updates, the byte decoder, local prediction heads, learning and memory access. The corresponding cost model is:

`cost_per_byte = byte_cost + level1_cost/4 + level2_cost/16 + level3_cost/64 + output_cost + memory_cost + learning_cost`

Each level cost includes its child-group adapter. Adding a 256-byte scale adds only 1/256 expert events per byte, but a very expensive composer or predictor can still dominate. Measure the complete training loop.

The initial engineering budget is 15,000 training bytes/s, or 66.7 microseconds per byte, on the same CPU placement used for its comparison. It is a target. Keep generation speed and aggregate multi-stream throughput separate.

| Implementation rule | Why it matters |
| --- | --- |
| Fixed arrays for level state and event rings | Predictable memory and no per-event object allocation |
| One shared byte decoder | Avoid a full vocabulary head at every scale |
| Recompute slow projections only when their state changes | Preserve the benefit of infrequent events |
| Selected-bank updates | Stored capacity does not force dense optimizer traffic |
| Bounded routing candidates at larger bank sizes | Top-1 execution alone does not make an all-expert search cheap |
| Explicit level counters and bottom-up promotion | Avoid a general graph scheduler |
| Inspection outside the training loop | Keep measurement machinery out of ordinary execution |

The first small model may scan its small bank. Scaling toward 100M requires qualifying a hierarchical or indexed shortlist router; constant active expert count is not by itself constant routing cost.

<!-- pagebreak -->

## 5. Build order and acceptance

Start a separate implementation with the old system retained as a benchmark. Do not begin by importing its regional scheduler, publication DAG, replay machinery or compatibility layers.

1. **One fast SHoE core.** Reuse the selected expert kernel, routing/reinforcement arithmetic and compact state primitive. Establish the new runtime's byte-only speed and learning curve.
2. **Add one composition level.** Give it native-scale prediction and a direct path to the shared decoder. Confirm that it learns an identifiable long-dependency task before adding more levels.
3. **Expand to four scales.** Keep the total fitted parameter budget near 1M. Use fixed groups first and separately tune the eight-event update policy.
4. **Add the single association store.** Test new bindings, distractors and delayed retrieval, including exact payload recovery.
5. **Learn boundaries.** Only after the fixed-rate hierarchy earns its compute, train boundaries to improve future prediction under an event-rate constraint. Compare on held-out shifted boundaries, not just training loss.
6. **Scale bank capacity.** Compare 1M and 5M with the same active-expert budget; then qualify larger routers, memory traffic and batching.

| Test | Question it answers |
| --- | --- |
| Matched recent suffix, different distant cue | Does the slow state carry information unavailable to the fast suffix? |
| Same child inventory, different order | Does composition encode arrangement rather than a bag of features? |
| Familiar parts in held-out combinations | Does the hierarchy reuse learned components? |
| Increasing distractor gaps | Do useful dependencies grow with added scales? |
| Shifted chunk boundaries | Is success tied to the arbitrary four-byte grouping? |
| New key-value binding and delayed query | Does the store supply fast association and exact recall? |
| Mask or shuffle individual slow states | Does the trained decoder actually use their information? |
| Retrained smaller matched-budget models | Does the hierarchy improve learning beyond extra parameters? |

Frozen interventions show use, while retrained controls establish architectural value. Keep both. A nonzero gradient or changed output is not sufficient evidence that a level improves prediction.

For language, use Tiny Shakespeare as a quick regression task and a held-out mixed prose/code corpus for selection. Compare new byte-only SHoE, new hierarchical SHoE, current SHoEbox and nanoGPT with matched data, parameter accounting and hardware. Report held-out nats/byte against both bytes and wall time, time to an attainable loss, and several matched seeds. Recount all fitted routers, projection heads and memory maps.

Require the hierarchy to improve long-dependency accuracy and held-out quality per training second over its fast-only control. Use identical candidate output settings and separate training from generation measurements. Do not choose a larger arrangement merely because every region has observable activity.

<!-- pagebreak -->

## 6. Evidence and related work

### Inspected project evidence

- Current source revision: [JRowe47/SHoEbox, 5dc8166](https://github.com/JRowe47/SHoEbox/tree/5dc81667cde1ad6eef2a89b91f34f60f9e430aca).
- [Current reference description](https://github.com/JRowe47/SHoEbox/blob/5dc81667cde1ad6eef2a89b91f34f60f9e430aca/README.md): regional execution, learned content, teaching and inference policies.
- [SHoE baseline](https://github.com/JRowe47/SHoEbox/blob/5dc81667cde1ad6eef2a89b91f34f60f9e430aca/SHoE_BASELINE.md): semantic routing, expert sharing and HAT interfaces.
- [Event learning implementation](https://github.com/JRowe47/SHoEbox/blob/5dc81667cde1ad6eef2a89b91f34f60f9e430aca/experiments/identity_semantics/runtime/hierarchy_lab/throughput_probe/event_learning.inc): exact frontier, directional credit and replay.
- [Local credit implementation](https://github.com/JRowe47/SHoEbox/blob/5dc81667cde1ad6eef2a89b91f34f60f9e430aca/experiments/identity_semantics/runtime/hierarchy_lab/throughput_probe/local_credit.inc): transport through retained inputs and feedback producers.
- [Compact context implementation](https://github.com/JRowe47/SHoEbox/blob/5dc81667cde1ad6eef2a89b91f34f60f9e430aca/experiments/identity_semantics/runtime/hierarchy_lab/throughput_probe/minkron_context.inc): small associative state with read-before-write updates.
- Existing project report, *TinyShakespeare_1M_Comparison.md*, 14 November 2026: 1,000,000 targets, seed 7, 1,014,800 SHoEbox and 1,033,600 nanoGPT parameters, same two-CPU placement. The 1M figures quoted here come from the report's full held-out split, not its smaller curve subset.
- Existing *SHoE Architecture and Learning Specification*, version 0.1, 31 October 2026: route provenance and semantic content are distinct; completed records carry event and order information.

### Relevant primary research

**A Clockwork RNN** (Koutnik et al., 2014) provides a precedent for updating different recurrent modules at different rates. The lesson used here is to assign explicit clocks to computation. It does not establish the performance of the proposed sparse hierarchy.

[Paper](https://proceedings.mlr.press/v32/koutnik14.html)

**Hierarchical Multiscale Recurrent Neural Networks** (Chung et al., 2016/2017) studies temporal hierarchy and learned boundary/update behavior. It motivates separating native event clocks from raw input positions.

[Paper](https://arxiv.org/abs/1609.01704)

**Dynamic Chunking for End-to-End Hierarchical Sequence Modeling / H-Net** (Hwang, Wang and Gu, 2025) demonstrates learned content-dependent chunking in an explicit language-model hierarchy. Its relevant design principle is to spend more expensive processing on a compressed sequence while preserving a path to fine-scale output.

[Paper](https://arxiv.org/abs/2507.07955)

The proposed combination, local update schedule, compute budget and build sequence are architectural recommendations derived from this review. The first decisive experiment is the fast core versus the same-budget core with one genuinely useful slow level.
