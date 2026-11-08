# SHoEbox: Query-Conditioned Lace Separation
## Research plan and next-run handoff - v1.0
**Prepared November 7, 2026 (America/Denver)**

Implement a small routing intervention that reinforces useful question-specific laces and weakens shared routes when task feedback shows that they cause interference. Reuse the paired questions, routing activations and gradients already computed by SHoEbox. The immediate target is reliable switching between two questions about the same retained history.

This is an implementation and experiment plan. No new training or performance result is claimed here.

### 1. Execute this next

Read the current repository handoff and the hippocampus binding protocol, preserve intervening work, implement the default-off mechanism below, run the four matched pilot arms, and publish the completed code and results under the project's standing authorization. Keep the retained strong parent unless the existing promotion gates pass.

Do not expand this run into an adversarial HAT, a second language model, a replay system, a new memory store or global optimizer changes. The experiment changes query-conditioned routing and its local learning signal.

### 2. Starting evidence and recovery

Repository: [JRowe47/SHoEbox](https://github.com/JRowe47/SHoEbox). Inspected main revision: `f762518a30c7cc01cc308a4a362a3be73d62b52e`. Inspect newer commits before editing.

| Latest bounded-CA1 endpoint | Recorded result |
|---|---|
| Both conflicting questions correct | 0/96 histories at added delays 0, 32 and 64 |
| Question-induced answer switches | 0/96 immediate pairs |
| Single-fact accuracy at +64 | 98.4% |
| Correction accuracy at +64 | 97.7% |
| Diagnostic query-cue cosine | 0.999947 |
| Unique hippocampus-model parameters | 391,174 |

The previous repair increased delivered retrieval evidence substantially without solving question-conditioned selection. This experiment tests separation at routing decisions; it does not assume the stored facts have disappeared.

**Experimental starting checkpoint:** `artifacts/hippocampus_binding/ca1/bounded/last.pt`. Use the fixed final endpoint, not `best.pt`, which selected update 0 in that continuation. Start every new arm from the same inherited weights, optimizer, RNG, carried language state and next document.

**Retained promotion reference:** `artifacts/hippocampus/parent.pt`, the bundled strong memory parent. It remains the working model while the experiment runs.

If the workspace is missing, recover `SHoEbox_Hippocampus_Binding_Checkpoints.zip` and, if needed, `SHoEbox_Hippocampus_Binding_Source.zip`. Read `STATUS.md`, `docs/hippocampus_binding.md` and `docs/hippocampus_binding_report.md`. Preserve the recorded corpus and checksums.

This is an explicit new-protocol continuation. Do not bypass the source guards for an old exact-resume run. Record the new source revision and initialize only the added parameters and their optimizer state.

<!-- PAGEBREAK -->

### 3. Mechanism and boundaries

For corresponding routing decisions under questions A and B, define the selected lace sets:

- Shared: `O = S_A intersection S_B`.
- Unique to A: `U_A = S_A minus S_B`.
- Unique to B: `U_B = S_B minus S_A`.

Identity includes the owning controller/node, layer and lace ID. Compare only aligned identities. A numeric lace ID in another controller is not the same semantic object.

The intervention has two components:

1. **Unique-lace reinforcement:** increase the routing preference for unique laces whose ordinary loss gradient indicates that more weight would help.
2. **Contentious-overlap inhibition:** reduce a shared lace's preference under the question where more weight would hurt, when the other question gives it opposing credit.

A shared lace useful to both questions receives no extra overlap penalty. A shared lace useful to A but harmful to B is inhibited under B. The first version does not automatically erase it under A. Symmetric suppression of all shared activity is a separate, deferred ablation.

Lowering shared weights within a normalized routing distribution can increase the relative contribution of unique laces. Subtracting an equal constant from two whole-bundle scores cannot distinguish them. The adjustment must depend on the question and act before selection/normalization.

**Preserve shared computation.** For "Alice's color?" and "Bob's color?", shared color-retrieval machinery may be useful. The separation target is the association that selects the requested binding. Apply the first implementation to recurrent-node lace routing, leaving byte ingress/HAT symbol formation and stored capsules unchanged.

**No required unique path.** Identical hard top-k sets can still differ in weights. Opposing credit on their shared laces can train that difference. If both the question features and routing credit are indistinguishable, record the failed symmetry break; do not invent unique paths using answer IDs or random forced disjointness.

**Two timescales.** At inference, the learned bias changes current routing immediately as question context changes. During training, local credit updates the bias parameters. The pilot adds no persistent inhibition trace and does not weaken memory capsules after unrelated events.

### 4. Small question-conditioned bias

Add a low-rank head to each existing recurrent-node lace controller:

```text
z = concatenate(current local routing context, dedicated regional cue)
h = U z
raw_bias = reshape(V h, [layers, laces])
bias = cap * tanh(raw_bias / cap)
adjusted_scores = original_scores + bias
```

Use the current causal dedicated cue already available to the region and the local context that is supplied to its lace controller. Document the exact tensors and their event timing. Never supply question IDs, answer values, source positions or evaluator owner annotations to this head.

Initial settings: rank 4, cap 0.5 logits, deterministic small initialization for U, and exactly zero V. All new parameters remain trainable. Use isolated RNG initialization so installing a disabled head cannot change the parent trajectory.

The zero head must reproduce parent logits before training. Add the bias to fresh scores before top-k in both runtime and reference implementations. Preserve the existing shared utility bias as a separate input. Apply the new bias once to each fresh selection calculation; never accumulate it repeatedly into carried scores.

This uses a small head rather than another expert execution. Parameter cost per controller is approximately `rank * (feature_width + layers * laces)`, excluding any explicitly added feature normalization. Record actual counts.

<!-- PAGEBREAK -->

### 5. Local credit rule: reuse one ordinary backward pass

Use the existing paired training branches. Capture the selected-score tensor for the final lace-refinement iteration and its normal backward gradient at the aligned answer-predicting decisions. Selection itself remains the existing hard top-k operation.

Record only the decision records needed for the current update: controller identity, selected IDs, selected scores/weights, detached z and h, and the head's bounded bias. Release them after applying credit. This is temporary optimizer bookkeeping, not an example replay buffer.

Let `g[q,i]` be the gradient of the ordinary loss through a selected lace score. Negative means increasing that score would locally reduce loss; positive means it would increase loss. This is first-order contribution evidence, not a counterfactual per-lace ablation. The direct load-balancing branch must not be mistaken for task credit.

Use a detached pair scale `a = RMS(g[A], g[B])`. Skip a pair with zero/nonfinite usable scale. Set `gbar = clamp(g / (a + epsilon), -2, 2)`, with epsilon chosen for the working dtype. For shared laces, define:

```text
conflict[i] = 1 if gbar[A,i] * gbar[B,i] < -(0.1 ** 2), else 0

extra_grad[q,i] = a * (
    - alpha * unique[q,i] * relu(-gbar[q,i])
    + beta * shared[i] * conflict[i] * relu(gbar[q,i])
)
```

Default coefficients: `alpha = 0.10`, `beta = 0.05`. Set the absent arm's coefficient to zero. These are added gradient contributions to the bias head only; ordinary gradients remain unchanged everywhere else.

**Supervision gate.** Apply extra credit only to aligned native answer targets that differ between the paired questions. Use existing training metadata outside the forward model. From already computed answer logits, measure each correct target's margin over the other question's target. Apply the correction only while at least one margin is below 1 nat. Same-target pairs receive ordinary learning only. No targets or pair identity enter inference.

**Apply without another model pass.** After the normal backward, scatter extra_grad into the relevant head outputs. With `v = extra_grad * (1 - (bias/cap)^2)`, accumulate the following local additions using the cached forward values:

```text
delta_V_grad += outer(flatten(v), h)
delta_U_grad += outer(V.T @ flatten(v), z)
```

Sum over records; the captured gradients already include the original loss reduction, so do not average by batch size a second time. Sum by parameter identity if a head is shared. Cap the combined extra head-gradient norm at 25% of its ordinary head-gradient norm; a zero ordinary norm gives zero extra correction. Then use the existing global clipping and optimizer step once.

Use the unmodified forward-time V for the local derivative. Account for any mixed-precision gradient scaling exactly once, or disable the extra-credit path until its scaling is verified. Do not use a second full backward, second-order gradients, a retained history graph beyond its existing window, or manual changes to optimizer moments.

Ordinary task gradients still train the cue path, regions, experts and new head. The additional local correction intentionally treats z as fixed; it does not claim to train the upstream cue encoder through this extra path.

<!-- PAGEBREAK -->

### 6. Integration and preflight

Verified implementation anchors:

| Existing file | Relevant change |
|---|---|
| `experiments/hat_minkron/semantic_core.py` | Reference LaceController: add optional per-query bias before top-k |
| `experiments/text_regions/runtime.py` | RuntimeLaceController: match reference behavior and expose optional live selected-score records |
| `experiments/text_regions/node.py` | Supply causal cue/context to the small head; retain decision records only when requested |
| `experiments/text_regions/paired_recall.py` | Align existing question branches and supervised answer decisions outside model input |
| `experiments/text_regions/config.py` | Default-off settings, rank/cap/credit coefficients, independent ablations |
| `experiments/text_regions/hippocampus_interface.py` | Reuse regional cue; preserve cue/offer separation and clock semantics |

The existing node header and diagnostic trace are not a differentiable credit interface. Capture live selected scores before their diagnostic detachment. Do not build a new dense expert-bank gradient or a global all-pairs lace table.

Proposed additions, not existing files: `lace_separation.py`, `test_lace_separation.py` under `experiments/text_regions/`; `scripts/run_lace_separation.py`; `docs/lace_separation.md`.

Keep default-off behavior exact. Extend checkpoint migration, reset behavior, runtime caching and activation recomputation consistently. A recomputed forward must not register the same credit twice. Restoring a checkpoint must preserve the new parameters, optimizer state, experiment settings and next source position.

**Preflight before training:**

1. Reproduce the bounded endpoint's evaluation and inspect the actual run configuration.
2. Capture both native questions from the same history at delays 0 and 32. Measure hard lace overlap, weight differences, head-input differences and opposing credit.
3. Verify bias reaches selected laces, expert inputs and answer logits. A frozen-weight question-cue permutation is a diagnostic, not a training signal.
4. Confirm the cue changes with question content. If z is exactly identical, resolve its wiring/timing first. If it differs weakly, retain that observation and test the head's sensitivity.
5. Time ordinary routing, instrumentation and credit application on the same workload. Keep instrumentation bounded and disabled in normal evaluation.

Required focused checks: disabled/zero-head parity; correct pair/layer/owner alignment; helpful unique credit sign; harmful shared credit sign; useful-to-both and same-target no-penalty cases; analytic head-gradient agreement with a tiny local autograd reference; inactive/padded-row exclusion; runtime/reference parity; apply-once behavior under recomputation; checkpoint/resume parity.

Do not require every overlapping route to become disjoint. Do not claim causality from hard-ID overlap alone.

### 7. Locked pilot arms

All arms retain the same bounded CA1 policy, event cue clock, reinstatement placement, inherited state and source schedule. Install identical heads in all arms; bypass them in A.

| Arm | Bias head | Unique reinforcement | Overlap inhibition |
|---|---|---|---|
| A: unchanged continuation | Bypassed | Off | Off |
| B: capacity control | Normal task learning | Off | Off |
| C: unique credit | Normal task learning | alpha 0.10 | Off |
| D: full proposal | Normal task learning | alpha 0.10 | beta 0.05 |

A versus B measures the new conditional head. B versus C measures extra unique credit. C versus D isolates contentious-overlap inhibition. Do not attribute B's improvements to the overlap rule.

<!-- PAGEBREAK -->

### 8. Training and evaluation protocol

Run **32 optimizer updates per arm**, saving at least every eight updates. This is the declared pilot endpoint. Preserve the inherited learning rate, AdamW settings, objective weights, global clipping, native clocks, memory budgets, active expert counts and source schedule. Do not change them differently across arms.

Continue the established paired/paired/single/correction cycle and delay curriculum after confirming the inherited protocol. Start each arm from the same next unused document and RNG state. Initialize new optimizer slots consistently. Log actual answer targets, logical task bytes, executed positions, natural-text bytes and wall time.

Use the same activation-recomputation schedule in every arm. Checkpoint at optimizer boundaries. Write compact result files and a short status handoff so recovery does not depend on chat output.

Before looking at results, enumerate and hash the new cohorts. Proposed fresh cohorts are:

| Cohort | Seed | Start index | Use |
|---|---:|---:|---|
| Diagnostic | 2309 | 180000 | Wiring, credit and timing |
| Validation | 2411 | 200000 | Fixed gates and checkpoint selection |
| Final test | 2539 | 240000 | One locked endpoint evaluation |

Verify these case identities are unused and mutually disjoint against the available ledgers; replace and record them before training if necessary. Index ranges alone are not proof of disjointness. Do not inspect final-test outcomes to choose coefficients or training length.

Use at least eight validation templates and sixteen final paired templates. Score at least 96 validation and 192 final histories per delay, with both questions evaluated on the same unchanged history. Expand and lock the counts before training if balancing the full source-value combinations requires more cases. Include equal-answer controls, both source orders and unseen question wording. Preserve the established single-fact, correction and natural-text cohorts and report their exact counts.

**Primary endpoint:** both questions correct at added delays 0, 32 and 64. Also report each direction's accuracy and question-induced answer switching on distinct-answer pairs. Switching alone is not success.

**Mechanism measurements:** hard overlap, soft-weight separation, mass on shared/unique laces, opposing-credit counts, fraction of eligible pairs actually updated, bias magnitude/saturation, extra-gradient norm, head-input discrimination and update coverage. Stratify same-target pairs separately. Lower overlap without better answers does not establish progress.

**Retention and cost:** single-fact and correction accuracy, natural-text NLL, executed experts, memory reads/refinements, parameters, peak memory, training wall time and inference latency. Measure normalized contribution changes; do not equate functional pruning with fewer executed experts under fixed top-k.

Evaluate same-weight ablations on D's final endpoint: bias off, question cues permuted within the same valid history, and hippocampal readbacks blocked. These probe reliance on the learned path. Reversing a training update at inference is not a valid ablation; C versus D is the inhibition comparison.

Report fixed update-32 endpoints as primary. Validation-selected checkpoints may be secondary and must retain their update numbers. Keep final test sealed until implementation, coefficients, cohorts and endpoints are locked.

### 9. Decisions after the pilot

Promotion still requires at least 50% both-correct validation pairs at delays 0 and 32, at most a two-percentage-point single-fact drop, at most a three-point correction drop, and no more than +0.05 natural-text NLL versus the retained strong parent on matching validation cohorts. Report fresh test results separately.

If D improves on C while preserving the other behaviors, retain the inhibition mechanism for a larger run. If B accounts for the gain, favor the simpler head. If C helps but D does not, retain unique credit and keep inhibition independently ablatable.

If credit rarely fires, inputs remain indistinguishable or routes lack task influence, diagnose that specific obstruction. Do not respond by increasing inhibition blindly or declaring the broader feature impossible. Preserve plausible mechanisms behind switches for later capacity/training ablations.

<!-- PAGEBREAK -->

### 10. Completion, recovery and next-chat prompt

Finish the implementation, focused checks, four pilot runs and locked evaluation. If an external interruption prevents completion, save the current update, full resumable state and exact next command; distinguish completed arms from pending work. Do not present a partial run as a completed comparison.

Publish the source and measured results under the project's existing authorization. Update `STATUS.md` with the chosen working checkpoint, failed/passed gates, remaining question and next command. Provide ordinary downloadable Markdown and PDF reports plus source/checkpoint archives. Include optimizer/RNG/stream state, configs, cohort manifests, predictions, timing, checksums and reproduction commands. Do not use Pages or Spaces.

Keep progress updates and the final chat summary short. Put detailed evidence in the report and machine-readable results.

**Suggested opening message for the next chat:**

> Execute the attached SHoEbox Query-Conditioned Lace Separation research plan through the four matched pilot arms and final evaluation. Recover the latest hippocampus binding source/checkpoints, inspect current STATUS.md and preserve intervening work. Implement the default-off low-rank question-conditioned lace bias, unique-lace reinforcement and selective contentious-overlap inhibition using existing paired-query activations and ordinary backward credit. Keep all model parameters trainable, preserve the global optimizer/source schedule and bounded CA1 repair, and retain the strong parent unless the established gates pass. Publish completed code/results under the standing authorization and attach ordinary Markdown/PDF reports and source/checkpoint archives. This request authorizes implementation and experiments, not just another plan.

### 11. Research relationship and sources

The motivating paper uses adversarial speculative drafts to trigger verification boundaries at important decisions. Its mechanism concerns numerical precision during speculative decoding. This plan adapts the broader idea of cheap competition controlling consequential computation; it does not reproduce the paper's decoding mechanism or import its output-distribution guarantee.

The intervention specified here comes from the user's proposal: reuse lace overlap and task feedback to strengthen distinct answer paths, inhibit contentious associations and encourage natural specialization. No adversarial draft model is required for the initial experiment.

1. Luo et al., *Adversarial Speculative Reasoning Improves Long Horizon Agents*, arXiv:2611.05345: [abstract](https://arxiv.org/abs/2611.05345), [paper](https://arxiv.org/pdf/2611.05345).
2. SHoEbox [inspected revision](https://github.com/JRowe47/SHoEbox/commit/f762518a30c7cc01cc308a4a362a3be73d62b52e) and [handoff](https://github.com/JRowe47/SHoEbox/blob/f762518a30c7cc01cc308a4a362a3be73d62b52e/STATUS.md).
3. [Binding report](https://github.com/JRowe47/SHoEbox/blob/f762518a30c7cc01cc308a4a362a3be73d62b52e/docs/hippocampus_binding_report.md) and [recovery protocol](https://github.com/JRowe47/SHoEbox/blob/f762518a30c7cc01cc308a4a362a3be73d62b52e/docs/hippocampus_binding.md).
4. [Reference lace controller](https://github.com/JRowe47/SHoEbox/blob/f762518a30c7cc01cc308a4a362a3be73d62b52e/experiments/hat_minkron/semantic_core.py), [runtime controller](https://github.com/JRowe47/SHoEbox/blob/f762518a30c7cc01cc308a4a362a3be73d62b52e/experiments/text_regions/runtime.py), and [node integration](https://github.com/JRowe47/SHoEbox/blob/f762518a30c7cc01cc308a4a362a3be73d62b52e/experiments/text_regions/node.py).
5. [Paired-question path](https://github.com/JRowe47/SHoEbox/blob/f762518a30c7cc01cc308a4a362a3be73d62b52e/experiments/text_regions/paired_recall.py) and [regional cue interface](https://github.com/JRowe47/SHoEbox/blob/f762518a30c7cc01cc308a4a362a3be73d62b52e/experiments/text_regions/hippocampus_interface.py).

**Expected outcome of this run:** a reproducible answer to whether inexpensive, question-conditioned lace competition improves binding and context switching, with a recoverable implementation whether or not the pilot passes promotion.

