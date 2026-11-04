# Recurrent text regions

This is SHoEbox's event-major reference implementation of the [development specification](../../docs/SHoE_Text_Development_Specification.md). It reuses the baseline's SymbolLevel, RecurrentChild, minKron, lace controller, and split expert banks. No old experiment is copied into this directory.

## Components

| File | Responsibility |
|---|---|
| `config.py` | Serializable topology, ownership, sharing, rounds; forward-DAG validation |
| `node.py` | Existing two-level symbols over latent inputs; existing three-block SHoE |
| `ports.py` | Identity, semantic namespaces, dependencies, stage/commit, snapshots, invalidation |
| `state.py` | Ingress/node state, clocks, reset, detach, serialization, byte accounting |
| `graph.py` | Directed value/context projections, learned gates, diagnostic substitutions |
| `model.py` | Event-major stages, delayed feedback, shared byte decoder, generation |
| `data.py` | Ordered per-row cursors, document boundaries, resumable positions |
| `train.py` | AdamW, inherited schedule/clipping, isolated validation, full checkpoints |
| `probes.py` | Seeded recall/correction, shared/private, and relation-chain fixtures |
| `episode_data.py` | Compact query-supervised recall/correction and aligned view streams |
| `probe_train.py`, `probe_eval.py` | Probe training, exact resume, held-out scoring, equal-execution controls |
| `objectives.py`, `learning.py` | Independent view losses, source baseline, explicit edge/lace utility commits |
| `verify.py` | Baseline parity and engine invariant tests |
| `evaluate.py` | Common measurements for baseline and graph |

## API

```python
import torch
from experiments.text_regions import GraphConfig, GraphState, RegionLanguageModel

torch.set_num_threads(2)
model = RegionLanguageModel(GraphConfig(rounds=2, recurrent_feedback=True))
x = torch.tensor([[65, 66, 67]], dtype=torch.long)
y = torch.tensor([[66, 67, 68]], dtype=torch.long)
result = model(x, y, trace=True)
result['loss'].backward()
state = result['state'].detach()  # after a training segment
more = model(torch.tensor([[68]]), state=state)
restored_state = GraphState.from_dict(state.to_dict())
generated, final_state = model.generate('AB', count=40)
```

The default is K=1 with recurrent edges disabled. Override `rounds` and `feedback` per call. To compare K=1,2,4 with the same recurrent model and observations, reuse the same unmodified prefix state and keep `feedback=True` for all calls. Feedback snapshots continue across external events, so K=1 with feedback enabled differs from the initial feedforward integration configuration.

Every activation updates both contextualizers and all three minKron blocks. Each node owns its state even when expert weights are shared. Forward calls do not update parameters or learning buffers. The default objective is byte cross-entropy plus routing/symbol balance. Optional M6 objectives and association commits are described below.

## Views and interventions

Ordinary input is a byte tensor or the baseline `ByteBatch`, preserving exact link/slot addresses. One HAT occurrence supplies the three branch adapters. Controlled inputs provide all configured views:

```python
views = {'A': x, 'B': x + 1, 'C': x + 2}
available = {'B': torch.zeros_like(x, dtype=torch.bool)}
result = model(views, availability=available)

# Information mask; node/expert execution counts stay unchanged.
edge_mask = {'A__U__forward': torch.zeros_like(x, dtype=torch.bool)}
result = model(views, edge_masks=edge_mask)

# Execution mask; skips node B while preserving its state/publication.
active = {'B': torch.zeros_like(x, dtype=torch.bool)}
result = model(views, node_masks=active)
```

Views have aligned `[B,T]` tensors and separate ingress state under shared HAT weights. Unavailable bytes never pass through HAT or advance its source clock. An available zero byte is an ordinary input. Source nodes can still execute with retained state or feedback when their view is absent; use node/edge masks to isolate those paths too.

`EdgeSubstitution(payload, memberships)` replaces both parts of a message, with shapes `[B,T,width]` and `[B,T,3,groups]`. Replacing both prevents original semantic evidence from leaking into an intervention. Dependencies mark substituted messages. Masks are boolean `[B,T]`; reset accepts `[B]` or `[B,T]`.

## State contracts

Publications contain payload, validity, source/port, per-row version, stream epoch, external occurrence, round, local memberships, selected group/lace IDs, and consumed dependency IDs. Ingress also retains link/slot IDs. Local coordinates are namespaced by `(node, block)` and translated by each edge's learned map.

Stages commit by replacing references. Existing readers keep their old tensors. Runtime code never mutates supplied state or publications. Frozen records do not make PyTorch storage read-only: callers must avoid in-place edits too. Dependencies are bounded IDs and masks, never recursive references to historical payloads. Batch-row position is part of stream identity and remains stable until reset.

`detach()` severs every carried tensor at a segment boundary while preserving values. Crossing a region or round does not detach. `reset(rows)` clears ingress matrices, node matrices, ports, memberships, validity, clocks, and dependency rows, and advances stream epochs. Inactive rows retain state and publications. Switching between text and controlled-view ingress requires fresh state.

`storage_bytes()` counts deduplicated allocated tensor storage; `tensor_bytes()` counts logical bytes including repeated references. Both exclude weights, gradients, optimizer tensors, and retained training activations. Last-occurrence provenance is copied so it cannot retain an entire input window.

## Training and exact resume

Run from the repository root:

```bash
python -m experiments.text_regions.train --tiny --context 8 --batch 4 \
  --steps 100 --out artifacts/overfit

# Preserve the 20-step schedule while stopping at step 5.
python -m experiments.text_regions.train --tiny --context 8 --batch 4 \
  --steps 20 --stop-after 5 --out artifacts/resume_check
python -m experiments.text_regions.train --tiny --context 8 --batch 4 \
  --steps 20 --resume artifacts/resume_check/checkpoint.pt --out artifacts/resume_check

python -m experiments.text_regions.train --data experiments/hat_minkron/data/tiny_shakespeare.txt \
  --rounds 2 --context 128 --batch 4 --steps 100 --out artifacts/language_k2
```

Data uses the existing contiguous 90/5/5 split. Validation owns fresh state, separate cursors, and an isolated RNG context; test data remains untouched. Each split is partitioned into nonoverlapping row shards. Chunks continue at the next byte; completed shards start new independent streams and reset that row. Targets never cross shard boundaries. Explicit document lists are also supported, including resets inside a chunk.

Checkpoints include configuration, weights, optimizer, graph state, ports, validity, clocks, stream positions, dataset fingerprint, Python/PyTorch RNGs, step, and schedule horizon. Exact resume rejects changes to data, batch, context, threads, rounds, LR, or horizon. Loading uses tensor/primitive-only serialization and `weights_only=True`.

Delayed message gates initialize at 0.05 and forward gates at 0.5; both remain trainable. This damps initial closed-loop transport while retaining gradients through every round. The original minKron recurrence and baseline initialization are unchanged.

## Verification and measurement

```bash
python scripts/check_all.py
python -m experiments.text_regions.evaluate --baseline --batch 4 --context 128 \
  --out artifacts/baseline_profile.json
python -m experiments.text_regions.evaluate --rounds 1 --batch 4 --context 128 \
  --out artifacts/graph_k1_profile.json
python -m experiments.text_regions.evaluate --rounds 2 --batch 4 --context 128 \
  --out artifacts/graph_k2_profile.json
python -m experiments.text_regions.probes --kind recall --split validation \
  --count 512 --distractors 32 --out artifacts/recall_validation.jsonl
```

Profiles separate unique parameters, repeated expert calls, and unique physical experts touched in the declared window. They measure CPU time, routing/expert forward components, state/optimizer bytes, and process peak RSS. Random-byte profiles test execution and gradients; learning runs use separately declared data. Full autograd over 128 bytes consumes substantially more memory than persistent inference state.

The original `probes.py` retains the richer distractor and relation-chain fixtures. The new `episode_data.py` provides compact shaping tasks: four possible byte values, explicit query markers, fresh split-specific identifiers, and optional corrected or private facts. Query outputs receive supervision; answers are not fed back as observations. Intermediate query markers provide within-segment training signals for writes and retention. Final-query evaluation ignores those intermediate labels and takes argmax over all 256 bytes.

Recall/correction evaluation compares carried state with a complete state reset immediately before the final query. Correction also replaces the entire update with neutral filler, keeping query positions and expert execution counts unchanged; it scores both the corrected answer and the old answer. Shared/private evaluation includes A-only, B-only, C-only and private-value-withheld controls. These masks remove information while retaining node execution. Each source has independent HAT state; missing bytes never become observations.

```bash
python -m experiments.text_regions.probe_train --kind correction --steps 300 \
  --batch 8 --segment 16 --gap 16 --eval-gap 32 --eval-episodes 512 \
  --eval-batch 64 --threads 2 --out artifacts/probes/correction_300
python -m experiments.text_regions.probe_eval artifacts/probes/correction_300/checkpoint.pt \
  --gaps 16 32 --split test --episodes 512 --out artifacts/probes/correction_test.json

# The schedule horizon remains 100 through stop/resume.
python -m experiments.text_regions.probe_train --kind shared_private \
  --view-objective --utility --steps 100 --stop-after 5 --out artifacts/probes/views
python -m experiments.text_regions.probe_train --kind shared_private \
  --view-objective --utility --steps 100 --resume artifacts/probes/views/checkpoint.pt \
  --out artifacts/probes/views
```

`--steps` counts TBPTT segments; logs separately count optimizer steps, supervised query outputs, external events, and expert calls. `--gap` is the filler count after the last write. Recall/correction's final value-to-query distance is `gap+2`. Results declare this distance explicitly. The training cursor resets at each new episode and detaches state between chunks. Schema-2 checkpoints include optional learning configuration and accept earlier schema-1 bare checkpoints. The evaluator starts independent state for each episode batch and never resumes a training cursor.

`--checkpoint-every 25` atomically saves recovery state during training as well as at the final step. `--resume` continues the exact schedule, cursor, optimizer and RNG. `--warm-start` loads matching model weights and learned utility buffers into a new training phase with a fresh optimizer, graph state, cursor and RNG; it permits a new training seed and records the initializer checkpoint. These modes are mutually exclusive. Use `--eval-split test` or `probe_eval --split test` for the independent test stream.

## Optional representation and contribution learning

`RegionLearningModel` wraps the ordinary graph. With no options it preserves parameter count, logits, losses, and state. Optional heads use `return_view_activity=True`, which exposes the HAT activations already produced by each source before any graph fusion. A/B/C node publications can contain peer feedback and are not source-only codes.

`PairedViewObjective` provides separately enabled agreement, cross-view prediction, variance, covariance, private-byte prediction, and shared/private separation terms. All are off by default. Each description pools its own last available occurrence (or independently masked mean); a longer description is not truncated to the other's mask. Paired terms require both sources. Per-view variance/covariance and private losses use that view's own available samples. Private labels enter only the loss head. The `probe_train` CLI's `--view-objective` enables A/C agreement, prediction, variance and covariance; private-detail recovery remains supervised through the main query loss. The private-code head and separation term are separately available through the Python API and enabled by the factorized `view_train` benchmark below.

`--utility` adds a trainable source-only byte baseline and detached signed-gain packets. The baseline uses text ingress for ordinary episodes and A ingress for controlled views. Eligible inter-node dependencies must have been consumed and must not be substituted. Lace eligibility records the actual pre-execution selection, separately from the post-execution observed lace. Packets contain bounded sufficient statistics, not examples or replay data. Each packet compares the current local baseline CE against regional CE and is committed explicitly after `optimizer.step()`. Subsequent passes may use bounded edge-gate and lace-score biases. Evaluation uses committed biases but never creates or commits feedback.

```python
from experiments.text_regions import RegionLearningModel
from experiments.text_regions.objectives import DelayedUtility

utility = DelayedUtility(tuple(model.connections.edges), tuple(model.nodes),
                         model.cfg.laces, enabled=True)
learner = RegionLearningModel(model, utility=utility)
optimizer = torch.optim.AdamW(learner.parameters(), lr=0.002)
optimizer.zero_grad(set_to_none=True)
result = learner(x, y, state=state)
result['loss'].backward()
optimizer.step()
learner.commit_feedback(result['utility_packet'])
state = result['state'].detach()
```

Save `learner.get_config()` with `learner.state_dict()`; rebuild via `RegionLearningModel.from_config(config)` before loading weights. `probe_train` handles this, optimizer state, source/graph state, cursor, and RNG together. Each auxiliary term is logged separately. Forward calls and `eval()` cannot commit learning buffers. These utility estimates are signed associations, not edgewise causal effects.

Missing views can produce exactly zero latent directions while their nodes continue to execute. `GraphConfig.zero_safe_norm=True` assigns zero derivatives to absent symbol/minKron query directions, preventing nested `1/epsilon` Jacobians from dominating training. Normalization values and gradients above epsilon are unchanged. The preserved baseline constructors retain their original defaults; the behavior is enabled explicitly by the new graph.

Three-seed comparisons and learned composition remain acceptance work. The transport test verifies that A cannot affect Q in round one and can in round two. Measured capability results are in the [probe development report](../../docs/SHoEbox_Probe_Development_Report.md).

See the [development report](../../docs/SHoEbox_Development_Report.md) and [raw results](results).

## Factorized shared/private benchmark

`view_data.py` adds three four-way factors: a shared fact in A/B/C, left-private information in A, and right-private information in C. Each source independently chooses a byte rendering and field order; descriptions are aligned through explicit availability masks. All evidence arrives before query markers at events 14/15. The first 16-event segment supplies local learning signals, followed by `gap` filler events and a final query. Gap 8 gives 25 events per episode. Query answers are labels only. Identifiers are unique opaque permutations independent of answer allocation.

Training uses 32 factor combinations, validation 16, and test the remaining 16. Every individual factor value and source-local pair occurs in training. A separate `independent` evaluation cohort exhausts all 64 combinations: it supplies information-removal controls with a true 25% missing-factor prior. Restricted combination holdouts can make a missing factor inferable from the others, so they are evaluated with all views present. Counts are balanced by query and answer; the recorded runs use 192 episodes per query, or 576 per cohort.

Training source dropout retains at least one source that contains the requested answer. Private auxiliary labels are available only after that source observes its value and are cleared by resets. Checkpoints reconstruct deterministic cursors and preserve model/head/utility weights, optimizer slots, graph state, RNG, counters, and schedule. A complete initializer checkpoint is saved before the first expensive pass, followed by atomic recovery checkpoints every 20 segments.

`view_train.py` holds module allocation and initial weights constant across all four view-objective/utility combinations. The source-only baseline is trained in every variant, isolating utility commits from the effect of adding a baseline loss. View learning enables agreement, prediction, variance, covariance, private-byte prediction and shared/private separation, with separate logged terms. The private loss weight is 0.05; other weights retain `ViewObjectiveConfig` defaults. All variants have 265,694 parameters including auxiliary heads, of which the underlying graph has 246,142.

```bash
# Extract the downloadable checkpoint bundle into this repository first.
python -m experiments.text_regions.view_train --steps 240 --batch 32 \
  --segment 16 --gap 8 --dropout 0.25 --seed 20261106 --threads 2 \
  --base-checkpoint artifacts/probes/correction_warm150/checkpoint.pt \
  --view-objective --out artifacts/views/views

python -m experiments.text_regions.view_score artifacts/views/views/checkpoint.pt \
  --count-per-query 192 --batch 64 --predictions --out artifacts/views/views/evaluation
```

Use no learning flags for `control`, `--view-objective` for `views`, `--utility` for `utility`, and both for `both`; use a distinct output directory for each. The four recorded runs have the same initial tensor SHA, examples, source observations, expert calls and optimizer-step count. `--stop-after` and `--resume` retain an unchanged schedule horizon. A bare graph initializer starts fresh optimizer and runtime state.

`view_score.py` evaluates fresh episodes without loading the training cursor into inference. It separately scores held-out tuples and the full-factorial control cohort, saves compact per-control predictions, and checks that evaluation leaves model/utility tensors unchanged. Controls include all views, A/B/C alone, private-value withholding, and reset immediately before the final query; expert execution counts remain identical.

Frozen ridge readouts fit 128 training descriptions, then score independent source ingress and projected codes at the common pre-query boundary. Fit and evaluation identities are disjoint. These probes measure decodability without altering the model. Direct private-head diagnostics use that same boundary, two events earlier than the usual first-chunk training pooling point; their timing distinction is recorded explicitly. Saved readouts can be restored with `load_source_readouts(path)` from `view_eval.py` using tensor-only loading.

## Require evidence before publishing

`GraphConfig.evidence_gated_ports` is an optional, default-off diagnostic. With it enabled, a node's publication becomes valid after it consumes a valid dependency; an already-valid publication remains backed by the node's retained memory. Selective resets clear that history. Valid delayed feedback can establish evidence in a source whose own view is absent. Node execution, clocks, state updates and expert counts continue normally.

This suppresses valid-looking prior outputs from nodes that have never received evidence. It also excludes invalid node publications from lace utility eligibility. Literal zero bytes remain observations when their availability mask is true. Existing checkpoints retain the original default behavior.

```bash
python -m experiments.text_regions.view_score artifacts/views/views/checkpoint.pt \
  --cohorts independent --controls b_only --evidence-gated-ports --skip-readouts \
  --count-per-query 192 --batch 64 --predictions --out artifacts/views/views/evidence_gated
```

The initial intervention uses unchanged weights: for the view-trained model, B-only shared accuracy rises from 24.5% to 66.7% on 192 shared queries while expert calls remain equal. Training with the policy is available through `view_train --evidence-gated-ports`. The original four-way comparison preserves the original publication policy; the subsequent comparison separates training and inference effects below.

## Training-policy comparisons and regional diagnostics

`view_score --publication-policy trained|original|evidence` selects the inference policy independently of the checkpoint's training policy. The default keeps the saved configuration. The earlier `--evidence-gated-ports` scoring option remains an alias for `--publication-policy evidence`. Neither override changes learned tensors or expert counts.

`evidence_compare.py` validates a manifest of matched original/gated training pairs. Each pair supplies original-trained/original-inference, original-trained/gated-inference, and gated-trained/gated-inference results; the fourth combination is optional. This separates inference-policy effects from learning with the policy. Comparisons check initialization, configuration, exposure, budgets, checkpoint fingerprints, episode identities, balanced answers, and scores recomputed from raw predictions. Seed aggregates keep view-objective and utility settings separate. Training source coverage is reconstructed from the saved stream, including rare supported singleton inputs.

`transfer_eval.py` fits frozen linear readouts on training descriptions and captures actual node publications at the pre-query boundary and final query. It reports both pooled fits and separate query-matched fits, along with publication validity, independent final source ingress, and the model's own answers. Invalid payloads are excluded. Saved readouts bind to the model tensors and full inference configuration, including publication policy; `load_transfer_readouts(path)` validates tensor-only artifacts before reuse. Labels and query metadata never enter the model's prediction features.

`query_interference.py` tests facts supplied only by B while A/C receive no observations, only the final shared-query marker, or all query markers. Fact values stay masked. The diagnostic verifies identical cases, byte tensors, and expert counts, reports source-observation counts separately, and restores the model's modes and policy after scoring.

```bash
python -m experiments.text_regions.view_score artifacts/evidence/views/checkpoint.pt \
  --publication-policy evidence --count-per-query 512 --batch 64 --predictions \
  --out artifacts/evidence/views/policy_evaluation/gated

python -m experiments.text_regions.transfer_eval artifacts/evidence/views/checkpoint.pt \
  --publication-policy evidence --fit-count 192 --count-per-query 192 --batch 64 \
  --out artifacts/evidence/transfer_trained_gated

python -m experiments.text_regions.query_interference artifacts/views/views/checkpoint.pt \
  --policy both --count 512 --batch 128 --out artifacts/evidence/query_interference

python -m experiments.text_regions.evidence_compare \
  --manifest experiments/text_regions/results/evidence_policy/manifest.json \
  --out artifacts/evidence/recomputed_comparison.json
```

The policy comparison uses 512 fresh episodes per query (1,536 per control), with original/gated training paired at seeds 20261106, 20261107, and 20261108. These are fine-tuning seeds from a common correction-trained graph. Mean B-only shared recovery improves from 35.1% to 85.1%; all-view held-out accuracy falls from 78.2% to 68.2%, with private recovery regressing in every pair. The publication gate remains default-off. See the [full comparison and diagnostic report](../../docs/SHoEbox_Evidence_Policy_Report.md) and [raw results](results/evidence_policy). The next ablation isolates encoder gradients from the A-only baseline loss; learned composition and additional regional rounds remain separate experiments.

## Baseline encoder-gradient ablation

`view_train --baseline-detach-features` disconnects the source-only baseline loss from its input features. The baseline head still trains on the same targets with the same weight and normalization. Its logits and utility eligibility are unchanged for fixed model weights. Main answer and view-objective losses still update the encoder normally. This removes **all** baseline-to-encoder gradients, including supported shared and left-private targets; it does not selectively remove right-private supervision.

The switch defaults to false and is stored in both learner configuration and training protocol. Older checkpoints without the field retain their original behavior. Exact resume requires the same setting; a fresh ablation starts from the common bare graph initializer.

```bash
python -m experiments.text_regions.view_train --steps 240 --batch 32 \
  --segment 16 --gap 8 --dropout 0.25 --seed 20261106 --threads 2 \
  --base-checkpoint artifacts/probes/correction_warm150/checkpoint.pt \
  --view-objective --baseline-detach-features --evidence-gated-ports \
  --out artifacts/baseline_detach/gated_seed1

python -m experiments.text_regions.view_score \
  artifacts/baseline_detach/gated_seed1/checkpoint.pt --publication-policy evidence \
  --count-per-query 512 --batch 64 --threads 2 --predictions \
  --out artifacts/baseline_detach/gated_seed1/evaluation
```

Compare attached and detached training at the same publication policy, initializer, source stream, optimizer settings, and compute budget. Utility is disabled in this ablation. Three gated-policy seeds measure the detach effect under that policy; the additional original-policy seed supplies a one-seed interaction check. It is not a three-seed comparison of that interaction.

`baseline_compare.py` accepts a version-1 manifest with `publication_policy`, `original`, and `detached` entries per pair. Each arm names a `run_directory` and `evaluation_directory`. It verifies all training settings except detachment, source exposure, parameter and expert budgets, exact evaluation cases and answer balance, then recomputes scores and paired improvements/regressions from raw predictions. Aggregates keep publication policies and auxiliary settings separate. Legacy missing detachment fields normalize to false while hashes retain the original bytes.
