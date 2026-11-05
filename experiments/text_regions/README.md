# Recurrent text regions

This is SHoEbox's event-major reference implementation of the [development specification](../../docs/SHoE_Text_Development_Specification.md). It reuses the baseline's SymbolLevel, RecurrentChild, minKron, lace controller, and split expert banks. No old experiment is copied into this directory.

## Learned publication retention

`GraphConfig.selective_publication` names the nodes whose computed latent
controls a hard write/hold decision. It defaults to an empty tuple, preserving
legacy checkpoint shapes and behavior. Each enabled node adds one affine
gate (`width + 1` parameters). Holding retains the publication while the
node continues its ordinary sparse computation and recurrent update.

`return_publication_activity=True` returns `candidate` with shape `[B,T,K,D]`
and `logit`/`written` with shape `[B,T,K]` for configured nodes. The optional
`publication_overrides={name: 'always'}` or `'hold'` controls support ablations;
ordinary learned inference omits them. Write labels never enter `forward`.

The producer experiment extracts node A and its existing ingress/shared banks
from the selected gradient-horizon checkpoint. It compares learned and
unconditional publication at matched initialization, labels and expert work.
See the [locked protocol](../../docs/SHoEbox_Selective_Publication_Protocol.md),
[report](../../docs/SHoEbox_Selective_Publication_Report.md), and
[raw results](results/selective_publication).

```bash
python scripts/run_selective_publication.py
python scripts/audit_selective_publication.py --evaluations
```

The artifact archive includes resumable `last.pt`, validation-selected
`best.pt`, and compact `inference.pt` for each run. Inference exports contain
`config`, `weights`, and `publication_policy`. Load the first two into
`RegionLanguageModel(GraphConfig.from_dict(saved['config']))`; use an
`always` override for the control export and no override for the learned
export. Both use the same 256-byte decoder and return ordinary `GraphState`.

## Learned regional composition

The composition phase trains the graph on `g(f(x))`: B and C independently encode two relation tables, then A receives the query. All seven nodes execute during each query round. Evaluation compares K=1,2,4 from the same encoded prefix, with delayed-exchange, shuffled-bridge, counterfactual-answer and state-reset controls. One shared affine byte head reads the full P/Q outputs; it is trained jointly with the graph.

See the [composition protocol](../../docs/SHoEbox_Composition_Protocol.md) for the exact source boundaries, held-out table combinations, lookup curriculum, checkpoints, and compute comparison. Run:

```bash
python -m experiments.text_regions.composition_train \
  --out artifacts/composition_seed1 --steps 1000 --batch 64 --seed 20261105
python -m experiments.text_regions.composition_eval \
  artifacts/composition_seed1/best.pt --split test --controls \
  --out artifacts/composition_seed1/test.json
```

`GraphConfig(readout_mode='concat')` enables the common affine head over concatenated publications. The default `mixed` setting retains the previous mixing projection, normalization, decoder dimensions and parameter initialization. Existing configurations that omit `readout_mode` load as `mixed`.

Composition training also supports `--copy-warmup 100 --source-reconstruction 0.5`. The former begins with training-only inverse-table pairs, whose unchanged composition target equals the query. The latter trains source representations to reconstruct their own observed table bytes from pre-query and post-query states. The small shared reconstruction head is not used by the prediction path. Checkpoints restore its parameters and optimizer state exactly.

## Recovery package

The cohort-diversity report survived; its implementation, expanded features, and 432 fitted heads did not. Use the [recovery report](../../docs/SHoEbox_Recovery_Report.md) for the available files and exact missing revision. The recovered-artifacts ZIP restores all twelve frozen graphs and the preceding shared-preservation study, including its dependencies and four longer-fit diagnostics.

The preservation audit accepts at most 1e-12 absolute/relative roundoff in rebuilt float64 normalization and named loss fields. It still checks stored file hashes, predictions, chosen strengths and penalties, eligibility flags, and correct counts exactly.

```bash
python -m experiments.text_regions.preservation_compare \
  --source artifacts/shared_preservation_v1 \
  --convergence artifacts/preservation_convergence_v1 \
  --out artifacts/reproduced_preservation_comparison.json
```

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

The policy comparison uses 512 fresh episodes per query (1,536 per control), with original/gated training paired at seeds 20261106, 20261107, and 20261108. These are fine-tuning seeds from a common correction-trained graph. Mean B-only shared recovery improves from 35.1% to 85.1%; all-view held-out accuracy falls from 78.2% to 68.2%, with private recovery regressing in every pair. The publication gate remains default-off. See the [full comparison and diagnostic report](../../docs/SHoEbox_Evidence_Policy_Report.md) and [raw results](results/evidence_policy). The baseline-gradient ablation below follows this comparison; learned composition and additional regional rounds remain separate experiments.

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

The completed gated three-seed comparison improves mean held-out accuracy from 68.2% to 77.6% and private accuracy from 57.9% to 73.7%, while B-only recovery falls from 85.1% to 67.3%. Seed 1's held-out left accuracy falls from 63.5% to 35.7%. The original-policy seed-1 control improves overall accuracy from 79.0% to 84.8%. Detachment remains optional; source-observability masking below tests the narrower intervention. See the [report](../../docs/SHoEbox_Baseline_Gradient_Report.md).

```bash
python -m experiments.text_regions.baseline_compare \
  --manifest experiments/text_regions/results/baseline_gradient/manifest.json \
  --out artifacts/reproduction/baseline_comparison.json
```

## Source-observed baseline targets

`view_train --baseline-observed-only` retains baseline encoder gradients for targets the selected source has actually observed. `ViewBatch.baseline_observed_mask(view)` derives eligibility from causal shared/private observation metadata, query type, availability and resets. A coincidentally equal value in a different field does not establish knowledge of the requested fact. The mask is training-loss metadata and never enters graph inputs, publications, routing, or runtime state.

The loss numerator excludes unsupported targets; the denominator remains the original count of supervised positions with an available baseline source. Loss weights, optimizer settings, utility reference losses and utility eligibility are unchanged. This option is independent of `--baseline-detach-features`; the primary observability ablation keeps features attached in both arms. Both options default to false and preserve old checkpoints.

```bash
python -m experiments.text_regions.view_train --steps 240 --batch 32 \
  --segment 16 --gap 8 --dropout 0.25 --seed 20261106 --threads 2 \
  --base-checkpoint artifacts/probes/correction_warm150/checkpoint.pt \
  --view-objective --baseline-observed-only --evidence-gated-ports \
  --out artifacts/baseline_observed/gated_seed1

python -m experiments.text_regions.view_score \
  artifacts/baseline_observed/gated_seed1/checkpoint.pt --publication-policy evidence \
  --count-per-query 512 --batch 64 --threads 2 --predictions \
  --out artifacts/baseline_observed/gated_seed1/evaluation
```

Step metrics, periodic histories and cumulative summaries include eligible, observed, used and suppressed baseline target counts by query. `recorded_segments` and `unrecorded_segments` distinguish complete audits from a legacy checkpoint resumed without earlier counters. Exact resume preserves both independent switches and the audit boundary.

`observability_compare.py` uses the same comparison engine as `baseline_compare.py`. Its manifest names `original` and `observed` arms at a declared publication policy, with optional `detached` results for context. The primary pair may differ only in `baseline_observed_only`, with detachment disabled. The comparator independently reconstructs causal target eligibility from source fact positions, availability and resets, and checks the recorded count audit even when general coverage reconstruction is disabled. Detached context is separately validated against the original arm.

The completed three-seed gated comparison yields 68.6% held-out overall, 61.0% private and 70.3% B-only accuracy, versus 68.2%, 57.9% and 85.1% for the original baseline. The seed-1 original-policy control also falls from 79.0% to 71.2% overall. Observed-only supervision remains default-off; the balanced-source experiment below tests supported whole-source visibility without changing loss settings. The [report](../../docs/SHoEbox_Source_Observability_Report.md) includes exact target audits and all three baseline variants.

```bash
python -m experiments.text_regions.observability_compare \
  --manifest experiments/text_regions/results/baseline_observed/manifest.json \
  --out artifacts/reproduction/observability_comparison.json
```


## Balanced supported source exposure

`view_train --source-schedule balanced --dropout 0` rotates complete source descriptions through supported subsets. Shared queries use all seven nonempty combinations; left-private queries retain A in each of A, AB, AC and ABC; right-private queries retain C in C, AC, BC and ABC. Missing sources remain absent for the whole episode. The default `dropout` schedule retains the existing behavior.

The schedule uses query-local episode ordinals and a seeded subset permutation. Each group of four answers receives the same subset; another rotation across the 32-episode tuple cycle prevents a factor tuple from being locked to one subset. Over 1,280 episodes per query, each shared subset receives 180 or 184 episodes, exactly balanced across answer values, and each private subset receives 320. Episode identifiers, fact values, text renderings and query positions are identical to the matched dropout stream. The schedule changes availability only and has no model parameters.

```bash
python -m experiments.text_regions.view_train \
  --steps 240 --batch 32 --segment 16 --gap 8 \
  --dropout 0 --source-schedule balanced --seed 20261106 --threads 2 \
  --base-checkpoint artifacts/probes/correction_warm150/checkpoint.pt \
  --view-objective --evidence-gated-ports \
  --out artifacts/source_schedule/gated_seed1

python -m experiments.text_regions.view_score \
  artifacts/source_schedule/gated_seed1/checkpoint.pt --publication-policy evidence \
  --count-per-query 512 --batch 64 --threads 2 --predictions \
  --out artifacts/source_schedule/gated_seed1/evaluation
```

The primary experiment retains the original baseline loss (`baseline_detach_features=false`, `baseline_observed_only=false`), view objectives on and utility commits off. It compares both publication policies at three matched fine-tuning seeds, keeping optimizer steps, aligned events, expert executions, initializer and loss weights fixed. Source occurrences and auxiliary-label counts intentionally change and are reported separately.

`source_exposure` records episode/final-query counts by query and retained subset, supported final queries, source occurrences by view, pooled private-label eligibility, available A/C samples and jointly available A/C pairs. Counts cover each actual training chunk. Checkpoints preserve the schedule and audit; legacy checkpoints default to `dropout` and mark prior segments as unaudited. Exact resume rejects a changed schedule.

This first curriculum uses only whole-source removal. Query-only or identifier-only views require causal shared-fact eligibility in the paired-view losses before they can be added.

`source_schedule_compare.py` verifies matched initialization, parameters, loss settings, ordered training content, external events and expert executions. It reconstructs source and auxiliary eligibility from fact positions, availability and resets, verifies balanced subset/answer/tuple coverage, and recomputes evaluation scores from raw predictions. Cross-policy summaries retain paired seeds and report the schedule-by-publication interaction. Legacy runs without recorded exposure counters are explicitly marked as reconstructed only.

```bash
python -m experiments.text_regions.source_schedule_compare \
  --manifest experiments/text_regions/results/source_schedule/manifest.json \
  --out artifacts/reproduction/source_schedule_comparison.json
```

The completed six-pair study raises mean B-only shared recovery from **35.1% to 99.8%** under original publication and **85.1% to 96.5%** under evidence gating. Held-out overall accuracy falls from **78.2% to 61.5%** and **68.2% to 63.4%**; private accuracy falls from **75.6% to 50.1%** and **57.9% to 49.5%**. The default schedule remains unchanged. The [report](../../docs/SHoEbox_Balanced_Source_Report.md) records per-seed results, exact source/auxiliary exposure, and the next frozen-graph readout diagnostic.

## Frozen common byte decoders

`readout_diagnostic.py` captures actual regional publications at the description boundary and final query. It compares the 64-dimensional P/Q concatenation with the existing 32-dimensional mixed/normalized latent, reusing each frozen graph execution for both answer heads and all factor probes. One 256-byte linear decoder per representation serves every query and source combination. Query labels and source-control names never enter features or select answer heads.

Fitting uses all 15 supported query/source-subset cells, with equal loss mass per query type. Every model sees the same 960 fitting cases and 3,840 evaluation cases. Fixed factor probes are fitted separately at each boundary and evaluated both at their own boundary and across time. Graph tensors, module modes, RNG, and publication policy remain unchanged, and K stays one.

```bash
python -m experiments.text_regions.readout_diagnostic \
  artifacts/source_schedule/original_seed1/checkpoint.pt \
  --fit-per-cell 64 --count-per-query 128 --max-iter 300 \
  --out artifacts/readout_single

python scripts/run_readout_study.py --jobs 3 --threads 2 \
  --out artifacts/readout_reproduction
python scripts/package_readout_study.py \
  --source artifacts/readout_reproduction \
  --out artifacts/readout_reproduction_results
python -m experiments.text_regions.readout_compare \
  --manifest experiments/text_regions/results/readout_diagnostic/manifest.json \
  --out artifacts/reproduced_readout_comparison.json
```

Use new output paths when reproducing an existing study. Start from the complete downloadable package: it restores the twelve measured graphs, fitted readouts, and raw `results/readout_diagnostic/runs/` files needed by the saved-result comparator. GitHub contains the aggregate comparison and provenance. `load_diagnostic` restores fitted answer and factor heads using tensor-only loading; `capture_readout_features` produces their inputs without labels. The comparator supports both ordinary JSON results and their lossless gzip copies.

Across three original-policy/dropout seeds, B-only recovery is **34.4% existing / 70.1% refitted mixed / 85.7% refitted raw**. Held-out private accuracy is **76.3% / 73.2% / 73.7%**. Gated/dropout models reach **95.8% B-only and 60.9% private** with the raw decoder. The [full report](../../docs/SHoEbox_Frozen_Readout_Report.md) records all twelve models, probe-transfer results, budgets, and the next shared nonlinear readout experiment.

## Shared emission adapters

`emission_adapter.py` fits one 64-unit hidden layer and one 256-byte output layer to frozen raw or mixed features. The matched linear control uses identical tensors and fitting settings with identity activation; the nonlinear arm uses GELU. Query and source labels supply loss weighting and reporting subsets only. `EmissionAdapter.logits(features)` accepts the observed feature vector alone.

Three initialization seeds per representation/activation yield twelve heads per frozen graph and 144 heads overall. Fitting uses the same 960 supported cases, equal query mass, training-only weighted normalization, L2 0.001, and a fixed 400-iteration L-BFGS budget. No evaluation result selects an initialization or setting. Captures exactly reproduce the prior direct and linear predictions before any adapter is fitted.

```bash
python scripts/run_emission_study.py \
  --out artifacts/emission_reproduction --jobs 3 --threads 2
python scripts/package_emission_study.py \
  --source artifacts/emission_reproduction \
  --out artifacts/emission_reproduction_results

# Audit the saved study from the complete downloadable package:
python -m experiments.text_regions.emission_compare \
  --source artifacts/emission_adapter_v1 \
  --out artifacts/reproduced_emission_comparison.json
```

The complete download restores original checkpoints, prior readouts, new fitted heads, compact feature caches and synthetic prediction rows. `load_adapters(path)` safely restores heads with tensor-only loading. Cached `features.pt` files retain `final/raw` and `final/mixed` observations for further fitting without rerunning the experts; their other fields are fitting/scoring metadata.

The audit recomputes every prediction and fitted training objective, checks original baseline agreement, reconstructs generated episode identities/targets, verifies normalization and matched parameter budgets, and checks frozen-model fingerprints and source hashes. Raw experiment rows stay in the downloadable package; GitHub contains aggregates and provenance.

Raw GELU improves balanced-source private accuracy to **62.8%** under both publication policies, versus **56.0% / 54.9%** for the previous linear heads. Original/dropout B-only accuracy reaches **97.6%**. Held-out shared accuracy regresses, motivating a residual correction around the existing linear head next. See the [report](../../docs/SHoEbox_Emission_Adapter_Report.md) for all conditions, initialization variation and optimizer budgets.

## Validation-scaled residual corrections

`residual_adapter.py` preserves each raw linear head and fits a 16-unit correction with identity or GELU activation. The output projection starts at zero, so initial logits exactly match the frozen baseline. Normalization remains fixed. There are 5,392 trainable correction parameters and 16,640 frozen base parameters; inference consumes one observed raw feature vector and emits all 256 byte logits.

The fitting/selection protocol is fixed before evaluation: 960 supported training cases, then 1,920 validation cases on separate fact combinations. Validation selects one strength from `0, 0.125, 0.25, 0.5, 1`, minimizing query-balanced cross entropy while preserving all-source/shared and B-only/shared correct counts relative to the frozen head. Zero is always feasible. Each selection is written to `selection_lock.json` before test scoring. Training, validation and evaluation identities are disjoint.

```bash
# Restore the previous complete emission package, then fit all twelve models.
python scripts/run_residual_study.py \
  --out artifacts/residual_reproduction --jobs 3 --threads 2
python scripts/package_residual_study.py \
  --source artifacts/residual_reproduction \
  --out artifacts/residual_reproduction_results

# Or reproduce all saved results directly from the complete residual package.
python -m experiments.text_regions.residual_compare \
  --source artifacts/residual_adapter_v1 \
  --out artifacts/reproduced_residual_comparison.json
```

The completed study fits 72 corrections: twelve frozen graphs, two matched activations, and three initializations. It reuses prior training/evaluation feature caches and adds only validation capture: 48,384,000 expert executions in total. Fitting makes no additional expert calls. All fits use the same 400-iteration cap and L2 0.001; all reach the cap. `load_residuals(path)` restores tensor-only artifacts and checks the frozen base, dimensions, normalization, fitting metadata, and selected strength.

Selected GELU raises B-only recovery from **85.7% to 90.1%** on original/dropout graphs and **95.8% to 97.3%** on gated/dropout graphs. Validation chooses zero in 22 of 36 GELU fits; private accuracy remains essentially at the baseline. Full-strength corrections recover balanced-source private gains but again lose shared accuracy. The default remains unchanged. The [report](../../docs/SHoEbox_Residual_Adapter_Report.md) records every condition, validation decision, seed variation, and the next training-objective experiment.

The artifact audit reconstructs cohorts, training objectives, validation selection and every saved prediction/loss. Prior raw-linear predictions reproduce exactly. Raw rows and tensors are delivered in the downloadable package; only code, reports and aggregate provenance are published. Prediction exports are atomic and hashed before a run is marked complete.

## Shared-answer preservation during training

`shared_preservation.py` adds a squared margin-shortfall loss on shared training
examples answered correctly by the frozen baseline. The target margin is capped
at 1 nat. Private examples and incorrect baseline answers remain supervised by
the ordinary byte objective. Query metadata defines anchors during training only;
every deployed head still predicts from observed raw features alone.

The study fixes two penalty coefficients (1 and 10), two matched activations,
three initializations and twelve graphs: 144 new fits at 400 L-BFGS iterations.
The architecture, frozen baseline, normalization and 5,392 trainable parameters
match the previous residual experiment. Validation selects both coefficient and
scale with the existing shared-answer guards. All caches are reused, with no
additional graph executions. The default model is unchanged.

```bash
python scripts/run_preservation_study.py \
  --out artifacts/preservation_repeat --jobs 3 --threads 2

# Audit the complete downloadable package, including four longer-fit checks:
python -m experiments.text_regions.preservation_compare \
  --source artifacts/shared_preservation_v1 \
  --convergence artifacts/preservation_convergence_v1 \
  --out artifacts/reproduced_preservation_comparison.json

# Independently repeat the training-gradient-selected convergence diagnostic:
python scripts/check_preservation_convergence.py \
  --source artifacts/shared_preservation_v1 \
  --out artifacts/convergence_repeat
```

`load_residuals(path)` restores both penalty candidates and `selected.pt`.
Selected artifacts retain every candidate's validation scores and the coefficient
decision in `head.selection['preservation_choice']`. The audit reconstructs the
objective independently and verifies all predictions from saved tensors.

Selected GELU improves balanced-source private accuracy from **56.0% to 57.7%**
under original publication and **55.2% to 55.9%** under gating. Shared accuracy
changes from **80.5% to 79.4%** and **87.4% to 87.5%**, respectively. Nineteen
of 36 GELU selections remain zero. All 295 tests pass. See the
[report](../../docs/SHoEbox_Shared_Preservation_Report.md) and
[audited aggregates](results/shared_preservation) for complete results.

## Decoder cohort-diversity reproduction

The unavailable cohort phase has been rebuilt from its preserved protocol and
rerun with the original twelve frozen graphs. Each graph receives nested training
cohorts of 64 and 256 examples per supported source/query cell (960 and 3,840 total),
penalties 0/1/10, identity/GELU corrections and three initializations: 432 fits.
The inherited linear base, its normalization and all validation/evaluation
caches are fixed. Smaller-cohort features are exact slices of the fresh larger
capture. Fitting and selection use the existing preservation/residual mechanisms.

Extract `SHoEbox_cohort_reproduction_artifacts.zip` before auditing or repeating
the study. The source-only archive includes tests and published aggregate results,
but lacks the required graph checkpoints and caches.

```bash
# Audit every objective, selection, prediction and retained legacy output:
python -m experiments.text_regions.diversity_compare \
  --source artifacts/cohort_reproduction_v1 \
  --out artifacts/reproduced_cohort_comparison.json --jobs 4 --threads 2

# Fit a fresh repetition into a separate directory:
python scripts/run_diversity_study.py \
  --out artifacts/cohort_repeat --jobs 4 --threads 2

# Continue an interrupted repetition with the identical inputs and source:
python scripts/run_diversity_study.py \
  --out artifacts/cohort_repeat --jobs 4 --threads 2 --resume

python scripts/check_all.py
```

Each graph saves `training.pt`, an atomically replaced `fitted.pt` after every
completed fit, `invocation.json`, and final results/predictions. Resume verifies
runtime-source and input hashes, fitting settings, capture identity and frozen
base tensors before using saved heads. The new study's
[audited aggregates](results/cohort_reproduction) include the original-report
comparison and verification logs. The [report](../../docs/SHoEbox_Cohort_Reproduction_Report.md)
and its PDF document the results and exact historical recovery boundary.

## Full-rank refits and compiled readouts

The next decoder phase fits a common 64-feature, 256-byte affine decoder on the
same nested 960/3,840 cohorts. Graphs and normalization remain frozen. The 72
new fits cover twelve graphs, two cohort sizes and penalties 0/1/10. Fitting is
deterministic; validation selects strength and positive preservation penalty.
The independent audit also reconstructs all 432 earlier identity/GELU fits.

```bash
python scripts/run_full_rank_study.py --out artifacts/full_rank_v1 --jobs 4 --threads 2
python -m experiments.text_regions.full_rank_compare --export-compiled \
  --out artifacts/full_rank_comparison.json
python scripts/benchmark_full_rank_readout.py --out artifacts/full_rank_benchmark.json
```

Use the full artifact archive for the fixed graph/feature dependencies. To
repeat fitting, pass a new `--out`; add `--resume` after interruption. Resuming
checks source, settings, input hashes and frozen-base identity. For a different
study directory, pass the same `--source` to the audit and timing scripts.

`compiled_readout.py` provides an optional frozen `nn.Module` with no query or
source-label inputs. It folds normalization and affine corrections offline.
An identity correction or full-rank interpolation requires one affine matrix;
GELU retains its nonlinear correction. Every export is checked against its
reference decoder on both training cohorts, validation and all evaluations.
The graph's default decoder is unchanged.

```python
import torch
from experiments.text_regions.compiled_readout import restore_compiled

export = torch.load(
    'artifacts/full_rank_v1/original_dropout_seed1/compiled.pt',
    map_location='cpu', weights_only=True)
head = restore_compiled(export['heads']['256/identity/20261121'])
# raw_pq: [batch, 64], the concatenated final P/Q feature vector.
# This exported head must be paired with the same frozen graph and feature layout.
with torch.inference_mode():
    logits = head(raw_pq)       # [batch, 256], float64
    answer_bytes = logits.argmax(-1)
```

The compiled buffers are independent copies and support PyTorch device/dtype
moves. Published equivalence and CPU timings use float64; converting dtype is
a separate numerical change. The artifact keeps graph and feature provenance
alongside the buffers. See the [full-rank report](../../docs/SHoEbox_Full_Rank_Readout_Report.md)
and [aggregate results](results/full_rank).

## Lookup mastery and regional supervision

The regional-supervision round reached **75.00%** on 592 held-out relation-composition queries, versus 47.30% for a matched continuation and 33.78% previously. See the [report](../../docs/SHoEbox_Lookup_Report.md) and [complete protocol](../../docs/SHoEbox_Lookup_Protocol.md).

`composition_curriculum` implements separate consecutive lookup gates and checkpointed stage transitions. `composition_train --regional-supervision 1` adds a shared training-only head at P round one and Q round two. `--initialize` starts a new continuation from a saved composition model; `--resume` restores the complete current run.

`RegionLanguageModel(..., return_round_activity=True)` exposes differentiable readout-node payloads `[B,T,K,width]` and validity `[B,T,K]` under `round_activity`. The observations preserve predictions, state and expert counts. The export audit in `results.lookup_mastery.audit` removes both training-only heads and verifies unchanged inference.

## Second lookup continuation

The subsequent continuation reaches **95.78%** held-out composition, 96.45% local P first lookup and 96.11% local Q second lookup, with both complete source tables retained in every case. The architecture and losses are unchanged. See the [report](../../docs/SHoEbox_Second_Lookup_Report.md), [protocol](../../docs/SHoEbox_Second_Lookup_Protocol.md) and [audit](results/second_lookup/audit.json).

`composition_continue` restores optimizer moments, counters and both RNG states from its parent. `optimizer_step_offset` keeps the existing learning-rate schedule. Full validation checks answer, lookup, regional and source-retention capabilities; checkpoint selection uses the lowest validation NLL among qualifying checkpoints. Two consecutive passes establish a separate stability gate. The recorded run selects phase update 500 (regional optimizer update 1,100), and finishes its 600-update budget without passing the consecutive-check gate.

```bash
python -m experiments.text_regions.composition_continue \
  --parent artifacts/second_lookup/parent.pt --out artifacts/second_lookup/continuation \
  --steps 600 --validate-every 100 --stop-on-mastery
python -m experiments.text_regions.composition_eval \
  artifacts/second_lookup/continuation/best.pt --controls --threads 1 \
  --out artifacts/second_lookup/continuation/test.json
python -m experiments.text_regions.results.second_lookup.audit \
  --root artifacts/second_lookup --out artifacts/second_lookup/audit.json
```

Append `--resume` to the same continuation command after an interruption. The artifact package contains the parent, full resumable states and an inference-only export. `selection.json` records the selected filename and hash; `history.json` distinguishes local phase steps from regional optimizer updates. Full-cohort evaluation, exact resumption and guarded selection are covered by the six new tests; all 342 repository tests pass.

## Integrated capability training

`integrated_train` trains composition, delayed correction and factorized views
in every update, with separate carried stream states and one shared graph and
byte head. It preserves the parent optimizer moments and RNG state, uses atomic
complete checkpoints, and supports exact `--resume`. Validation selection first
requires full capability qualification, then preserves all composition gates,
then maximizes the weakest capability, macro accuracy and negative macro NLL.

The 200-update baseline and matched 100-update per-task gradient-cap follow-up
are complete. Held-out composition is 99.83% for balanced 100 versus
94.43% for baseline 100. Baseline 200 is selected by validation and
scores 97.97% composition. The additional tasks remain near chance, so
full integration is not mastered. Balanced updates 50 and 100 both preserve
every composition validation gate. No inference capacity is added.

`integrated_eval` evaluates the same checkpoint on full composition cohorts,
fresh recall/correction at gaps 16/32, held-out factor combinations and an
independent full-factorial source-control cohort. All answers are argmax over
256 bytes. Pure inference exports remove the existing training-only source
and regional heads with exact logit/state parity checks.

See the [report](../../docs/SHoEbox_Integrated_Report.md),
[protocol](../../docs/SHoEbox_Integrated_Protocol.md) and
[complete training/evaluation/audit commands](results/integrated/README.md).
All 349 repository tests pass. The artifact package includes both full runs,
the parent, the matched baseline-100 snapshot, diagnostic checkpoint 25,
inference-only exports, raw predictions and the audited validation selection.
