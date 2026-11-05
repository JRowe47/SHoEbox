# SHoEbox integrated capability protocol

This round jointly trains the existing composition candidate on distributed
composition, delayed correction and factorized shared/private facts. It uses
one seven-node graph, its existing 384 physical experts, and one common affine
256-byte output head. No task identifier, answer, or capability-specific
decoder is added to inference.

## Parent and budget

The parent is the selected second-lookup checkpoint (phase 500, regional update
1,100), SHA-256 `f0abd8d038d5c7dc047d2d3dd4d7a81794bec003212727f92407caa379f80ed8`.
The parent weights, AdamW moments, composition sampler and Torch RNG are
preserved. New correction and view streams start at their first episodes.

Declare 200 joint optimizer updates, batch 64 composition cases and batch 16
per ordered stream, truncated backpropagation every 16 bytes, constant learning
rate 0.0005, clipping at 1, and one Torch thread. Each optimizer update averages
three task losses before stepping: composition, one correction segment and one
shared/private segment. The existing composition source reconstruction weight
0.5 and regional supervision weight 1.0 remain active. The two other tasks use
query cross-entropy plus the graph's existing routing auxiliary losses.

Composition continues its original 296 whole-table training pairs. Correction
uses the existing one-fact/update fixture with gap 16 and intermediate query
markers every four bytes. Views use the existing 32 training tuples, gap 8,
independently rendered sources and balanced supported source subsets. Stream
states are separate from one another and from composition episodes, while all
learned parameters are shared. This is joint training on separate task episodes;
it does not combine all task formats in one uninterrupted episode.

All tasks use K=2 with delayed regional exchange. The composition prefix keeps
the previous independent four-byte B/C encoding at K=1. Its query uses K=2.
Memory and view streams run the whole graph for every event. Both stream states
detach after each segment; values and cursor positions persist.

## Validation-only selection

Validate at initialization and every 50 updates, with all 528 composition
cases, 64 fresh recall and correction episodes each, 64 episodes for each
held-out view query and complete 64-tuple independent source-control cohorts.
The validation inputs are fixed across checks. No held-out test scoring occurs
until selection completes.

A qualified checkpoint requires every previous composition gate (90% answer
and local lookup scores, 100% exact B/C tables before and after query), plus 90%
recall, correction, shared, left and right accuracy, and 90% for each supported
single-source answer: A/shared, A/left, B/shared, C/shared and C/right. Two
consecutive qualified validation checkpoints establish the stability target.

Selection is lexicographic: qualified checkpoints first; then checkpoints that
retain every composition gate; then highest minimum capability accuracy;
then highest macro accuracy; then lowest macro answer NLL. A candidate that
fails the composition gates cannot replace a preserved candidate. The initial
model is eligible, so a failed integration run cannot silently overwrite it.
Report all remaining gate failures, including when the selected model is the
initial parent. Also retain the candidate with the best minimum accuracy,
macro accuracy and NLL irrespective of composition gates, explicitly marked
as diagnostic rather than an accepted replacement. Preserve the final
checkpoint for diagnosis. The 200-update budget follows a disposable CPU
timing smoke test; its updated weights are discarded.

## Evaluation and causal controls

After selection, evaluate all 592 original held-out composition cases at
K=1/2/4, delayed exchange disabled, memory reset, and shuffled B with original
and counterfactual answers. Score P/Q lookups and source-table retention.

Score fresh test recall and correction at gaps 16 and 32, with state-reset
controls and a withheld-update correction control. Report old-answer recovery
when the update is absent. Score 192 test episodes per view query on the
unchanged held-out tuples. Use the independent full factorial cohort for
all-source, A/B/C-only, no-private and reset controls. Independent test controls
use seed 104729 above the validation-control seed, so their identifiers and
renderings are not reused in selection. Count expert executions, observations,
aligned events and supervised/final queries explicitly.

Inference selects the largest of all 256 logits. Training-only reconstruction
and regional heads are removed from the export; verify exact logit equality on
composition, correction and view cases. Preserve atomic checkpoints containing
optimizer, both runtime states/cursors, RNGs, counters and full history. An
interrupted/resumed run must match uninterrupted training exactly.

## Commands

```bash
python -m experiments.text_regions.integrated_train \
  --parent artifacts/integrated/parent.pt --out artifacts/integrated/joint \
  --steps 200 --stream-batch 16 --composition-batch 64 --segment 16 \
  --lr 0.0005 --validate-every 50 --validation-count 64 --threads 1

python -m experiments.text_regions.integrated_eval \
  artifacts/integrated/joint/best.pt --count 192 \
  --out artifacts/integrated/joint/evaluation.json

python scripts/check_all.py
```

This experiment leaves extra-round stability, comparable-budget one-pass
training, independent initializations and natural-text validation as subsequent
acceptance work. It does not change the default model configuration.

## Controlled gradient follow-up

The first 100 baseline updates leave the new skills near chance. A recorded
next-batch diagnostic at update 25 measures gradient norms of 14.09 for
composition, 813.70 for correction and 111.14 for views. A second, fixed
100-update run therefore tests one change: clip each unweighted task gradient
to norm 1 before averaging the three gradients. Retain the final global clip,
the same parent tensors and optimizer moments, learning rate, examples, stream
cursors, objective weights, validation inputs and selection rule.

This follow-up is declared after examining training/validation diagnostics and
before any held-out test scoring. Its matched comparison is the preserved
baseline update-100 checkpoint. The original run still completes 200 updates;
that endpoint represents additional compute and is labeled separately.
No inference component or auxiliary head changes. The script and its gradient
policy are recorded in every follow-up checkpoint. Tests verify identical
first-step observations, losses, execution counts and forward states, plus
exact interrupted/resumed training under the new gradient rule.

```bash
python scripts/train_integrated_balanced.py \
  --parent artifacts/integrated/parent.pt \
  --baseline-protocol artifacts/integrated/joint/protocol.json \
  --out artifacts/integrated/balanced --steps 100

python scripts/measure_joint_gradients.py \
  artifacts/integrated/diagnostic_step25.pt \
  --out reproduced_gradient_diagnostic.json
```

Compare accepted candidates across runs by the same validation selection key.
Keep a diagnostic candidate separately when no new checkpoint retains all
composition gates. Score the fixed matched endpoints as well as the selected
candidate; do not select a model from held-out test results.
