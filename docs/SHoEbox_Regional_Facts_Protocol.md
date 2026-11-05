# Regional retained facts and corrections

This phase tests one change: training-only local supervision for retained facts
and corrections, alongside the integrated composition, correction and view
objectives. The inference graph, common 256-byte decoder, expert capacity,
round count, stream data and optimizer settings are unchanged.

## Locked comparison

Both arms continue from the validation-selected integrated checkpoint at update
200 (`artifacts/integrated/selected.pt`). Preserve its weights, AdamW moments,
composition sampler, Torch RNG, both stream cursors and both carried states.
Use the inherited constant learning rate 0.0005 and cap each task's gradient at
norm 1 before averaging and clipping the merged gradient. Each arm performs
100 additional updates: 64 composition cases, one batch-16 correction segment
and one batch-16 balanced-view segment per update. Segment length is 16;
correction gap is 16 and view gap is 8. Evaluation occurs every 25 updates.

Both arms receive identical auxiliary heads with seed 20261108. Their addition
does not consume the parent's RNG. The control's local-loss weight is zero;
the intervention's is one. Existing optimizer moments remain intact. Only the
new heads begin without optimizer history. Checkpoints are atomic and save all
runtime state every ten updates, and retain every validation endpoint.

## Causal labels and local losses

A shared four-class memory head decodes the current observed value at every
node after each external event's final regional round. Labels begin only when
an actual `=A` through `=D` write occurs, change on an observed correction and
clear on reset. Withholding a correction leaves the old value as the target.
Compute labels before TBPTT slicing to preserve causal eligibility across
segments. The labels are never graph inputs.

A shared twelve-logit fact head decodes shared, left-private and right-private
values at every node. Each factor is supervised only if it was observed by a
source in that node's configured forward ancestry. Delayed edges do not expand
auxiliary eligibility. Thus B has shared targets only; P can retain A/B facts;
Q can retain B/C facts. Padding does not erase observed facts. Withheld values,
future facts, unsupported private factors and facts cleared by reset contribute
no local loss. Reset-position metadata survives slicing.

For each task, average cross entropy within each nonempty node/factor group,
then average the group means. Add this local loss to the existing query and
routing objectives before the task gradient cap. This adds 528 training-only
parameters at width 32. No extra graph executions are performed. All four
auxiliary heads (including the existing composition heads) are stripped on
export, with identical inference logits and recurrent states required.

## Validation, tests and held-out scoring

Use the existing composition answer/lookup/source-retention gates and integrated
selection key: full qualification, composition preservation, minimum capability
accuracy, macro accuracy, then lower macro NLL. Keep the initial checkpoint as a
valid fallback. Select independently within each arm using validation only.
Report the equal-budget final endpoints even if selection prefers an earlier
checkpoint. Held-out scores never choose a checkpoint or change the protocol.

Evaluate composition on all 592 held-out cases with K=1/2/4, exchange disabled,
state reset and shuffled-table/counterfactual controls. Evaluate memory at gaps
16 and 32, with query reset and withheld corrections. Evaluate held-out factor
combinations and independent full-factorial source controls, using 512 episodes
per memory condition or view query. The four-value
reference is 25%; answers are argmax over all 256 byte logits.

Held-out stream evaluation uses batches of 256, while composition retains its
original 64-case batching and table-shuffle controls. Capability summaries reuse
the already scored control cohorts. The summary assembly is checked against
both preserved integrated reports (all metrics and cohort contents, excluding
wall-clock timing). Batch-64 versus batch-256 predictions are checked on fresh
validation episodes before held-out evaluation. Each completed evaluation stage
is saved atomically and can resume independently of training.

Tests must cover causal correction labels, source observability, reset and
slice consistency, unchanged inference/costs, gradients into the graph,
auxiliary-head removal and exact interrupted/resumed training. The matched
audit must verify parent preservation, initial equality, stream/sampler
equality, per-update expert budgets and validation-only selection.

```bash
python -m experiments.text_regions.regional_train \
  --parent artifacts/integrated/selected.pt --regional-weight 0 \
  --out artifacts/regional_facts/control
python -m experiments.text_regions.regional_train \
  --parent artifacts/integrated/selected.pt --regional-weight 1 \
  --out artifacts/regional_facts/regional
```

Append `--resume` with the same arguments after interruption. `--stop-after`
ends a run at a recoverable checkpoint without changing the declared budget.
