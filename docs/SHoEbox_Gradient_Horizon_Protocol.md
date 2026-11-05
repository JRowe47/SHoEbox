# Complete-episode gradient horizon

Locked November 5, 2026, before either arm is trained or evaluated on test data.

The regional-facts phase found weak fact acquisition and loss of readable
memory between observed writes and later queries. This phase tests whether
16-event gradient truncation contributes to that failure.

## Matched comparison

Both arms start from regional-facts `control/best.pt` at phase update 100.
Preserve every weight, all AdamW moments and the composition sampler RNG.
The parent already contains identical, untrained memory/fact auxiliary heads.
Use their local loss with weight 1 in both arms. There are no new parameters.

Start fresh independent correction and balanced-view episode streams with
seed 20261109 and zero runtime state at each episode. The new episode cohort
is intentional: parent stream cursors and partial runtime states are not
continued. No stored examples are replayed.

Run 32 optimizer updates per arm. Each update contains 64 fresh composition
cases, 16 complete correction episodes (gap 16) and 16 complete view episodes
(gap 8). This is 2,048 composition cases and 512 complete episodes of each
stream task per arm. Both use the inherited LR 0.0005, equal task weights,
per-task gradient norm cap 1 and merged gradient cap 1.

Both execute the same 16-event chunks in the same order. The **truncated** arm
detaches all carried graph state between chunks. The **full** arm retains its
training graph until the complete episode loss is backpropagated. Every
episode starts with a reset. Neither arm changes the inference runtime.

Compute query cross entropy over all supervised positions of the complete
episode, then add local losses averaged within each supported node/factor
group and across groups. Auxiliary routing terms retain their original
activation-weighted means. This prevents chunk length or boundary placement
from changing the loss weights. Compare equal episodes and expert executions,
not equal wall time or peak training memory. Whole-episode objective weighting
and update frequency differ from the preceding phase in **both** arms; only
the matched pair isolates the gradient-horizon intervention.

## Selection and evaluation

Validate at steps 0, 16 and 32 on the same validation cohorts. Use the existing
composition/lookup/exact-source-retention gates and integrated selection key:
all gates, then composition preservation, weakest task, macro accuracy and
lower macro NLL. Initial checkpoints remain eligible. Save atomic recovery
checkpoints every four updates and retain each validation endpoint.

Lock checkpoint selection before test scoring. Report both matched endpoints
even if the initial checkpoint wins selection. Evaluate the selected candidate
as well if it differs from either endpoint. Use all 592 held-out composition
cases and 512 episodes per stream condition/query. Preserve K=1/2/4,
no-exchange, changed-table, reset, withheld-correction and independent
full-factorial missing-source controls. Evaluate memory at gaps 16 and 32.
Run frozen local-head diagnostics on 64 fresh test episodes per task.

Tests must establish identical forward values and budgets under detachment,
full-gradient agreement with an unsliced episode, removal of prefix gradient
only at the intended boundary, loss-only labels and exact interrupted/resumed
training. Audit identical initial parameters/moments, actual per-update data
hashes, budgets, validation-only selection and inference export equality.

```bash
python -m experiments.text_regions.horizon_train \
  --parent artifacts/gradient_horizon/parent.pt \
  --out artifacts/gradient_horizon/truncated --detach-between
python -m experiments.text_regions.horizon_train \
  --parent artifacts/gradient_horizon/parent.pt \
  --out artifacts/gradient_horizon/full
```

Append `--resume` with identical arguments after an interruption.
