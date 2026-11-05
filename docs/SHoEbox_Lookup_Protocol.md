# Lookup mastery and regional composition

This phase follows the regional-composition experiment at GitHub revision
`8efa47556fcdcff6c0a4e33c0f8cb8206f5d1759`. That model retains both source
tables but answers only 33.78% of held-out composition queries correctly.

## Mastery curriculum

The network, byte interface, source encoding and whole-pair split are unchanged.
The two lookup orientations are scored separately:

- B-side lookup: a nonidentity table f at B and identity g at C.
- C-side lookup: identity f at B and a nonidentity table g at C.
- Full composition: both tables are nonidentity.

The identity/identity pair is excluded from these skill groups. Every sampled
pair contributes all four queries. Each lookup batch has equal numbers of the
two orientations. Training uses only pairs assigned to the training split.

| Skill | Training cases | Validation cases | Test cases |
| --- | ---: | ---: | ---: |
| B-side lookup | 40 | 32 | 20 |
| C-side lookup | 48 | 16 | 28 |
| Full composition | 1,092 | 480 | 544 |

The mastery's default threshold is 90% in **both** lookup orientations at two
consecutive validation checks. A failed check clears the streak. A maximum
training budget does not override the gate. Validation evaluates every unique
case in each group; the test split is reserved for final scoring.

After the gate opens, half of the training batch consists of newly sampled
lookup examples and half of full-composition examples. No stored examples or
replay buffer are used. The regular all-256-byte answer loss, routing losses,
and optional source reconstruction train the complete graph.

The model retains a lowest-lookup-NLL checkpoint. Once composition training
begins, a separate checkpoint is selected by composition validation NLL only
when both lookup accuracies still meet the threshold. The exact transition
checkpoint and the last checkpoint are also retained. Checkpoints include the
curriculum state, consecutive-check streak, optimizer, RNGs, counters and
source hashes.

## Initialization and reproducibility

`--initialize` starts a new phase from the prior source-reconstruction model.
It imports model tensors, resets the optimizer, and records the initialization
file's SHA-256. This is continued development of the same learned network,
not a fresh seed replication. `--resume` continues the current phase exactly,
including optimizer and curriculum state, and rejects protocol/source changes.

```bash
python -m experiments.text_regions.composition_curriculum \
  --out artifacts/mastery/gated_seed1 \
  --initialize artifacts/mastery/parent.pt \
  --steps 1500 --batch 64 --lr .001 --threads 1

python -m unittest -v experiments.text_regions.test_composition_curriculum
```

The parent is the validation-selected `composition_reconstruction_seed1/best.pt`
from the previous phase. `--stop-after` permits bounded checkpoints without
changing the planned protocol; use the same arguments with `--resume` to
continue. Final claims and any additional learning changes belong in the phase
report, alongside the actual completed budgets and results.

## Regional supervision follow-up

The first bounded mastery run stopped at step 300 with its gate closed. Its
training lookup accuracy improved, but validation lookup did not generalize.
The follow-up therefore teaches the two regional operations directly during
training, using all original training pairs rather than only the small identity
subsets.

An optional differentiable activity output records P/Q publications after each
complete query round, shaped `[batch, events, rounds, width]`. Enabling it does
not change graph inputs, state, predictions or expert executions. One shared
132-parameter affine training head maps P after round one to `f(x)` and Q after
round two to `g(f(x))`. The average four-class cross-entropy of those two
predictions is added with weight 1.0. The original all-256-byte answer head is
still trained and used for inference. Neither target is supplied as an input,
publication, feedback message or replacement state. The auxiliary head is not
used at inference.

The follow-up and its control both start from the previous phase's selected
model, reset AdamW, use seed 20261105, train for 600 steps with batch 64, and
draw identical original training-pair sequences. Both use LR 0.001 after a
20-step warmup, source reconstruction weight 0.5, K=2, the same routing losses
and clipping at 1. Only the regional loss/head is added in the follow-up. Each
run has 38,400 training episodes and 10,137,600 expert executions. The extra
auxiliary head adds arithmetic but does not change expert execution counts.

This comparison uses full-composition training from the first step; the lookup
gate remains the acceptance diagnostic, rather than a claim that the failed
curriculum passed. Both checkpoints are selected by lowest answer NLL on all
528 validation cases, checked every 100 steps, before any test scoring.

```bash
python -m experiments.text_regions.composition_train \
  --out artifacts/mastery/regional --initialize artifacts/mastery/parent.pt \
  --steps 600 --batch 64 --lr .001 --lookup-warmup 0 \
  --source-reconstruction .5 --regional-supervision 1 \
  --validation-count 528 --validate-every 100 --threads 1

# Matched continuation control: identical command, omit regional supervision.
python -m experiments.text_regions.composition_train \
  --out artifacts/mastery/control --initialize artifacts/mastery/parent.pt \
  --steps 600 --batch 64 --lr .001 --lookup-warmup 0 \
  --source-reconstruction .5 \
  --validation-count 528 --validate-every 100 --threads 1

python -m experiments.text_regions.composition_eval \
  artifacts/mastery/regional/best.pt --split test --controls \
  --out artifacts/mastery/regional/test.json
```

The original mastery runtime is preserved separately because exact resumption
rejects source changes. Use that snapshot for the original gated run. The
paired follow-ups use the final regional-supervision runtime.

## Diagnostics, audit and inference export

Run the following commands for both `regional` and `control` after training:

```bash
python -m experiments.text_regions.results.lookup_mastery.diagnostics \
  artifacts/mastery/regional/best.pt \
  --out artifacts/mastery/regional/diagnostics.json

python -m experiments.text_regions.results.lookup_mastery.audit \
  --root artifacts/mastery --out artifacts/mastery/audit.json
```

The audit checks initial tensors, identical final sampling RNG, completed
budgets, validation-only checkpoint selection, raw prediction counts, source
hashes and execution accounting. It exports each `inference.pt` after removing
both training-only heads, and requires exactly unchanged ordinary graph logits
on 64 held-out cases. The exported model contains 252,478 parameters.

```python
import torch
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel

saved = torch.load('artifacts/mastery/regional/inference.pt', weights_only=True)
model = RegionLanguageModel(GraphConfig.from_dict(saved['config']))
model.load_state_dict(saved['weights'])
model.eval()
```

Use the existing `encode_sources` and `answer_query` functions for the relation
fixture. The ordinary model interface also accepts byte streams; this training
phase evaluates the four-symbol relation task, rather than natural-text loss.
