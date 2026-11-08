# Bidirectional regional training

The six-region hierarchy supports joint AdamW and a rotating regional schedule.
Architecture, source recipes, sparse expert selection and runtime clocks do not
change. A cycle changes which **parameters** can learn, never which regions run.

## Schedule

After 64 joint warm-up updates, the default cycle repeats:

| Phase | Updates | Trainable parameters |
|---|---:|---|
| Upper development | 32 | L2/S2/L3, future heads, Y and decoder except local projection |
| Lower alignment | 32 | Byte embedding and A1/B1 |
| Joint learning | 64 | Everything; embedding, A1/B1 and local projection use half learning rate |

During upper development, every fourth chunk scales the decoder's direct local
slot to 0.5, consistently throughout that chunk. The scale affects the local
inputs to decoder interactions and the final local residual. It does not alter
the observations or publications feeding any producer region. Other chunks and
all validation/sampling use the full route. Set `--attenuation-every 0` to disable.

Lower alignment starts only after **both** future-span training losses have
improved by 2% against their initial values, using a 0.9-decay EMA. Readiness
latches once established. Until then the lower block becomes joint learning.
This is a small per-objective readiness gate, not a global byte-loss threshold
or a claim that an upper region has mastered abstraction. The rotation lengths
remain fixed. GradNorm and adaptive phase lengths are not part of this experiment.

Frozen parameters have `requires_grad=False` and `grad=None`. AdamW skips their
moments, step counters and weight decay. Their recurrent state still evolves,
and autograd differentiates their computations with respect to trainable inputs.
No region-wide `no_grad`, state freezing or extra replay is introduced. Adam
moments persist across thawing; TBPTT remains at the optimizer boundary.

## Run and resume

```bash
python -m shoebox.train_hierarchy --data data/hierarchy_disjoint \
  --out artifacts/cycle --training-mode cycle --updates 256
```

Fork the existing jointly trained checkpoint, retaining its weights, optimizer
moments, runtime state, data cursor and observed-pair counts:

```bash
python -m shoebox.train_hierarchy --data data/hierarchy_disjoint \
  --initialize-from artifacts/hierarchy_gate256/last.pt \
  --out artifacts/cycle_cont256 --training-mode cycle --cycle-warmup 0 \
  --updates 256 --validate-every 32

python -m shoebox.train_hierarchy --data data/hierarchy_disjoint \
  --initialize-from artifacts/hierarchy_gate256/last.pt \
  --out artifacts/joint_cont256 --training-mode joint --cycle-warmup 0 \
  --updates 256 --validate-every 32
```

`--updates` is the additional budget for a fork. Results distinguish additional
and lifetime exposures. The source checkpoint checksum joins the exact protocol.
The previous decay/no-decay optimizer groups are migrated by parameter identity;
the next joint update is bit-identical to the previous optimizer.

Use `--max-updates N` to pause. Resume with `--resume OUT/last.pt`, the same
training options and total additional budget, omitting `--initialize-from`.
The controller, schedule position, optimizer, runtime state, RNG and corpus cursor
are restored. A changed code/data/schedule protocol is rejected for exact resume;
`--initialize-from` explicitly starts a new experiment instead.

## Measurements

Training logs include phase, decoder scale, trainable parameter count, pre-clip
regional gradient norms, and each scale's prediction loss. Evaluation reports
document-weighted next-byte and 4/16-byte future losses, including valid tails.

Two distinct held-out interventions are reported:

- `upper`: permute only the middle/high slots entering the decoder; producer
  trajectories and top-down feedback remain intact.
- `upper_all`: also permute every top-down context into A1/B1/L2/S2 and rerun
  recurrent trajectories from fresh document states. Internal upper-to-upper
  bottom-up connections remain part of the upper producer group.

The latter tests the combined outward contribution of upper messages. Neither
intervention is a capacity-matched architecture ablation. Permutations preserve
timestamps and span identities while mismatching the context's document row.

```bash
python -m scripts.compare_hierarchy_cycle \
  artifacts/joint_cont256/results.json artifacts/cycle_cont256/results.json \
  --out docs/regional_training_results.json
python scripts/check_all.py
python -m scripts.audit_hierarchy_checkpoint artifacts/cycle_cont256 \
  --data data/hierarchy_disjoint
```

The matched comparison changes the training regimen as a package: phase rotation,
joint-phase fast learning rates and occasional local-slot attenuation. It does
not individually attribute changes to those three components.

An optional `--initial-audit PATH` adds an initial-checkpoint audit to the report.
It is produced by loading that checkpoint with `shoebox.checkpoint.load` and
calling `evaluate(model, validation, intervention=...)` for `None`, `feedback`,
and `upper_all`; the ordinary call also uses `span_metrics=True`. These entries
are named `ordinary`, `feedback_only`, and `all_upper` in the audit JSON.

## Literature informing the schedule

- Felbo et al. (2017), chain-thaw: https://aclanthology.org/D17-1169/
- Howard and Ruder (2018), discriminative fine-tuning and gradual unfreezing:
  https://aclanthology.org/P18-1031/
- He et al. (2019), lagging inference networks: https://arxiv.org/abs/1901.05534
- Bowman et al. (2016), decoder word dropout: https://aclanthology.org/K16-1002/

The regional bidirectional cycle and its exact counts are SHoEbox adaptations.
