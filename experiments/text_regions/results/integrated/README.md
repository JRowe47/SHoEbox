# Integrated capability training: recorded results

The baseline completes 200 updates from the selected second-lookup checkpoint.
The per-task gradient-cap follow-up completes 100 updates from exactly the same
initial weights, optimizer moments, RNGs and task streams. Compare it with
`baseline_step100.pt`; the baseline endpoint at 200 uses additional compute.
The report is [SHoEbox_Integrated_Report.md](../../../../docs/SHoEbox_Integrated_Report.md).

All JSON here is copied from the completed artifact directory. Per-case
predictions, generated targets/identities, complete histories, protocols,
training audits and the cross-run validation selection are retained. Large
tensor checkpoints are in the downloadable artifact archive. Extract both the
source ZIP and artifact ZIP into the same directory; both use `SHoEbox/`.

## Selection and checkpoints

`selection.json` records the declared validation-only selection rule and all
candidates. The selected integrated candidate is `joint/best.pt`, update 200.
The diagnostic winner is the same endpoint. Neither run meets full joint
capability qualification. Balanced update 100 is retained as the controlled
follow-up and has consecutive composition-gate passes at validation updates
50 and 100.

`joint/last.pt` and `balanced/last.pt` are full resumable checkpoints. Their
`inference.pt` exports contain the common inference graph without either
training-only head. The balanced best and last weights are identical.
`parent.pt`, `baseline_step100.pt` and `diagnostic_step25.pt` are also included.

## Reproduce

From the repository root, after installing `requirements.txt`:

```bash
python -m experiments.text_regions.integrated_train \
  --parent artifacts/integrated/parent.pt --out artifacts/reproduced_joint \
  --steps 200 --stream-batch 16 --composition-batch 64 --segment 16 \
  --lr 0.0005 --validate-every 50 --validation-count 64 --threads 1

python scripts/train_integrated_balanced.py \
  --parent artifacts/integrated/parent.pt \
  --baseline-protocol artifacts/integrated/joint/protocol.json \
  --out artifacts/reproduced_balanced --steps 100

python -m experiments.text_regions.integrated_eval \
  artifacts/integrated/baseline_step100.pt --count 192 \
  --out reproduced_baseline100.json
python -m experiments.text_regions.integrated_eval \
  artifacts/integrated/joint/best.pt --count 192 \
  --out reproduced_baseline200.json
python -m experiments.text_regions.integrated_eval \
  artifacts/integrated/balanced/last.pt --count 192 \
  --out reproduced_balanced100.json

python scripts/audit_integrated.py \
  --parent artifacts/integrated/parent.pt --run artifacts/integrated/joint \
  --evaluation artifacts/integrated/joint/evaluation.json \
  --out reproduced_joint_audit.json
python scripts/audit_integrated.py \
  --parent artifacts/integrated/parent.pt --run artifacts/integrated/balanced \
  --evaluation artifacts/integrated/balanced/evaluation.json \
  --out reproduced_balanced_audit.json
python scripts/audit_integrated_comparison.py --root artifacts/integrated
python scripts/audit_integrated_evaluations.py --root artifacts/integrated
python scripts/measure_joint_gradients.py \
  artifacts/integrated/diagnostic_step25.pt --out reproduced_gradients.json
python scripts/check_all.py
```

Use `--resume` with the same training command after an interruption. To recreate
the baseline update-100 snapshot, run its fixed 200-update protocol with
`--stop-after 100`, copy `last.pt`, then resume the remaining updates. Checkpoint
steps in this round count joint updates after the parent; they are not the
parent's regional optimizer step count.

The two trainable source-supervision heads are removed only for inference.
Evaluation of their local lookup and exact source-retention metrics requires
the full training checkpoint. All task answers use the common 256-byte head.
