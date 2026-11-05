# Regional retained facts and corrections

This directory records two matched 100-update continuations of the integrated
update-200 parent. The control uses local-loss weight zero; the intervention
uses weight one. All other training settings, initial weights, optimizer
moments, data streams and execution budgets match.

See [the completed report](../../../../docs/SHoEbox_Regional_Facts_Report.md)
and [the locked protocol](../../../../docs/SHoEbox_Regional_Facts_Protocol.md).

## Selection and evidence

`selection.json` records validation-only checkpoint selection, the paired
training audit and exact inference-export checks. `evaluation_audit.json`
recomputes saved prediction metrics and verifies generated cohorts, targets
and matched evaluation budgets. `verification.json` checks full completion,
source/checkpoint manifests, diagnostic arithmetic and recovery provenance.

Both endpoint evaluations use all 592 held-out composition cases and 512
episodes per memory condition or view query. `control/` and `regional/`
contain complete histories, fixed protocols, summaries, raw evaluation
predictions and frozen local-head diagnostics. These JSON files are copied
unchanged from the completed artifact directory.

The selected checkpoint is the control at phase update 100. The regional
arm's own validation selection retains its initial checkpoint. Its update-100
endpoint is reported separately as the matched intervention result; it is
not promoted using held-out scores.

## Reproduction

Extract the source and artifact archives into the same parent directory;
both contain `SHoEbox/`. Tensor checkpoints are in the artifact archive and
are excluded from git. The archive includes the integrated parent,
`initial.pt`, `last.pt`, `best.pt`, `best_candidate.pt`, every validation
endpoint, the update-90 recovery snapshots, and inference-only exports.

From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/audit_regional_facts.py
python scripts/audit_regional_facts.py --evaluations
python scripts/verify_regional_completion.py
python scripts/report_regional_facts.py
```

For a fresh continuation, use new output directories:

```bash
python -m experiments.text_regions.regional_train \
  --parent artifacts/integrated/selected.pt --regional-weight 0 \
  --out artifacts/reproduced_regional_facts/control
python -m experiments.text_regions.regional_train \
  --parent artifacts/integrated/selected.pt --regional-weight 1 \
  --out artifacts/reproduced_regional_facts/regional
```

Append `--resume` with identical arguments after an interruption. To recover
specifically from the preserved update-90 state, copy its checkpoint to
`last.pt` in a separate copy of the corresponding run directory before
resuming. Do not overwrite the completed run.

For either arm, score its endpoint with:

```bash
python scripts/evaluate_regional_facts.py \
  artifacts/regional_facts/control/last.pt --count 512 --batch 256 \
  --out reproduced_control_evaluation.json
python scripts/probe_regional_facts.py \
  artifacts/regional_facts/control/last.pt \
  --out reproduced_control_local_diagnostic.json
```

Substitute `regional` for the intervention. Evaluation saves each completed
stage atomically; append `--resume` to reuse it with the same checkpoint,
count, batch size and source. The frozen local heads are diagnostic only.
All inference answers still come from the shared 256-byte output head.
