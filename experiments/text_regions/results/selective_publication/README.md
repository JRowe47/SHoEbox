# Selective publication: producer acquisition and retention

Three paired seeds compare a learned hard publication decision with unconditional publication in the same extracted SHoE producer. Each arm has 71,867 parameters, identical paired initialization, the same three losses, 240 updates and 7,680 fresh episodes. The task uses the common 256-byte head. A separate validation-only threshold calibration preserves the original comparison.

- [Result report](../../../../docs/SHoEbox_Selective_Publication_Report.md)
- [Locked training and evaluation protocol](../../../../docs/SHoEbox_Selective_Publication_Protocol.md)
- [Validation calibration protocol](../../../../docs/SHoEbox_Publication_Calibration_Protocol.md)
- `summary.json`: task thresholds, decision thresholds and mean scores.
- `training_audit.json` and `evaluation_audit.json`: initialization, exposure, selection, export and prediction checks.
- `compatibility.json`: exact original/new-engine composition comparison.
- Each run directory: protocol, history, every test prediction and validation calibration where applicable.

The separately downloadable artifacts include `parent.pt`, `initial.pt`, resumable `last.pt`, validation-selected `best.pt`, and `inference.pt` for all six runs. Learned runs also have `calibrated_inference.pt`; its only weight change from the selected model is the validation-selected gate-bias offset. Every export explicitly records its publication policy. The original seven-node integrated model is unchanged.

Extract the source and artifact archives into the same directory, then run:

```bash
python -m pip install -r requirements.txt
python scripts/audit_selective_publication.py --evaluations
python scripts/check_all.py
```

The audit uses three independent CPU workers by default. `--workers 1` runs it sequentially. The resumable experiment driver is `scripts/run_selective_publication.py`. To train a fresh individual run in another directory:

```bash
python -m experiments.text_regions.publication_train \
  --parent artifacts/selective_publication/parent.pt \
  --out artifacts/fresh_publication --policy learned \
  --seed 20261110 --steps 240 --validate-every 60
python scripts/evaluate_selective_publication.py artifacts/fresh_publication
```

Append `--resume` to the exact training command to recover a partial run. Use `--policy always` for the matched unconditional-publication control. Test evaluations never select weights or calibration offsets.

To repeat the legacy-engine compatibility comparison from a Git checkout:

```bash
git worktree add --detach ../SHoEbox_previous 334661f0099ced5835f0689afdc9011488028bdc
python scripts/check_publication_compatibility.py --previous ../SHoEbox_previous
```

The artifact package preserves the original compatibility snapshots and the separate 40-update development pilot. The pilot does not contribute to the confirmatory means or compute budgets.
