# SHoEbox handoff

## Completed: selective producer-state commits

- Implementation: `483eb11`; base `150c8a0`.
- Added optional `--state-commit publication`. Owner contextualizer and recurrent
  block contents hold on unpublished events. Ingress, composer, clocks and all
  candidate computation continue. No additional parameters or persistent tensors.
- Live straight-through gradients cover the state path. Detached delayed credit
  still covers the conditional publication path, not historical state derivatives.
- All 512 checks and the live-output example passed. Ten new checks cover exact
  holds, row/reset isolation, gradients, exact resume, and state-content audits.
- Eight 600-update arms completed: seeds 41/43 crossed with always/held producer
  state and stored/fresh reads. Four new held-state runs use four prior controls
  restored with verified hashes. Every arm has 549,075 parameters, 307,200 training
  bytes and 18,432,000 expert executions. Held arms commit 23.95–24.83% of candidates.
- Stored-read selected test NLL averages 1.2008 held versus 1.2079 always; final
  test averages 1.2035 versus 1.2063. Selected long-delay NLL improves from 1.3481
  to 1.3046, then reverses at the final checkpoint: 1.2649 held versus 1.2554 always.
- Held models barely depend on protected producer contents during query windows.
  Old-producer and old-publication interventions show no repeatable retention gain.
  These interventions preserve pre-query ingress/composer state and publications
  when changing producer contents. State holding remains optional/default-off.
- All 32 checkpoint/split audits reproduce intact answers: zero argmax changes,
  maximum logit drift 3.8147e-6. Commit age measures time since the last update,
  not the age of every fact in a recurrent state.
- [Report](docs/SHoEbox_State_Commit_Report.md),
  [PDF](docs/SHoEbox_State_Commit_Report.pdf),
  [protocol](docs/SHoEbox_State_Commit_Protocol.md),
  [metrics](experiments/text_regions/results/task_state_commit/summary.json).
- Source, selected/final/acquisition/bridge checkpoints, optimizer/runtime states,
  histories, audits, validation logs, baseline provenance and file hashes are in
  `SHoEbox_State_Commit_Artifacts.zip`.

## Next target: distinct contextual read slots

Test two contextual read slots against a parameter-matched single-read control.
Keep retrieved pieces separate until the expert composer combines them. Require
task effects from retained content, including older-content interventions and
timely corrections, before claiming a retention gain or scaling up. Keep the
short-delay training mixture as a separate comparison. Fresh-read controls replace
read values only; publication-aware admission remains active.

## Reproduction

```bash
python scripts/run_state_commit.py --out artifacts/state_commit --workers 4
python scripts/summarize_state_commit.py --root artifacts/state_commit \
  --out experiments/text_regions/results/task_state_commit/summary.json
python scripts/check_all.py
```

The runner resumes existing checkpoints. The completed study trained the held
arms with `--arms held held_fresh --skip-audits`, then ran `--audit-only` for each
pair of arms. Restored controls are mapped in `baseline_provenance.json`.

Prior completed pass: [delayed payload credit](docs/SHoEbox_Payload_Credit_Report.md)
(`a7a14ab` implementation, `150c8a0` results). Its six matched runs also showed
mixed retention results; payload credit remains optional.
