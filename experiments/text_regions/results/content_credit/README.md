# Content-credit repair measurements

Base: `bd9b14db2b7fa1d410e7d8fb3189e7615f917f43`.
Parent: the selected update-32 state-credit checkpoint,
SHA-256 `9c56112ad8eca71706f1451dd58a163e0e598103f561729f8918a0443a77778a`.

- `long_probe.json`: the 148-byte observation-to-answer derivative at source
  position 32 and query position 180, including raw FP64 gradients and logits.
  All modes process the same 192-byte window. The old 16-byte cut returns zero;
  repaired execution chunks match the full reference exactly.
- `smoke.json`: one real paired-source update at the default 384-byte horizon,
  batch 1 and 16-byte execution segments. Includes before/after short probes,
  gradient/clipping telemetry, update time and peak process memory.
- `summary.json`: compact results and validation scope.
- `checks.txt`: all 35 relevant checks passed in 52.982 seconds.

The smoke update uses 384 natural bytes and 384 task bytes at identical weights
before summing gradients, applying global norm cap 1.0 and stepping AdamW.
Parent weights and moments transfer; runtime state and stream cursors start fresh.
Its checkpoint is not validation selected and does not establish recall quality.
No earlier experimental approximation is used in the repaired trainer.

Reproduction commands and training-boundary semantics are in
[`docs/content_credit.md`](../../../../docs/content_credit.md). The downloadable
checkpoint archive contains the parent, smoke checkpoints and corpus. Extract it
at the repository root before running those commands; choose a fresh `--out`
directory when repeating the smoke update.
