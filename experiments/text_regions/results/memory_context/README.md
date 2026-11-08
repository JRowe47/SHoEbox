# First-class memory acceptance

The compact matched bank and all three evaluations are complete. See
`summary.json` for metrics, source/checkpoint hashes, resume results and work
counts. `protocol.json` records the exact training configuration and guarded
sources. The four-update endpoint is experimental; no model is promoted.

Run `python scripts/summarize_memory_context.py --tests <tests.txt>` after
restoring the checkpoint archive to repeat the audit. The implementation and
recovery commands are in `docs/memory_context.md`.
