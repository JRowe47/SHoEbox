# SHoEbox handoff

## Completed: delayed payload credit

- Implementation: `a7a14ab`; base `d47ddc6`.
- Added optional identity payload adaptation and factored owner/version-bound
  delayed sensitivities. Four coefficients per owner; legacy rank defaults to zero.
- All 502 checks, the live-output example, and GitHub CI passed.
- Six runs completed: seeds 41/43, gate-only/payload/fresh, 600 updates each.
  Each arm used 307,200 bytes, 18,432,000 expert executions and 549,075 parameters.
- Selected test NLL means: gate-only 1.2079, payload 1.2177, fresh 1.2195.
- Selected long-delay improvement reverses against gate-only at the final budget.
  Older-content effects are small and inconsistent. Keep the mechanism optional.
- All 32 whole-query audits finished, including age cutoffs 16 and 64.
- [Report](docs/SHoEbox_Payload_Credit_Report.md),
  [protocol](docs/SHoEbox_Payload_Credit_Protocol.md),
  [metrics](experiments/text_regions/results/task_payload_credit/summary.json).
- Resumable selected/final/acquisition/bridge checkpoints, raw audits, histories,
  logs and hashes are in `SHoEbox_Payload_Credit_Artifacts.zip`.

## Next target

Test Raven-style internal owner-state holds against publication-only holds at
matched candidate compute. Keep read, internal-update and publication decisions
separate. Require older-content/task effects and timely corrections before
claiming a retention improvement or scaling up. Multiple read slots and the
persistent short-delay mixture remain independent comparisons.

Reproduce: `python scripts/run_payload_credit.py --out artifacts/payload_credit --workers 3`.
