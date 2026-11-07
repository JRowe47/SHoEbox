# SHoEbox handoff

## Current phase: delayed payload credit

- Base: `d47ddc6` (acquisition report).
- Added optional rank-four identity payload adaptation and factored,
  owner/version-bound delayed sensitivities. Legacy defaults remain rank zero.
- Protocol: [docs/SHoEbox_Payload_Credit_Protocol.md](docs/SHoEbox_Payload_Credit_Protocol.md).
- Focused gradient, optimizer, lifecycle and resume checks: 5 passed.
- Full preserved suites and six-arm study are running; results not yet claimed.
- Study: `artifacts/payload_credit/{41,43}/{gate_only,payload,fresh}`.
- Command: `python scripts/run_payload_credit.py --out artifacts/payload_credit --workers 6`.
- Each arm: 600 updates, batch 8, context 64, stages 200/200/200,
  307,200 bytes, 18,432,000 expert executions, 549,075 parameters.
- Keep selected and final checkpoints, optimizer/stream/RNG state, raw histories,
  evaluations, whole-query audits and completion records.
- Next decision depends on paired exact/long-delay losses and disruption of
  older publication content. Gradient connectivity alone does not establish
  useful retention. Reader slots and Raven internal-state holds remain separate.
