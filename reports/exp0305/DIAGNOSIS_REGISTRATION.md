# EXP03/EXP05: bounded-state diagnosis registration

Starting main: 9e34d8075abbc495f966ab37d18ecfa532bb854d.
Registered before diagnostic execution or new fitting, 16 November 2026.
Public E8, v19/E7, all prior constructors, checkpoints and gates remain fixed.

## Questions and fixed measurements

Use the existing qualified seed-7 E8 binding snapshot, frozen. No fitted updates,
no prior sweep. On 64 fresh development worlds (four record/query order siblings
per world), compare two bindings with padding, repeated neutral bytes, diverse
neutral bytes, two competing bindings, and overwrite of one original key.
Noise lengths are 0,16,64,256; training is absent. Extra writes are placed before
queries and have identical byte lengths to matched padding controls. Values
remain two distinct bytes, so the strongest first/last/constant shortcut is
reported. Repeated writes mean the latest value wins, and the other key must
remain unchanged. Counterfactual query order and record order share a group.

Record full-256-byte autoregressive whole-episode/position accuracy. Separately
trace teacher-fed prefixes: capture each required eight-byte constituent at
write completion, measure exact retained signature presence at query time,
all-window learned key/value decodability, inclusion in the top-two shortlist,
query identity, selected value and native output. Traces and oracle presence
never enter inference. Diagnostic interventions freeze only L1 retained writes
through the known distractor span, preserving its constituent frontier; this
is an externally timed diagnostic, not a deployable mechanism. Compare normal
and frozen-cache behavior and disclose both. No loss of an L1 signature is
claimed to prove absence from all other state.

## Freshness and exclusions

Salt: SHoEbox-EXP0305-bounded-state-v1; deterministic RNG 20261123. All concrete
initial-state families share one fold, including distractors, overwrites, phases,
record/query permutations and length variants. Enumerate and exclude every
opened binding audit group from EXP00/01, B, C, D2, D3, E1 and E2-family; also
exclude the union of their declared test folds conservatively where practical.
E2-family audits are opened. No generic corpus is scored or fitted: all E1/E7
opened document and generic-task groups are barred from any later qualification.
The frozen checkpoint itself is historical trained E8, not a fresh fit or audit.
Diagnostic development worlds use keys in its existing 48-key vocabulary;
fresh fitting may use the complete 64-key fixture vocabulary with its own lock.
No new test/audit is scored until a candidate and all prerequisites are locked.

## Decision and costs

If a required record survives and decodes but is excluded before query matching,
register query-first selection. If normal writes evict it and the diagnostic
freeze recovers it, register selective admission/replacement in the existing
16 slots. Repeated-key failures with surviving records require occurrence-aware
selection, separate from value reconstruction. Do not combine mechanisms or
add slots before distinguishing these causes. A candidate gets a separate
write-once registration with matched budgets and all original gates.

Log every inference/padding byte, clone, update (zero here), process CPU,
compilation and failed attempt. Keep inputs/checkpoints outside git. Preserve
source hashes and checkpoint/cursor receipts. No opened audit score selects a
numerical change.

Primary grounding: Arora et al. (ICLR 2024), MQAR and input-dependent selection,
https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html;
Hsieh et al. (COLM 2024), RULER's separate context length and retrieval complexity,
https://arxiv.org/abs/2404.06654; Schlag et al. (ICML 2021), correcting a key's
mapping rather than accumulating conflicting values,
https://proceedings.mlr.press/v139/schlag21a.html.
These motivate diagnostics; no paper's result is attributed to SHoEbox.
