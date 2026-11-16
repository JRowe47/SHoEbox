# EXP01-E1: joint native learning and learned residual applicability

Registered 16 November 2026 before EXP01-E fitting. Starting main:
`428fb0b0272d81b832a8a10263b44218b23cff3d`. Production remains minimal v19.

## Hypotheses and mechanisms

H1: Stable constituent features let native v19 continue learning while D3 learns
paired retrieval. Compare frozen D3 and joint D3 under identical answer masks.
H2: A learned sigmoid of the current existing two-byte stable constituent can
scale the D3 residual using next-byte cross-entropy alone. Both native and
residual weights receive error from the same combined prediction. The gate's
65 weights are deducted from the nominal budget. Gate initialization is zero
(sigmoid 0.5), normalized SGD rate 0.16, with no optimizer state or sweep.
H3: Every-byte training teaches when the residual is useful; it must preserve
binding, generic language, selective copy and bounded-state reconstruction.
Local presence/key/value targets remain explicit fixture supervision, supplied
only for the binding curriculum. Generic streams use no syntax labels. A
no-local-label control measures that dependence. Local heads use D3's rate and
weight; their source routing remains discrete, with no claimed gradient.

Primary grounding: Miller et al., [EMNLP 2016](https://aclanthology.org/D16-1147/)
for paired addressing/value mixing; See et al., [ACL 2017 §2.2](https://aclanthology.org/P17-1099/)
for a learned generator/retrieval switch. This implementation gates residual
logits rather than pointer probabilities. Arora et al., [ICML 2024](https://proceedings.mlr.press/v235/arora24a.html)
motivates simultaneous recall, state and throughput measurement. Jaderberg et
al., [ICML 2017](https://proceedings.mlr.press/v70/jaderberg17a.html) provides
context for local online credit; no synthetic-gradient model is added.

## Binding design

Fresh concrete-world grouped splits, salt `SHoEbox-EXP01E-joint-gate-v1`, data
RNG 20261120; 2,048 train / 512 development / 1,024 audit presentations.
All record/query-order siblings share a split. Exclude the union of opened D2
and D3 audit worlds from all E splits. Never exclude by model outcome.
Every arm consumes identical episode order (RNG 702), short spacers (RNG 1702),
and balanced phases. No long delay is trained. Native warmup uses RNG 701,
250,000 charged bytes, answer targets; continuation is 2,000,000 charged bytes.

Arms: native_masked; frozen_d3; joint_d3; native_all; gated_all; gated_no_local.
Masked arms score answers during training. Every-byte arms score every next
byte. The two native controls separate objective coverage from architecture.
All capacities are nominal 500K including the readout. D3 has 9,900 fitted
weights; gated has 9,965. Count actual indivisible bank sizes. Each arm gets its
own charged warmup, even if numerically identical. Frozen D3 freezes only its
core. Other core weights remain trainable. All head fitting stays online.

Pilot: seed 7, gated_all, 250K warmup plus 2M continuation, development only.
If it fails, retain it and register a focused amendment before a new fit.
Formal: seeds 1/7/19, all six arms, same budgets and order. Final checkpoint;
no best-of-audit selection. Candidate selection and source hashes must be locked
before opening the audit. Evaluate native full-256-byte autoregressive answers,
generated first answers feeding second queries, teacher-forced and restricted
scores separately; cloned-prefix counterfactual query orders.

## Unchanged D3 gates

- Primary exact mean >=95%, every seed >=90%; same for each answer position.
- Both query orders mean >=90%, every seed >=85%.
- All 64 offsets: mean >=95%, worst offset averaged across seeds >=90%.
- Independent +0/+64/+128/+256 padding delays: each mean >=90%, every seed >=85%.
- No-query drop >=20 percentage points; reversed selection strictly hurts.
- Preserve no-head, no-state, uniform, reversed and unpaired interventions;
  add gate-off/on interventions without changing the earlier ones.
- Isolated pinned-CPU full-256-byte inference >=80% native. Three interleaved
  65,536-byte repetitions, compilation/cloning excluded. No other jobs running.

Use D3's sample sizes: 1,024 primary presentations; 256 for interventions and
counterfactuals; identical first 64 presentations across each offset; next 16
presentations across all offsets and each independent delay. Shared worlds
across seeds are not independent samples.

## Production regressions and capacity

After binding development, run matched fresh native, D3-architecture/joint, and
gated-architecture/joint fits at 500K, seeds 1/7/19. FineWeb: 1M next-byte targets,
held-out development and audit 65,536 targets each, document-grouped hash splits,
UTF-8, no cross-document state. Pin IDs, complete-text hashes, byte hashes and
exact cursors before fitting. No fixture labels on language. Selective copy:
historical length 2/noise .5, 250K charged bytes, lr scale 2, native answer loss;
bounded reconstruction: length 4/vocab16, 250K bytes, lr scale .5, final-L3 MLP
budget included, generic native answer mask plus separate reconstruction credit.
Use fresh grouped task splits; train/dev/audit counts 256/128/128.

Retain historical margins: candidate minus native mean language NLL <=0.01
nats/byte; copy/reconstruction token accuracy loss <=1 percentage point. Also
report full-answer exactness and native output where applicable. Test retention
by continuing the selected binding checkpoint on the same 1M FineWeb targets
with no local labels, then re-running every binding gate. This is separately
charged; do not conflate independent task fits with retained binding ability.

Then repeat native/frozen D3/selected joint binding at nominal 1M and 5M,
seeds 1/7/19, 250K warmup +2M continuation; preserve all gates. Report loss by
bytes and CPU time, training and inference throughput, actual parameters,
auxiliary updates, persistent state and peak memory. A failed prerequisite is
not a production pass; later checks remain useful diagnostic measurements.

## Integrity, revisions and delivery

Registration files are write-once and include all executable dependency hashes.
Do not inspect audit scores until locked; any audit-informed revision gets a
new untouched grouped audit excluding all already opened worlds. Every pilot,
failed fit, padding byte, local update and CPU second remains in the ledger.
Pure correctness-test updates are separately identified, not research training.
No parser, source oracle, exact store, new slots, hidden replay or tape.

Require relevant Python/native tests, gate finite differences, native/D3 numerical
parity, label/target causality, frozen read-only behavior, exact checkpoint and
outer-cursor continuation, ASan/UBSan. Historical source and v19 serialization
remain unchanged. Publish compact code/evidence, downloadable Markdown/PNG and
an archive with selected checkpoints/cursors outside git. Promote only when all
production gates pass; otherwise preserve v19 and register the measured next
revision. Direct main publication is explicitly authorized; never force-push.
