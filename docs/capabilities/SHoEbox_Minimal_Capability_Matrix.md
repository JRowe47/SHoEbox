# SHoEbox minimal-baseline capability matrix

16 November 2026 (America/Denver) · Runtime: public E8 over minimal corrected C; explicit v19 preserved. [Integration verification](../../reports/e8_public/SHoEbox_E8_Public_Integration_Report.md).

Current inventory: **62 core capabilities, four optional multimodal capabilities, and eight engineering enablers**. This matrix is the current implementation/evidence view. The older full-C assessments are preserved in each backlog row's `historical_assessment` and in repository history.

**Yes** means a relevant executable mechanism remains; **Partial** means supporting pieces exist without a complete workflow; **No** means the capability is outside the minimal implementation. A research decoder is identified explicitly. **Untested** means no adequate minimal-specific capability result; it does not turn an old full-C failure into a minimal result. **Verified engineering** covers software contracts, not semantic competence.

No broad generalist capability is generally demonstrated. Minimal evidence supports next-byte learning, short selective copying, bounded-state reconstruction, and causal use of memory on the copy task. The E8 public readout qualifies narrow fixture-supervised two-binding recall across boundary phases and padding delays, with exact research-to-public numerical parity.

EXP01-E8 passes complete registered research qualification: 98.2747% held-out
binding retained through 1M language targets, 97.2656% offset mean, 93.75%
worst offset, 96.875% both query orders and 100% padding delays. Fresh generic
language increases 0.006233 nats/byte; copy/reconstruction remain within their
registered margins. Three-seed 1M/5M and >=80% throughput gates pass.
See [E8 results](../../reports/exp01e8/SHoEbox_EXP01E8_Report.md).
[Versioned public integration](../../reports/e8_public/SHoEbox_E8_Public_Integration_Report.md)
passes numerical/checkpoint parity and all speed gates; E8 is the default. E2-family
binding and E7 generic audits are now opened; never tune against them.

Latest status: **A9-F1 is complete; A9 fails development.** The exact 128/336
binary allocation repair preserves all 896 primary worlds and original streams.
The matched 20.25M-byte E8/A8/A9 campaign gives A9 88.281250% competing,
88.281250% corrected overwrite, 94.531250% clean, 86.328125% diverse256 and
68.202933% isolated native speed. E8 remains public; A1–A9 audits stay closed.
[Completed report](../../reports/exp0305a9f1/SHoEbox_EXP0305_A9_F1_Report.md)
and [diagnosis](../../reports/exp0305a9f1/DIAGNOSIS.md) separate frozen subset
localization from full autoregression. The sole next step is the unfitted,
unimplemented [A10 exact execution experiment](../../reports/exp0305a9f1/A10_REGISTRATION.md),
with zero research fitting bytes and no accuracy-promotion claim. All gates remain.

Historical blocked status: **A9 was implemented but blocked before research fitting.**
The original fixed split leaves 96 binary development worlds versus 128 required.
No A9 capability/throughput measurement or audit exists. E8 and all historical
qualifications remain unchanged. [Blocked report](../../reports/exp0305a9/SHoEbox_EXP0305_A9_Report.md)
and [prospective binary-only amendment](../../reports/exp0305a9/A9_F1_REGISTRATION.md)
record the sole next step; it is not implemented or fitted. The historical
A8 result below remains intact.

Latest focused result: **A8 fails development; E8 remains public and A1–A8 audits
stay closed.** [A8 report](../../reports/exp0305a8/SHoEbox_EXP0305_A8_Report.md)
and [diagnosis](../../reports/exp0305a8/DIAGNOSIS.md) preserve the exact 20.25M-byte
campaign. [A9 selective record-key credit](../../reports/exp0305a8/A9_REGISTRATION.md)
is registered only, not implemented or fitted. This supersedes historical
next-step statements below without replacing their measurements.

## Retained architecture

- Byte input/output; width 64; exact corrected-C sparse experts at 1 and 4 bytes.
- Ordered trainable summaries at 16 and 64 bytes, diagonal recurrence, memory reads and learned constant output modulation.
- MinKron context, addressed memory, retained-unit retrieval, reconstruction and bounded child credit.
- Separate fast/slow/query learning rates, coverage-only warmup, one-pass prediction and v19 exact checkpoint continuation.
- Upper expert banks/routing/forecast heads, state-dependent upper modulation, D predictors, loops, route-rate cycles and experimental credit multipliers are outside minimal.

## Preserved v19 qualification

Three paired seeds (1, 7, 19), nominal 500K models. FineWeb used 1M next-byte targets; copy and reconstruction each used 250K charged input bytes per fit.

| Measure | Full C | Minimal |
| --- | ---: | ---: |
| FineWeb held-out NLL (nats/byte) | 2.358039 | 2.252588 |
| Selective-copy token / exact whole-answer accuracy | 73.70% / 47.40% | 73.83% / 47.66% |
| Final-state reconstruction token / exact whole-answer accuracy | 70.77% / 22.14% | 80.14% / 40.36% |

Memory-read removal reduces minimal copy from 73.83% to 11.85%; zeroing final L3 state reduces reconstruction from 80.14% to 7.42%. Reconstruction uses a separately trained decoder included in the parameter budget, not the native language head. This table reproduces the earlier minimal qualification. The new EXP01-D research results are identified separately below.

[Protocol, raw results and verification](../../reports/minimal_core/SHoEbox_Minimal_Core_Baseline.md) · [Development handoff](../../DEVELOPMENT_HANDOFF.md) · [Experiment plan](SHoEbox_Capability_Research.md).

## Language and communication

| ID | Capability | Minimal implementation | Minimal evidence | Scope and next work |
| --- | --- | --- | --- | --- |
| L01 | Learn next-byte distributions | Yes | Narrow | Minimal FineWeb test NLL 2.252588 at 500K nominal parameters after 1M targets; three seeds. Next: EXP00, EXP13, EXP15, EXP21, EXP22. |
| L02 | Fluent, grammatical text generation | Partial | Untested | Autoregressive generation exists; fluent, grammatical output has no current qualification. Next: EXP15. |
| L03 | Semantic understanding, paraphrase and entailment | Partial | Untested | Generic byte learning or recurrent-state foundations exist; the complete capability has no current qualification. Next: EXP16. |
| L04 | Follow instructions and satisfy multiple constraints | Partial | Untested | Generic byte learning or recurrent-state foundations exist; the complete capability has no current qualification. Next: EXP15. |
| L05 | Dialogue, coreference and conversational intent | Partial | Untested | Generic byte learning or recurrent-state foundations exist; the complete capability has no current qualification. Next: EXP15, EXP16. |
| L06 | Summarize and synthesize without losing essential facts | Partial | Untested | Bounded-state reconstruction is measured separately and does not establish semantic summarization. Next: EXP16. |
| L07 | Extract, classify and organize information | Partial | Untested | Generic byte learning or recurrent-state foundations exist; the complete capability has no current qualification. Next: EXP16. |
| L08 | Rewrite, edit and transform text while preserving meaning | Partial | Untested | Generic byte learning or recurrent-state foundations exist; the complete capability has no current qualification. Next: EXP15, EXP16. |
| L09 | Multilingual understanding and translation | Partial | Untested | All-byte representation is available; multilingual competence is untested. Next: EXP15. |
| L10 | Maintain coherence and constraints over long outputs | Partial | Untested | Generic byte learning or recurrent-state foundations exist; the complete capability has no current qualification. Next: EXP15, EXP22. |
| L11 | Produce reliable structured output | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP15, EXP18. |
| L12 | Answer questions grounded in supplied documents | Partial | Untested | Generic byte learning or recurrent-state foundations exist; the complete capability has no current qualification. Next: EXP16. |
| L13 | Creative writing and controllable style | Partial | Untested | Generic byte learning or recurrent-state foundations exist; the complete capability has no current qualification. Next: EXP15. |

## Knowledge, memory and state

| ID | Capability | Minimal implementation | Minimal evidence | Scope and next work |
| --- | --- | --- | --- | --- |
| M01 | Memorize stable key–value facts in weights | Yes | Untested | Weight learning is retained. Fixed-table memorization was demonstrated in the earlier full-C development screen; minimal retest pending. Next: EXP13. |
| M02 | Acquire and retrieve broad factual and commonsense knowledge | Partial | Untested | Generic byte learning or recurrent-state foundations exist; the complete capability has no current qualification. Next: EXP13, EXP16. |
| M03 | Select and copy relevant short-context content | Yes | Narrow | Minimal selective copy: 73.83% token accuracy, 47.66% exact whole answers. Next: EXP00, EXP03. |
| M04 | Compress observed content into a usable bounded state | Yes — research decoder | Narrow | Minimal final-L3 reconstruction: 80.14% token accuracy, 40.36% exact; separately trained MLP decoder included in the budget. Next: EXP00, EXP02. |
| M05 | Use remembered information in prediction | Yes | In principle | Disabling addressed and retained reads lowers minimal copy token accuracy from 73.83% to 11.85%; establishes causal use on this task. Next: EXP01, EXP03, EXP08. |
| M06 | Store and recover exact externally supplied bindings | Yes — explicit API | Untested | bind/lookup/payload entry points remain available; automatic association dispatch is excluded from minimal. Earlier exact-API evidence is historical; no new minimal-specific binding qualification is claimed. Next: EXP14. |
| M07 | Bind a newly observed value to the right query | Yes — public E8 paired readout | Public narrowly qualified | E8 three-seed audit: 98.2747% native whole-episode exactness, retained through 1M language targets; 96.875% both orders; query/state/pairing ablations pass. Explicit fixture-local labels, 7,670 budgeted weights, no extra sequence state. Generic and 1M/5M gates pass. Public integration and exact checkpoint/cursor parity pass. Next: EXP03/05 interference. |
| M08 | Retain information through longer delays and interference | Yes — public E8 readout | Narrow for padding; interference development gate fails | E8 audit reaches 100% through +64/+128/+256 unseen padding bytes across all offsets on the registered delay sample; after language retention too. EXP03/05 diagnoses eviction under diverse filler and shortlist competition. A1 (research only) improves matched 64/256-byte diverse retention from 13.54/9.77% to 85.55/76.56%, but fails gates. E8 stays unchanged. Fresh three-value A2 query-first selection raises competing exactness 37.89% →92.97%, but fails development; On fresh matched folds A3 raises corrected overwrite 76.82% →95.96%, but competing binding falls 95.31% →85.81%; the full development gate and speed (71.75% native) fail. A4 query-relative matching also fails development: 86.72% competing bindings, 94.79% corrected overwrite, 64.45% diverse256 and 68.35% native speed. A5 local discriminative key reconstruction also fails: 90.23% competing, 95.70% corrected overwrite, 63.80% diverse256 and 67.57% native speed. A6 fixed record-presence cost also fails: 87.89% competing, 94.14% corrected overwrite, 84.38% diverse256 and 67.58% native speed; long-diverse signature survival reaches 100% on the registered subset. A7 also fails: 66.80% competing, 75.26% overwrite, 84.77% diverse256 and 69.71% native speed. A8 balanced PAD/diverse curriculum fails: 80.86% competing, 78.52% overwrite, 83.59% diverse256 and 68.59% native speed. A9 selective record-key credit with F1 allocation fails: 88.28% competing, 88.28% overwrite, 86.33% diverse256 and 68.20% native speed. E8 remains public and A1–A9 audits stay closed. The next A10 execution-only registration has zero research fitting bytes. |
| M09 | Recall despite noisy or approximate cues | Yes — memory path | Untested | Prior noisy/fuzzy full-C probes failed; minimal noisy-cue behavior is unqualified. Next: EXP03. |
| M10 | Update working variables without overwriting unrelated state | Partial | Untested | Research tasks exist; full-C initialization/set/copy/swap failed EXP00. No qualified state-update workflow. A1 binary overwrite has a perfect latest-value shortcut; its failed 39.06% is diagnostic only. Corrected three-value A2 improves overwrite 26.17% →66.02%, still failing; both versions survive and distance-only selection can prefer the old value. On fresh matched folds A3 raises corrected overwrite 76.82% →95.96%, but competing binding falls 95.31% →85.81%; the full development gate and speed (71.75% native) fail. A4 query-relative matching also fails development: 86.72% competing bindings, 94.79% corrected overwrite, 64.45% diverse256 and 68.35% native speed. A5 local discriminative key reconstruction also fails: 90.23% competing, 95.70% corrected overwrite, 63.80% diverse256 and 67.57% native speed. A6 fixed record-presence cost also fails: 87.89% competing, 94.14% corrected overwrite, 84.38% diverse256 and 67.58% native speed; long-diverse signature survival reaches 100% on the registered subset. A7 also fails: 66.80% competing, 75.26% overwrite, 84.77% diverse256 and 69.71% native speed. A8 balanced PAD/diverse curriculum fails: 80.86% competing, 78.52% overwrite, 83.59% diverse256 and 68.59% native speed. A9 selective record-key credit with F1 allocation fails: 88.28% competing, 88.28% overwrite, 86.33% diverse256 and 68.20% native speed. E8 remains public and A1–A9 audits stay closed. The next A10 execution-only registration has zero research fitting bytes. Next: EXP04, EXP05. |
| M11 | Maintain useful episodic memory across sessions | Partial | Untested | Checkpoint persistence exists; autonomous episodic admission, retrieval and answer use across sessions are unqualified. Next: EXP14. |
| M12 | Track who said what, when, and with which source | Partial | Untested | Temporal/address features exist; semantic speaker, time and source attribution are unqualified. Next: EXP14, EXP16. |
| M13 | Select memory writes, updates and forgetting appropriately | Partial | Untested | A1 learned replacement protects existing L1 slots and improves diverse-filler retention, but fails the complete gate and remains research only. Semantic correction handling is unfinished. Next: EXP03, EXP05, EXP14. |
| M14 | Consolidate episodic experience into reusable knowledge | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP14. |
| M15 | Scale memory capacity independently of expert parameters | Partial | Untested | State and retained memory exist; independently editable working-slot comparisons remain proposed. Next: EXP02, EXP21. |
| M16 | Retain a simple binary cue through repeated distractors | Partial | Untested | The passing binary-cue regression was historical routing A. Reproduce it explicitly on minimal; do not inherit its result. Next: EXP03. |

## Reasoning and generalization

| ID | Capability | Minimal implementation | Minimal evidence | Scope and next work |
| --- | --- | --- | --- | --- |
| R01 | Follow relations and retrieve through multiple hops | Partial | Untested | One-hop graph exactness failed in full-C EXP00; minimal multi-hop retrieval remains unqualified. Next: EXP08, EXP10. |
| R02 | Compose operations and execute small programs | Partial | Untested | Prior full-C composition/state tasks failed. Establish atomic operations before chaining them on minimal. Next: EXP04, EXP09, EXP10. |
| R03 | Deduction, Boolean logic, negation and quantifiers | Partial | Untested | Generic byte learning or recurrent-state foundations exist; the complete capability has no current qualification. Next: EXP17. |
| R04 | Arithmetic, quantitative reasoning and units | Partial | Untested | Byte learning is available; unsuccessful historical modulo-add/xor probes do not qualify arithmetic. Next: EXP12, EXP17. |
| R05 | Causal and counterfactual reasoning | Partial | Untested | Generic byte learning or recurrent-state foundations exist; the complete capability has no current qualification. Next: EXP17. |
| R06 | Temporal and spatial reasoning | Partial | Untested | Generic byte learning or recurrent-state foundations exist; the complete capability has no current qualification. Next: EXP17. |
| R07 | Plan, decompose goals and satisfy interacting constraints | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP10, EXP17. |
| R08 | Form abstractions, analogies and invariant concepts | Partial | Untested | Ordered trainable summaries remain; semantic abstraction and invariance are not demonstrated. Next: EXP06, EXP07, EXP09, EXP20. |
| R09 | Systematically recombine familiar parts in new combinations | Partial | Untested | Ordered composition machinery exists; systematic transfer to unseen combinations is unqualified. Next: EXP09, EXP20. |
| R10 | Generalize to longer sequences and deeper computations | Partial | Untested | Historical delay/depth probes failed; minimal sequence-length and computation-depth qualification is pending. Next: EXP05, EXP07, EXP08, EXP09, EXP21. |
| R11 | Learn a new task or rule from examples in the prompt | Partial | Untested | Generic byte learning or recurrent-state foundations exist; the complete capability has no current qualification. Next: EXP09, EXP20. |
| R12 | Generate and compare hypotheses using evidence | Partial | Untested | Generic byte learning or recurrent-state foundations exist; the complete capability has no current qualification. Next: EXP17. |
| R13 | Understand, generate and repair code | Partial | Untested | Code can be represented/generated as bytes; execution correctness and repair are unqualified. Next: EXP12, EXP18. |

## Learning and computation control

| ID | Capability | Minimal implementation | Minimal evidence | Scope and next work |
| --- | --- | --- | --- | --- |
| C01 | Learn useful conditional specialization in upper experts | No — excluded | Untested | Minimal has no 16/64-byte expert banks. Study summary contribution first; any restored upper experts are a separately named candidate. Next: EXP06, EXP07, EXP20, EXP22. |
| C02 | Teach internal state through intermediate supervision | Yes — public local readout | Readout qualified; upper-summary specialization unqualified | E8 jointly trains native core with a fixture-supervised extraction/applicability readout; binding, retention and generic regression gates pass. This does not establish generic byte-loss training of the local heads or learned upper-summary specialization. Public integration passes. Next: separately registered broader objectives. |
| C03 | Improve answers by performing extra latent refinement | No — full research only | Untested | Existing refinement loops are excluded from minimal; prior full-C tests did not establish useful inference-time refinement. Next: EXP10. |
| C04 | Stop refinement when its state has converged | No — full research only | Untested | Convergence stopping exists only with the full-research loops, not in minimal one-pass prediction. Next: EXP11. |
| C05 | Allocate more computation to genuinely difficult inputs | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP11. |
| C06 | Search alternative solutions and select with a verifier | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP12. |
| C07 | Detect and correct its own mistakes | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP12. |
| C08 | Express calibrated uncertainty and abstain appropriately | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP19. |
| C09 | Learn from corrections, feedback and preferences | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP14. |
| C10 | Adapt continually while retaining prior capabilities | Partial | Narrow supervised fixture | E8 retains 98.2747% held-out binding through 1M generic language targets across three seeds; independent generic regressions pass. Broad continual adaptation remains unqualified. Public integration passes. Next: EXP14. |
| C11 | Reuse experts and transfer learned skills between tasks | Partial | Untested | Shared lower expert banks exist; semantic skill transfer is untested. Next: EXP20. |
| C12 | Choose among recalling, reasoning, asking and using a tool | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP18. |

## Tools, grounding and dependable behavior

| ID | Capability | Minimal implementation | Minimal evidence | Scope and next work |
| --- | --- | --- | --- | --- |
| T01 | Retrieve external documents and search for missing knowledge | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP16, EXP18. |
| T02 | Call functions/tools with correct arguments | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP18. |
| T03 | Use code execution and tests in a feedback loop | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP18. |
| T04 | Cite sources and attribute claims to the right evidence | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP16. |
| T05 | Carry out multi-step tasks while tracking goals and progress | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP18. |
| T06 | Separate authoritative instructions from quoted or untrusted data | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP19. |
| T07 | Maintain factual consistency and recognize contradictions | Partial | Untested | Generic byte learning or recurrent-state foundations exist; the complete capability has no current qualification. Next: EXP16, EXP19. |
| T08 | Follow intended behavioral boundaries consistently and robustly | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP19. |

## Optional multimodal expansion

| ID | Capability | Minimal implementation | Minimal evidence | Scope and next work |
| --- | --- | --- | --- | --- |
| X01 | Understand images, OCR, charts and diagrams | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP23. |
| X02 | Understand and generate speech/audio | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP23. |
| X03 | Understand video and events over time | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP23. |
| X04 | Ground language in interactive environments | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP23. |

## Engineering enablers

| ID | Capability | Minimal implementation | Minimal evidence | Scope and next work |
| --- | --- | --- | --- | --- |
| E01 | Byte input/output and streaming execution | Yes | Verified engineering | Byte input/output and streaming lifecycle are retained and tested. Next: EXP15. |
| E02 | Sparse expert execution and exact routing | Yes | Verified engineering | Exact corrected-C search and one selected expert operate at the 1/4-byte scales only. Next: EXP20. |
| E03 | Fixed 1/4/16/64-byte hierarchy and ordered composition | Yes | Verified engineering | Four fixed clocks remain; 16/64-byte scales use ordered trainable summaries, diagonal recurrence and memory reads. Next: EXP06, EXP22. |
| E04 | Causal local learning and credit through live constituents | Yes | Verified engineering | Causal local learning, reconstruction and bounded child credit remain; summary derivatives and execution regressions were checked. Next: EXP06. |
| E05 | Exact checkpoint continuation and frozen evaluation | Yes | Verified engineering | Minimal v19 exact continuation, partial groups and source-preserving frozen inference are tested; historical checkpoints retain their policies. Next: EXP00, EXP14. |
| E06 | Bounded compute/state accounting and comparable evaluations | Yes | Verified engineering | MAD/resource accounting and paired minimal qualification are implemented in research tooling. Next: EXP00, EXP21. |
| E07 | Route-local learning-rate control | No — full research only | Untested | Route-local learning-rate cycles are excluded from minimal. Independent base fast/slow/query rates remain. Next: EXP21. |
| E08 | Learned segmentation instead of fixed byte groups | No | Untested | Grouping stays fixed at 1/4/16/64 bytes; learned boundaries are proposed. Next: EXP22. |

## Next development decision

E8 public integration is complete, with 29 exact imports/cursors, 14 historical
constructor trajectories and all four >=80% native speed checks passing.
Keep the E8 numerical recipe fixed. EXP03/EXP05-A1 completed: existing-slot retention
improves but gates fail; no audit opened. A2 query-first selection with corrected three-value overwrite is complete and
fails development. A3 chronological matching also failed. A4 is now complete: 86.72% competing
bindings, 94.79% corrected overwrite and 68.35% native speed; development fails.
A5 is also complete: 90.23% competing bindings, 95.70% corrected overwrite,
63.80% diverse256 and 67.57% native speed; development fails. All A1–A5 audits
remain closed. False record recognition causes both cache pressure and filler
selection. Next is registered A6 fixed record-presence negative cost, keeping
the A5 key objective, A4 inference/admission, existing state and all gates.
Read [the A5 report](../../reports/exp0305a5/SHoEbox_EXP0305_A5_Report.md),
[diagnosis](../../reports/exp0305a5/DIAGNOSIS.md) and
[exact A6 registration](../../reports/exp0305a5/A6_REGISTRATION.md).
Preregister fresh world/document groups excluding all opened audits and account
for each pilot, padding byte, auxiliary update and CPU cost. Local fixture
supervision remains distinct from generic byte objectives.
See [the handoff](../../DEVELOPMENT_HANDOFF.md).

### Completed A6 and next A7

Fresh matched E8/A5/A6 used exactly 20.25M bytes, seeds 1/7/19. A6 fails the
unchanged development gate: 87.890625% competing bindings, 94.140625% corrected
overwrite, 95.3125% clean, 84.375% diverse256 and 67.583762% native speed.
E8 remains public; all A1–A6 audits remain closed. On the first-32-world trace,
signature survival improves 93.75%→100% and false recognition 15.913655%→3.112450%,
while nearest-key identity stays 96.875%. These subset measurements do not
replace full autoregressive endpoints. All historical scores/costs above remain.

Next is [registered A7](../../reports/exp0305a6/A7_REGISTRATION.md), one diverse
record-context curriculum in the existing short continuation spacers. Keep
A6 arithmetic/state and all gates, including >=80% isolated native speed.
Read [the A6 report](../../reports/exp0305a6/SHoEbox_EXP0305_A6_Report.md) and
[diagnosis](../../reports/exp0305a6/DIAGNOSIS.md). A7 is not implemented or fitted.
This status supersedes the older next-step statements above.


## EXP03/05-A7: diverse record-context curriculum (16 November 2026)

A7 is implemented and its registered development campaign is complete. It fails;
qualified public E8 and every historical capability status remain unchanged.
Matched E8/A6/A7, seeds 1/7/19, used exactly 20.25M independent fitting bytes.
All A1–A7 audits remain closed.

| A7 endpoint | Result |
|---|---:|
| Clean binding | 87.174479% |
| Competing bindings | 66.796875% |
| Corrected three-value overwrite | 75.260417% |
| Diverse256 | 84.765625% |
| Isolated native throughput | 69.713538% (80% required) |

On matched first-32-world teacher-fed traces, A6/A7 signature survival is 100%
in both arms and removals are zero. False recognition improves 2.359438% to
0.853414%, but competing nearest-key identity falls 95.833333% to 89.0625%.
These subset measurements do not replace full autoregressive outcomes.
A8 registers alternating 64-episode PAD/diverse blocks; it is not implemented
or fitted. No new slots/state, public-policy change, audit opening or generic
qualification occurred. All original scores, timings, costs and gates remain.
Read [A7 report](../../reports/exp0305a7/SHoEbox_EXP0305_A7_Report.md),
[diagnosis](../../reports/exp0305a7/DIAGNOSIS.md) and
[A8 registration](../../reports/exp0305a7/A8_REGISTRATION.md).


## EXP03/05-A8: balanced PAD/diverse curriculum (16 November 2026)

A8 is implemented and its registered development campaign is complete. It fails;
qualified public E8 and all earlier capability qualifications remain unchanged.
Matched E8/A6/A8, seeds 1/7/19, used exactly 20.25M independent fitting bytes.
All A1–A8 audits remain closed. The only fitted intervention is the exact
alternating 64-episode spacer curriculum; A6 model arithmetic/state is unchanged.

| A8 endpoint | Result |
|---|---:|
| Clean binding | 87.890625% |
| Competing bindings | 80.859375% |
| Corrected three-value overwrite | 78.515625% |
| Diverse256 | 83.593750% |
| Isolated native throughput | 68.587044% (80% required) |

Matched A6/A8 first-32-world teacher-fed traces retain 100% of target signatures.
False recognition falls 3.212851% to 1.857430%, but competing nearest-key identity
falls 92.968750% to 88.802083%. These subset measurements do not replace the full
autoregressive outcomes. All phases occur in both spacer blocks; the intended
balance is verified. A9 registers one training-only record-key update mask:
keep A8 inputs and update record-key heads only in PAD blocks. It is not
implemented or fitted. Presence/query/value/applicability/native rules and all
original gates remain. No new public feature, generic qualification or audit
opening is claimed. See [A8 report](../../reports/exp0305a8/SHoEbox_EXP0305_A8_Report.md),
[diagnosis](../../reports/exp0305a8/DIAGNOSIS.md) and
[A9 registration](../../reports/exp0305a8/A9_REGISTRATION.md).

## EXP03/05-A9: pre-fit allocation blocked (16 November 2026)

Selective record-key credit is implemented and its numerical/cursor/isolation
controls pass. The mandatory 128-world binary regression sample cannot be
formed under the exact registered salt and exclusions: only 96 are available.
No fitting, development inference, isolated throughput, diagnosis or audit
was executed. There is no new capability qualification or model failure result.
All gates and samples remain unchanged. A9-F1 registers only the binary-pool
allocation repair; it is not implemented. Read the linked A9 report and F1
registration before continuing. All earlier results/timings/costs are retained.

## EXP03/05-A9-F1: selective credit completed (16 November 2026)

F1 selects exactly 128 binary worlds from 336 eligible, preserving all 896
primary worlds, every exclusion and original stream/credit hash. All pre-fit
controls pass and precede the exact independent 20.25M-byte E8/A8/A9 campaign.
A9 fails development: competing and corrected overwrite 88.281250%, clean
94.531250%, diverse256 86.328125%, isolated speed 68.202933% native.

Registered frozen subset diagnosis improves nearest-key identity 90.625% to
93.75%, record error 4.973879 to 4.076629 and changed-key old-value predictions
9.375% to 1.5625%. Signatures survive at 100%; false recognition is identical
at 1.405622%. These teacher-fed subset measures are separate from full
autoregressive endpoints. All outcomes were locked before diagnosis.

Public E8 and historical scores/costs remain unchanged. A1–A9 audits remain
closed. [A10](../../reports/exp0305a9f1/A10_REGISTRATION.md) registers only exact
row-parallel affine execution, with zero research fitting bytes. It is not
implemented or measured; a speed pass would not repair A9's accuracy failure.
