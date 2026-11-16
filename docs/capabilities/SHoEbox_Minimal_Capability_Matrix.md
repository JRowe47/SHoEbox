# SHoEbox minimal-baseline capability matrix

16 November 2026 (America/Denver) · Runtime: minimal corrected C, v19 · Source: `317fbcd023b9d9b3c2b359ba8f99f3dadbf46783`.

Current inventory: **62 core capabilities, four optional multimodal capabilities, and eight engineering enablers**. This matrix is the current implementation/evidence view. The older full-C assessments are preserved in each backlog row's `historical_assessment` and in repository history.

**Yes** means a relevant executable mechanism remains; **Partial** means supporting pieces exist without a complete workflow; **No** means the capability is outside the minimal implementation. A research decoder is identified explicitly. **Untested** means no adequate minimal-specific capability result; it does not turn an old full-C failure into a minimal result. **Verified engineering** covers software contracts, not semantic competence.

No broad generalist capability is generally demonstrated. Minimal evidence supports next-byte learning, short selective copying, bounded-state reconstruction, and causal use of memory on the copy task. The separate EXP01-D3 research readout now qualifies narrow two-binding native recall across boundary phases and padding delays; it is not yet the production default.

EXP01-E8 passes complete registered research qualification: 98.2747% held-out
binding retained through 1M language targets, 97.2656% offset mean, 93.75%
worst offset, 96.875% both query orders and 100% padding delays. Fresh generic
language increases 0.006233 nats/byte; copy/reconstruction remain within their
registered margins. Three-seed 1M/5M and >=80% throughput gates pass.
See [E8 results](../../reports/exp01e8/SHoEbox_EXP01E8_Report.md).
Next: [versioned public integration](../../reports/exp01e8/PUBLIC_INTEGRATION_PLAN.md)
with numerical and checkpoint parity before default promotion. E2-family
binding and E7 generic audits are now opened; never tune against them.

## Retained architecture

- Byte input/output; width 64; exact corrected-C sparse experts at 1 and 4 bytes.
- Ordered trainable summaries at 16 and 64 bytes, diagonal recurrence, memory reads and learned constant output modulation.
- MinKron context, addressed memory, retained-unit retrieval, reconstruction and bounded child credit.
- Separate fast/slow/query learning rates, coverage-only warmup, one-pass prediction and v19 exact checkpoint continuation.
- Upper expert banks/routing/forecast heads, state-dependent upper modulation, D predictors, loops, route-rate cycles and experimental credit multipliers are outside minimal.

## Minimal qualification

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
| M07 | Bind a newly observed value to the right query | Yes — memory path; research paired readout | Research narrowly qualified; default unworking | E8 three-seed audit: 98.2747% native whole-episode exactness, retained through 1M language targets; 96.875% both orders; query/state/pairing ablations pass. Explicit fixture-local labels, 7,670 budgeted weights, no extra sequence state. Generic and 1M/5M gates pass. Next: versioned public integration. |
| M08 | Retain information through longer delays and interference | Yes — memory path; research readout | Narrow for padding delays; interference untested | E8 audit reaches 100% through +64/+128/+256 unseen padding bytes across all offsets on the registered delay sample; after language retention too. Arbitrary distractors, additional bindings and eviction remain EXP03/05 work. |
| M09 | Recall despite noisy or approximate cues | Yes — memory path | Untested | Prior noisy/fuzzy full-C probes failed; minimal noisy-cue behavior is unqualified. Next: EXP03. |
| M10 | Update working variables without overwriting unrelated state | Partial | Untested | Research tasks exist; full-C initialization/set/copy/swap failed EXP00. No successful minimal state-update workflow. Next: EXP04, EXP05. |
| M11 | Maintain useful episodic memory across sessions | Partial | Untested | Checkpoint persistence exists; autonomous episodic admission, retrieval and answer use across sessions are unqualified. Next: EXP14. |
| M12 | Track who said what, when, and with which source | Partial | Untested | Temporal/address features exist; semantic speaker, time and source attribution are unqualified. Next: EXP14, EXP16. |
| M13 | Select memory writes, updates and forgetting appropriately | Partial | Untested | Bounded write/replacement/reset policies exist; semantic write selection and correction handling are unfinished. Next: EXP03, EXP05, EXP14. |
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
| C02 | Teach internal state through intermediate supervision | Partial — research local readout | Readout qualified; upper-summary specialization unqualified | E8 jointly trains native core with a fixture-supervised extraction/applicability readout; binding, retention and generic regression gates pass. This does not establish generic byte-loss training of the local heads or learned upper-summary specialization. Next: public integration, then separately registered broader objectives. |
| C03 | Improve answers by performing extra latent refinement | No — full research only | Untested | Existing refinement loops are excluded from minimal; prior full-C tests did not establish useful inference-time refinement. Next: EXP10. |
| C04 | Stop refinement when its state has converged | No — full research only | Untested | Convergence stopping exists only with the full-research loops, not in minimal one-pass prediction. Next: EXP11. |
| C05 | Allocate more computation to genuinely difficult inputs | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP11. |
| C06 | Search alternative solutions and select with a verifier | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP12. |
| C07 | Detect and correct its own mistakes | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP12. |
| C08 | Express calibrated uncertainty and abstain appropriately | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP19. |
| C09 | Learn from corrections, feedback and preferences | No | Untested | No dedicated integrated workflow in the minimal baseline. Next: EXP14. |
| C10 | Adapt continually while retaining prior capabilities | Partial | Narrow research fixture | E8 retains 98.2747% held-out binding through 1M generic language targets across three seeds; independent generic regressions pass. Broad continual adaptation remains unqualified. Next: public integration, EXP14. |
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

EXP01-D3 / EXP22 passes the registered binding/access development gates: 99.22% native two-answer exactness on every seed versus 50.29% native; 99.63% all-offset mean, 98.44% worst offset, 96.88% both-query-orders exactness, and 100% at +64/+128/+256-byte delays. [Results and evidence](../../reports/exp01d/SHoEbox_EXP01D_Report.md) establish the paired constituent
readout as the selected research solution. EXP01-E: integrate and qualify joint native-core learning and a learned residual-applicability gate, then language/copy/reconstruction and 1M/5M capacity regressions. Keep the D3 binding/delay fixture as a regression and minimal v19 as production until those gates pass.
See [the handoff](../../DEVELOPMENT_HANDOFF.md).
