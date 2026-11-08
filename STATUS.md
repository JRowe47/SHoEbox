# SHoEbox handoff

## Current: wider cortical transport, 1k profile complete

`cortical_wide()` has **14,976,882 parameters**, independent **4/8/16-byte**
internal ports, widths **128/96/64** at orders 1/2/later, and **4 of 32** experts
selected per regional layer. The first two orders hold about **55%** of capacity.
The 18-region topology and EC-only hippocampal boundary remain in place.

Completed **1,000 FineWeb byte targets**, B=4, TBPTT=32, eight updates, with
256-target learning-rate warmup and constant 0.001 thereafter. Training
forward/backward/update took **194.62 seconds**, excluding probes and a
10.42-second temporal-gradient diagnostic. Fixed training-slice probe NLL:
**5.584954 -> 5.060832**. This is a cold-start profile, not a holdout benchmark.

Final A1/B1/L2 primary ports emit **123/126/127** unique complete 16-byte frames
across 128 probe publications. Context ports remain near-uniform in their soft
probabilities, with L3 and EC-C showing reduced conditioning variation. The
old trained L3 1-byte context used only three codes on this probe (82% one code).
Task gradients reach A1/B1/L2 across 31 observations, at 2.8–5.9% of each
region's same-observation gradient RMS.

Implemented per-forward projection/key caches, terminal HAT reads retaining
all recurrent writes, private-bank dispatch, removal of unused lace work,
softmax reuse, and exact frame compaction at TBPTT boundaries. An initial
instrumentation leak was fixed; corrected training RSS plateaus around 4.3 GiB.
Matched B=4/T=32 execution is **16.6% faster** (30.18 -> 25.89 seconds), with
**23.4% less peak RSS** (5.21 -> 3.99 GiB). All **17 checks** and the endpoint
reload/data/state audit pass. The remaining operator profile points first to
fusing MinKron HAT intermediates and backward, then expert-gradient accumulation.

Endpoint: `artifacts/cortical_wide_profile/last.pt`, with optimizer, RNG and
carried state after 250 observations per stream. The matching report and raw
profiles document timings and operator allocation traffic. Source and runs are
delivered as downloadable files; this pass has not been pushed to GitHub.

[Wide configuration and reproduction](docs/cortical_wide_profile.md).

## Previous: 10M cortical FineWeb run complete

The cortical model has **9,982,290 parameters** and completed **100,000
FineWeb byte targets in one pass**, across 250 updates. All 18 regions, complete
byte ports, two rounds, and the EC-only hippocampus remain in place.

Fixed-holdout NLL: **5.556483 -> 3.266079**; final next-byte accuracy:
**17.65%**. Training took **17.63 minutes** at
**94.52 bytes/s** on two CPU threads. All 11 behavioral/data/runner
checks passed, and the final checkpoint reload/inference and exact byte/state
accounting audit passed.

The result is close to a byte-frequency predictor: the unigram count baseline
scores 3.255803 NLL, while the bigram baseline scores 2.680461. Samples remain
character-level fragments. Pure task gradients reach every region and the
hippocampus. At the endpoint, EC-C and L3 emit only 4 and 8 distinct complete
4-byte frames, respectively, across the 200 current stream states; Y's greedy
byte is identical across those states. The full loss curve and these raw
observations are in the run package. No training remains pending.

The final endpoint is `artifacts/cortical_fineweb_10m/last.pt`, with optimizer,
RNG and all carried states, at 100,000 targets / 500 observations per stream.
The exact corpus and unpublished source patch accompany the downloadable
checkpoint package. GitHub publication was blocked by automatic approval
review; the code and completed run are delivered as files.

[Training and reproduction](docs/cortical_fineweb.md) |
[Compact measured results](docs/cortical_fineweb_results.json).

## Previous: cortical input map v2 skeleton

The active development model is now `shoebox.CorticalModel`: 18 ordinary
HAT-SHoE regions and one shared hippocampus behind EC-C/EC-X. The old local
retention, conflicting-query, reserved-question and owner-readback mechanisms
are absent from this model. The older studies below are historical and do not
define the next development target.

The 635,466-parameter skeleton has complete source-tagged byte frames,
the map's ordered recipes, previous-round context, two-occurrence transport
queues, independent regional state, and once-per-observation episodic storage.
All nine behavioral checks passed locally. A 24-update fixed-fixture run reduced
NLL from 5.571365 to 3.629863 in 59.27 seconds on one CPU thread.
CI regenerates checkpoints, traces, a Markdown/PDF report and source package.
See [commands and implementation](docs/cortical_skeleton.md).

Next: use this model for ordinary stream/composition shakedowns. Keep the
18-region connectivity fixed while checking frame usage and learned context.
Do not resume the archived paired-question retention target.

## Archived development record

## Latest: first-class memory interface

The architecture-review refactor is implemented. `MemoryContextConfig` adds a
reserved query and up to eight distinct, versioned slots per consumer. All
nodes can request evidence. The common slot reader reaches routing/experts and
the final composer, and owner reads preserve direct content. Working-state
updates are independent of durable publication admission. Source/abstract
anchors share regional event bindings, repeated rounds revise existing pages,
and fetches are cached under a bounded fair request budget.

Use `scripts/run_memory_context.py` for an ordinary-loss continuation and
`scripts/evaluate_memory_context.py` for its compact acceptance bank. Extra
lace credit remains excluded from this phase. Exact resume retains source and
protocol guards; a new continuation uses a new output directory.

All 94 checks pass, exact optimizer/state resume passes, and 227 guarded files
match the training source. The fixed four-update continuation finishes at
`artifacts/memory_context/pilot/last.pt` (next document 667). Compact paired
accuracy remains 0/12 at +0/+32/+64; single/correction retention is 4/4 and 15/16
at +64. Paired NLL falls to 1.2978 at +64, and language NLL to 3.3627. No model is
promoted; keep `artifacts/memory_context/parent.pt` as the retained working model.

[Implementation and commands](docs/memory_context.md) |
[Measured report](docs/memory_context_report.md).

## Latest: lace-separation pilot complete

The default-off rank-four query head, helpful-unique credit and selective
shared-lace inhibition are implemented. All four matched arms complete their
fixed 32-update continuation from
`artifacts/hippocampus_binding/ca1/bounded/last.pt`. All 38 locked evaluation
sections are complete and audited. All 52 focused checks pass, and a separate
production-checkpoint audit reproduces one complete bypassed update exactly.

**Keep `artifacts/hippocampus/parent.pt` as the working model.** Every arm fails
the paired and correction validation gates. B/C/D pass single-fact validation;
A fails it. All four pass the natural-text gate. No checkpoint is promoted.

| Final native pairs | +0 / 192 | +32 / 192 | +64 / 192 |
| --- | ---: | ---: | ---: |
| A: bypass | 0 | 3 | 0 |
| B: learned head | 1 | 1 | 6 |
| C: unique credit | 0 | 1 | 1 |
| D: full rule | 2 | 1 | 0 |

B is the strongest continuation to inspect next: +64 single/correction accuracy
is 96.9%/88.3%, versus A's 90.6%/74.2%. C drops to 79.7%/69.5%; D recovers to
95.3%/84.4% without solving paired recall. Unseen wording does not produce a
reliable gain. D's credit is live (141/168 eligible final +32 decisions updated),
but swapping its dedicated cues leaves all 1,536 native predictions unchanged.
The same-weight bias/readback controls and full per-case predictions are saved.

Next development target: trace each question's target-versus-contender margin
through owner readbacks, expert inputs and answer readout at frozen weights.
Use source-specific interventions to locate the binding loss. Favor the simpler
head for that investigation; retain the extra-credit switches for later ablations.

Recovery: the fixed experimental endpoints are
`artifacts/lace_separation/{A,B,C,D}/last.pt`, all at step 32 / next document 856.
Initial and 8/16/24/32 snapshots preserve weights, AdamW, RNG and carried language
state. Each arm consumes 768 answer targets, 23,680 logical task bytes, 20,032
executed batch-byte positions and 4,096 language targets. Source ledgers match;
all 223 guarded files still match the training revision. No training or final
evaluation is pending.

Next command after restoring both archives:
`python scripts/summarize_lace_separation.py` verifies the completed evidence.
Exact resumes keep source/protocol guards; new training requires an explicit new
continuation. The public implementation revision is `827d8861c0e821a4e568d58db50fe72123aaeb28`.

[Report](docs/lace_separation_report.md) |
[Protocol and reproduction](docs/lace_separation.md) |
[Audited results](experiments/text_regions/results/lace_separation/summary.json).


## Earlier: hippocampus retrieval control repair

The new controls independently select an external-event cue clock, direct
expert-input reinstatement and bounded CA1 reread acceptance. The controller
repair removes suppression of refined evidence by the original competing
accept weight. A reread gets its own terminal accept/null decision. Owner
readers, bounded fetches, native rounds, global objectives and the 391,174
hippocampus parameters are unchanged. Legacy defaults preserve prior behavior.

The event/expert arm completes 64 updates from the original strong parent.
Matched legacy/bounded CA1 arms then each complete 16 additional updates from
that endpoint, with identical initial weights, optimizer, RNG, language state,
stream cursor and next document. Their source schedules match exactly.

**Keep the original retained parent as the working model.** Bounded CA1 has
100% single-fact and correction validation at gaps 0/32, and text NLL 3.161174
versus the original parent's 3.312089. However, both conflicting questions pass
in 0/96 histories at both gaps. The paired-recall promotion gate remains open.
The matched legacy continuation scores 0/96 and 1/96 pairs, with correction
accuracy 100%/85.9% and text NLL 3.112721.

All 42 targeted checks pass. A trained-checkpoint comparison against public
revision `59244b3` also produces exactly identical logits in legacy mode on
16 histories and 32 question branches. The completed report separates the
192-history placement tests from the fresh 96-history CA1 tests and includes
same-weight blocking/controller interventions and final signal audits.

The fresh CA1 test bank confirms the remaining newest-fact bias: corrected
immediate recall is 100% for the newer fact and 25% for the older fact, with no
query-induced answer switches among 96 conflicting pairs. Paired correctness
is 0/96 at all three delays. The final signal audit has about 8.1 times stronger
returned residuals than the matched legacy continuation, while query cue cosine
is still 0.999947. Next target: query-conditioned selection within the regional
cue/owner-read path; stronger delivery alone has not resolved binding.

On the fresh +64-byte retention bank, corrected/legacy single-fact accuracy is
98.4%/98.4%, correction accuracy is 97.7%/79.7%, and text NLL is
2.976824/2.955399. Blocking corrected readbacks or restoring its legacy policy
at fixed weights still gives 0/96 pairs at every delay.

Recovery: the working parent is `artifacts/hippocampus/parent.pt`; the new
placement endpoint is `artifacts/hippocampus_binding/event_expert/last.pt`;
matched continuations are `artifacts/hippocampus_binding/ca1/{legacy,bounded}/last.pt`.
Validation selects bounded `best.pt` at update 0 and legacy `best.pt` at update 16;
the reported matched results use both `last.pt` files at update 16. Exact-resume
source guards remain enforced; use the recorded source or start an explicit new
continuation. No additional training is pending from this pass.

[Report](docs/hippocampus_binding_report.md) |
[Protocol and recovery](docs/hippocampus_binding.md) |
[Audited results](experiments/text_regions/results/hippocampus_binding/summary.json).

## Earlier: explicit shared hippocampus implementation

The default-off hippocampal extension implements dedicated regional cues,
immutable owner-local anchors, shallow DG/CA3/CA1 controls, bounded shared lookup,
conditional owner readers and causal next-round commits. Regional intelligence
and the ordinary optimizer/objectives stay in place. The original strong
`artifacts/memory_selectivity/selective/best.pt` remains the working baseline.
Its bundled copy is `artifacts/hippocampus/parent.pt`.

[Implementation/protocol](docs/hippocampus.md) follows the attached
[hippocampus specification](docs/SHoEbox_Hippocampus_Spec_v1.md). Tests cover
legacy/zero-residual parity, live source/query/read credit, isolated forks,
causal versions, stale descriptors, revision, eviction, checkpoint migration,
exact optimizer resume and sparse activation recomputation. A lazy paged
reference service and bounded actor fabric exercise 64/128/256/512 mock regions.

The matched A/B/C arms finish 64 updates from the retained parent. A keeps the
unchanged model, B adds local cue/anchor access, C adds the shared index. A's
previous endpoint is reused after exact 30-update reproduction and complete
protocol/source-ledger verification. Every arm consumes the same 1,536 answer
targets, 46,976 logical task bytes, 40,384 executed task positions and 8,192
language bytes. All regional parameters remain trainable.

**C fails the declared promotion gate.** On the locked validation bank, both
conflicting questions pass in 3/96 immediate histories and 0/96 after 32 bytes.
Single-fact accuracy stays 100% at both gaps. Correction accuracy is 95.3%/93.0%,
and natural-text NLL improves from 3.312089 to 3.173513. Retain the original
strong parent, and retain the extension behind its default-off switch.

The implementation passes 51 targeted checks. H0-H2 structural exits and the
bounded H4 actor smoke pass; H3's behavioral exit fails. Actual distributed/GPU
deployment remains H5. Shared training used sparse segmented recomputation after
update 14 to fit host memory, preserving live credit and unused `None` gradients.
The recorded phase schedule and exact reproduction commands are in the docs.

[Completed report](docs/hippocampus_implementation_report.md) |
[audited results](experiments/text_regions/results/hippocampus/summary.json).
The checkpoint archive contains the original parent and A/B/C endpoint plus
validation-selected checkpoints. They restore under `artifacts/hippocampus/`;
the exact corpus restores to `data/tiny_shakespeare.txt`.


## Earlier: paired histories expose a query-binding bottleneck

Memory-only focus continues. **Keep `artifacts/memory_selectivity/selective/best.pt`
as the working joint-language/retention baseline.** The new paired continuation
is an experimental branch, not a replacement: it trades retention/revision for
greater older-value influence without reliable entity-specific selection.

The unchanged 312,440-parameter model completes 64 updates. Each paired template
executes one history, then forks live state for both native questions; both
losses backpropagate into the same history. The schedule mixes paired, paired,
single and correction updates. All weights, AdamW moments, language state/cursor,
RNG and the training document cursor carry forward. There is no replay buffer.

Validation selects update 64 (paired NLL 1.651034 -> 0.852152). Fresh seed-1031
tests show older-fact accuracy of 46.1% immediately and 44.1% at +64 bytes,
versus 25% for the parent. However, both conflicting questions are correct in
0/192, 1/192 and 1/192 histories at +0/+24/+64. Query-only changes switch outputs
2/192, 1/192 and 5/192 times. Newer-fact accuracy at +64 falls to 73.4%.
Single-fact retention falls from 100% to 85.2% at +64, and from 86.7% to 68.0%
at +128. Changed-value corrections at +64 fall from 100% to 64.1%.
Natural-text NLL stays close: 3.132340 -> 3.134060. Ordinary-stream later exact
accuracy stays 35.7% (84 queries); corrected exact is 43.3% versus 45.0%
(60 queries), and long-delay later exact is 32.5% versus 33.8% (77 queries).

The signal audit resolves a key ambiguity: positional diagnostic readouts
recover both old/new values from floating state in 256/256 fresh snapshots at
gaps 0 and 32, for both models. An entity-conditioned diagnostic probe solves
177/192 immediate conflicting pairs but only 40/192 delayed pairs in the parent;
the continuation scores 168/192 and 50/192. These probes never enter model
inference. Source keys/values and query keys receive nonzero gradients.
Preserved content is available; query-dependent selection and stable binding
are the next implementation target.

Budget: 1,536 answer targets (1,024 paired, 256 single, 256 correction), 46,976
logical task bytes, 8,192 language bytes; documents 660 through 771. Shared
histories avoid 11,968 duplicated task positions (22.9%). All 16 unique targeted
checks pass, including exact resume and next-phase continuation. Production
logits/gradients/states match duplicated prefixes; 757 parameter tensors and
275 state tensors are checked. Budgets/selection reconstruct and weights are finite.

Next phase: implement an ablatable learned interaction between query identity
and preserved content in the existing expert/recall path. Compare against paired
supervision alone, starting from the retained strong parent. Require actual
paired correctness and retention/correction preservation before promotion.
Keep broad GPU preparation, scaling and higher-order curricula deferred.

The experimental checkpoint is `artifacts/paired_recall/train/best.pt` (also the
final step-64 weights). [Protocol/results](docs/paired_recall.md) |
[audited evidence](experiments/text_regions/results/paired_recall/summary.json).

## Completed: stronger retention and the remaining fact-selection failure

Continue the memory-only development focus. GPU preparation, broader integrated
acceptance and scaling remain deferred.

Starting from acquisition update 96, matched 48-update retention arms compare
learning rates 0.003 and 0.001. Both select their final update by validation;
0.003 wins with 99.4% validation accuracy and NLL 0.055799, versus 47.5% and
1.056687. Both arms start with identical weights, optimizer moments, states and
source positions, and consume identical data schedules.
Each arm adds 768 answer targets, 34,944 task bytes and 6,144 language bytes.

A further 48-update selective continuation carries the winning checkpoint's
weights, optimizer, language state/cursor, RNG and document index. It cycles
single-fact retention, older/newer competing facts and corrections, with full
4 x 4 target/nuisance balance on the two-write tasks. It adds 768 answer targets
(192 per mode), 24,576 task bytes and 6,144 language bytes. Validation selects
update 48; the selection score falls from 1.154259 to 0.859912. All parameters
remain trainable and learned controls stay enabled; capacity is 312,440 parameters.

The final selected model scores 100% at +0/+8/+24 natural-text bytes,
98.4% (126/128) at +64 and 76.6% (98/128) at +128. Training insertions stop at
32 bytes. The base single-fact source/query delay is 8 bytes. Complete four-value
sets pass in 30/32 cases at +64 and 14/32 at +128. At +64, values 0/2/3 score
32/32 each; value 1 scores 30/32. The earlier persistent 1-to-0 collapse is gone.
Natural-text NLL returns to 3.132340, versus 3.126049 for the acquired parent
and 3.420890 after the selected retention-only continuation.

Competing-fact selection is still unresolved. Older-fact recall is 25% at
0/24/64 added bytes, with zero complete four-value sets. Both conflicting facts
are correct in 0/192 matched query pairs at every gap. Changing only the query
entity changes the prediction 0/192 times at gaps 0 and 24, and 1/192 at 64.
Lower loss therefore reflects confidence changes, not successful fact selection.
Changed-value corrections pass 192/192 controlled cases at +64. In full ordinary
streams, later exact queries rise from 29.3% to 36.6% (82 targets), corrected
exact queries from 32.8% to 42.2% (64), and long-delay later exact queries from
23.8% to 28.6% (84). These remain distinct from controlled single-fact retention.

All 15 targeted checks pass, including exact resume and rejection of the
latest-value shortcut. Completed production resume preserves checkpoint bytes;
a fresh selective phase can inherit weights, optimizer and both data cursors.
All budgets and selections reconstruct from evidence, weights are finite, and
parallel evaluation matches serial scores and traces exactly.

Next memory target: trace the entity/attribute-key paths into writes and query
reads, then train both query directions from the same history within one update.
Use `artifacts/memory_selectivity/selective/best.pt` as the current joint
language/retention parent; retain the two retention arms as controls. Preserve
existing ablatable features and add no deliberately higher-order curriculum.

[Protocol and reproduction](docs/memory_selectivity.md) |
[Audited results](experiments/text_regions/results/memory_selectivity/summary.json).

## Completed: balanced memory acquisition and retention

Memory-only development focus; GPU preparation, broader integrated acceptance
and scaling are deferred at the user's request.

The unchanged 312,440-parameter graph learns source-conditioned values after
dense balanced acquisition training. Held-out immediate recall rises from
25.0% to 97.7% (125/128), with 29/32 complete four-value sets. All parameters
remain trainable; no answer labels, owner hints, forced writes or replay enter
inference. Learning rate 0.003 wins the 16-update validation screen over 0.001.
The acquisition continuation completes 96 updates from connected recall's
selected update-16 parent.

A retention branch starts at acquisition update 48 and carries its weights,
AdamW moments, language state/cursor, RNG and next document index. It trains
48 further updates with 0/4/8/16/32 added natural-text bytes. Both lineages have
96 updates, 1,536 answer targets and 12,288 natural-source bytes; retention
adds 8,832 distractor bytes. Their validation selections are their final weights.

Retention scores 75.0% with 0/8/24/64 added bytes and 68.0% with 128. The base
source-to-query delay is 8 bytes. Through the 64-byte insertion, values 0/2/3
are correct and stored 1 consistently produces 0; no complete four-value sets
pass. Acquisition-only scores 24.2/19.5/16.4/15.6% at +8/+24/+64/+128.

On the ordinary full task, later exact queries improve from 28.2% to 49.4%
and corrected exact queries from 26.2% to 57.4% in both continuations. The harder
long-delay stream with competing writes remains weak: retention's later exact
accuracy is 29.6% on 81 queries. Semantic/context answers weaken under focused
exact-value supervision. Held-out natural NLL is 3.109 before, 3.126 after
acquisition and 3.505 after retention. The 32-update memory-only diagnostic
does not beat the joint run on validation and sharply worsens language loss.

Four data-integrity checks pass. The first retention update with gap zero
exactly matches uninterrupted acquisition update 49 in weights, optimizer,
language state/cursor, RNG and document index. All training budgets and selected
checkpoints reconstruct from saved evidence; all final weights are finite.

Next memory experiment: begin retention from the fully acquired update-96
checkpoint, resolve the 1-to-0 confusion, then train relevance among competing
facts and corrections. Preserve the broader controls; add no deliberately
higher-order curriculum. The current gap branch began before acquisition had
fully separated the four values.

[Protocol and results](docs/memory_acquisition.md) |
[Audited summary](experiments/text_regions/results/memory_acquisition/summary.json).

## Completed: connected recall continuation and bounded activation memory

Both matched arms finish 16 paired-source updates from the selected update-32
state-credit parent: 8,192 natural bytes and 8,192 task bytes, with 374 answer
targets per arm. Four streams use 128-byte optimizer windows and 16-byte
execution chunks. Both select update 16 on validation.

Exact source coverage rises from 64/291 (22.0%) to 209/291 (71.8%) queries with
the connected path. Immediate acquisition coverage is 86/90 versus 51/90;
exact queries beyond 96 bytes receive 16/71 versus 0/71 connected derivatives.
The audit checks actual source occurrences and resets, not nominal delays.

Held-out natural/answer NLL is 3.1089/1.6659 for connected credit versus
3.1065/1.6524 for the old cutoff; the parent is 3.1215/1.7924. Balanced
four-value immediate recall remains 25%, with zero complete sets in both arms.
The run establishes no useful value-conditioned recall advantage.

Optional activation recomputation drops measured peak process RAM from
1,634.1 to 685.2 MiB on a matched 64-byte, four-stream paired update, with
38.59 to 55.37 seconds runtime. State, traces and cursors match exactly;
post-update weights differ by at most 2.98e-8 from accumulation order.
The option stays off by default; this paired study uses ordinary activations.
Training kernels remain selected during checkpoint forwards, including large
banks and batched stages. All 43 targeted checks pass; exact resume works.

Next: prepare device-aware batches and checkpoint restoration for the planned
laptop/GPU run, then broaden ordinary shared-stream exposure with larger
connected windows. Reliable immediate value-conditioned recall remains the
behavioral target. Keep the existing controls ablatable and add no special
higher-order curriculum. The repair trainer retains its 384-byte default.

[Report](docs/SHoEbox_Connected_Recall_Report.md) |
[Protocol](docs/connected_recall.md) |
[Audited metrics](experiments/text_regions/results/connected_recall/summary.json).

## Completed: repair the historical content path

`ContentCreditRun` separates 16-byte execution chunks from a default 384-byte
optimizer/gradient window. Internal chunks carry the complete live graph.
Historical eligibility is captured once at window entry: capture itself detaches
state, and applying internal receipts would double-count exact autograd credit.
Global norm cap 1.0, shared language/task updates and inference are unchanged.

The original 148-byte source-to-answer derivative is restored from zero to
0.013164902495110228, with zero difference from the full reference in both
gradient and logits. All 35 relevant regression checks pass, including exact
resume, row resets, older credit, all-parameter gradients and runtime fusion.

The default 384-byte smoke update completes: 384 language bytes plus 384 task
bytes, 19 answer targets, 757 changed parameter tensors and finite weights.
It takes 217.3 CPU seconds and peaks at 7.07 GiB process memory. Immediate
acquisition stays at 25% after this single update; its checkpoint is not
validation selected. [Measurements](experiments/text_regions/results/content_credit/summary.json).

Use the [connected continuation protocol](docs/content_credit.md) for subsequent
training. Exact content credit ends at optimizer boundaries; existing older
conditional traces retain their limited scope. The next target is ordinary
shared-stream training that establishes reliable value-conditioned short recall,
then longer-delay recall. Keep the broader architectural controls ablatable.

## Completed: signal and gradient headcheck

The November 7 audit of runtime `9f813f7` and the update-32 state-credit
checkpoint found live forward modulation and nonzero answer-only gradients in
all 757 parameter tensors. Global clipping is active: norm 3.807 becomes 1.000.
Publication gates are unsaturated, and recall/feedback interventions change outputs.

Native immediate acquisition still predicts `1` for every query (8/32 correct).
Full historical autograd detaches every 16 bytes. Explicit delayed additions
reach only 45 admission-policy and 20 payload-adapter parameters; older content
construction remains outside that credit path. A 148-byte observation derivative
is 0.0131649 with an intact graph and disconnected with 16-byte truncation, with
identical forward logits. Sixty-three relevant regression checks pass.

Prioritize broader content-level credit and reliable value-conditioned short
recall before interpreting longer-delay scores. Retain global cap 1.0, existing
activation normalization, shared language/task learning, and ablatable controls.
This diagnostic takes priority over the earlier GPU-preparation target below.
No model or trainer behavior was changed in the audit.

[Audit measurements and reproduction](experiments/text_regions/results/headcheck/README.md)

## Completed: credit through recurrent producer state

Continues public commit `4d919d1d4e23d6af3fb052987c2fe1cdd0bcad59` and the
context-balance direct-sum checkpoint at update 128.

`StateCreditRun` adds a training-only producer-state eligibility trace alongside
existing publication and payload credit. The local MinKron decay/erase Jacobian
transports sensitivity to each owner's existing admission weights. Forced
commits omit gate injection; inactive rows hold; document resets clear traces.
Epochs and clocks bind traces to their state, and boundary receipts are single-use.
All 312,440 parameters and the inference graph remain unchanged.

The derivative is conditional on historical module inputs and policy features.
Historical inter-layer/inter-region and already-detached state-to-publication
paths remain truncated. Keep the mechanism independently ablatable and marked
as a potentially threshold-dependent emergent feature. No replay or separate
higher-order-memory curriculum was added.

## Completed evidence

Both arms finished 32 paired updates: 4,096 natural bytes, 4,096 task-stream
bytes, 203 answer targets and 1,376,256 expert executions per arm. Every weight
trains. Both collect the new traces; only state-gradient contraction differs.
Validation at 0/16/32 selects update 32 in both arms. Final and selected weights
therefore coincide. The state arm's combined validation score is 2.436000 versus
2.437005 for publication-only credit.

| Held-out measure | Parent | State credit | Publication only |
|---|---:|---:|---:|
| Natural-byte NLL | 3.115499 | 3.121512 | 3.124802 |
| Answer NLL | 1.863883 | 1.773150 | 1.768821 |
| Exact NLL beyond 96 bytes | 1.609711 | 1.611467 | 1.602640 |
| Exact accuracy beyond 96 bytes | 32.14% | 32.14% | 32.14% |

Testing uses 1,024 natural targets, 144 ordinary answer targets and 56 exact
long-delay targets beyond 96 bytes. Eight paired observation-change probes
alter only an earlier relevant value; no checkpoint answers both members of
any pair correctly. Producer-state swaps have tiny query-time effects. The new
learning path is verified; this run establishes no recall advantage.

All 571 unique checks pass: the 568-check regression run and three separately
completed probe checks. The current check driver includes both. Fixed-input
FP64 derivative agreement, forced/inactive/reset behavior, disabled-credit
parity, backend agreement, cleanup, source isolation and exact resume pass.

Persistent producer sensitivities add 115,520 bytes per four-stream source.
The small CPU timing probe costs 1.40x the publication-only forward/backward
path. Recording currently recomputes coefficients and bypasses stage batching
for tracked contextualizers. The original ContextBalancedRun has no trace cost.

## Next development target

Prepare the integrated implementation for the planned laptop/GPU run. First
reuse MinKron transition intermediates and preserve batched contextualizers
while recording state credit; test GPU/device and precision behavior. Keep
language and task learning shared. Use greater ordinary stream exposure to test
useful retained content, without adding a special higher-order curriculum or
optimizing publication age itself. Retain all existing controls for later ablation.

For continuing this feature, the archive contains
`artifacts/state_credit/state/best.pt` and its publication-only comparator.
A new continuation should advance the task seed; the current runner's fixed
seed 483 reproduces this completed comparison. Both states/cursors/optimizers
are also retained for exact recovery of the existing protocol.

[Report](docs/SHoEbox_State_Credit_Report.md) |
[PDF](docs/SHoEbox_State_Credit_Report.pdf) |
[protocol](docs/state_credit.md) |
[metrics](experiments/text_regions/results/state_credit/summary.json).

## Reproduce

```bash
python scripts/run_state_credit.py --parent artifacts/state_credit/parent.pt --out artifacts/reproduction
python scripts/run_state_credit_evaluation.py --root artifacts/reproduction
python scripts/check_all.py
```

The archive includes runnable source, corpus, parent, selected/final checkpoints,
optimizer/runtime/credit states, complete logs, causal query traces and hashes.
