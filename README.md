# SHoEbox

**State-owning HAT-SHoE regions with recurrent exchange over persistent tensor ports.** SHoEbox builds on [Sparse Hierarchy of Experts](https://github.com/JRowe47/Sparse-Hierarchy-of-Experts) at revision `b4bb7cfea18c3833819f6ed08973fd014f15f107`, preserving its history, experiments, and recorded results. The default graph has seven nodes; earlier retained-memory experiments also exercise eleven-node layouts.

The new experiment is [experiments/text_regions](experiments/text_regions). It executes sparse expert computation inside each node, keeps runtime memory separate from shared weights, and carries state across ordered text chunks. A second regional round reruns contextualization, semantic routing, all three SHoE blocks, and state updates. External bytes are ingested once.

| Default capacity | Count |
|---|---:|
| Unique trainable parameters | 2,561,212 |
| Nodes / state-owning regions | 7 / 3 |
| Physical expert MLPs | 1,536 |
| Unique expert-matrix parameters | 1,228,800 |
| Logical experts per node | 384 |
| Expert executions per byte per round | 168 |
| Input / output vocabulary | 256 bytes |

## Run

Python 3.10+ and PyTorch are sufficient. Run these commands from the repository root:

```bash
git clone https://github.com/JRowe47/SHoEbox.git
cd SHoEbox
python -m pip install -r requirements.txt
python scripts/check_all.py

# Fast learning check: seven nodes with reduced internal capacity.
python -m experiments.text_regions.train --tiny --context 8 --steps 100 \
  --batch 4 --out artifacts/overfit

# Default 2.56M model: ordered Tiny Shakespeare streams, K=1.
python scripts/prepare_data.py
python -m experiments.text_regions.train \
  --data experiments/hat_minkron/data/tiny_shakespeare.txt \
  --steps 100 --context 128 --batch 4 --out artifacts/language_k1

# Enable full regional recurrence with --rounds 2.
python -m experiments.text_regions.verify
```

## CPU execution

The regional engine uses an optimized PyTorch path by default. It reuses derived routing tables and projections within each forward, specializes the one-event recurrence, and avoids gathers and scatters when every row is active. Weights, topology, expert counts and checkpoint keys are unchanged. PyTorch selects the host's supported CPU instructions, including AVX-512 on the profiled machine.

The second performance pass moves recurrent, symbol-readout and selected-expert tensor operations into PyTorch scripted functions, batches compatible direction normalizations, and simplifies expert dispatch. It requires no external compiler. See the [operator execution report](docs/SHoEbox_Operator_Performance_Report.md) for measurements against the previously optimized release.

The latest pass adds compact expert-gradient accumulation, fused CPU expert and recurrent backwards, version-checked generation caches, batched training contextualizers, and reused gradient statistics. Large expert banks allocate one dense gradient per bank per byte event; AdamW still receives ordinary dense gradients. Large-bank configurations select the combined backward and stage batching automatically. Smaller configurations and inference retain the cheaper scripted path. See the [runtime optimization update](docs/SHoEbox_Runtime_Optimization_Update.md) for matched release timings and allocation counts.

On Linux, training optionally builds the CPU extension once using a local C++20 compiler. No downloads or Ninja are required; other environments use the portable PyTorch fallback. Set `SHOEBOX_NATIVE=0` to disable the extension or `model.batch_independent=False` to disable stage batching. Checkpoint keys and model capacity remain unchanged. Routing projections retain their original multiplication order to avoid amplified rounding differences in long recurrent streams. The runtime optimization report records timings from before this numerical-correctness repair.

Use `model.set_execution_backend('reference')` to select the preserved arithmetic path, or `'optimized'` to switch back. Both paths run the same sparse experts and regional rounds. Caches expire at the end of every forward, including failed calls; they never retain derived weights across optimizer updates.

The [runtime profiling report](docs/SHoEbox_Runtime_Profiling_Report.md) includes timings, checkpoint comparisons and remaining bottlenecks. Reproduce the matched measurements with:

```bash
python scripts/benchmark_runtime.py \
  --checkpoint artifacts/joint_stability/study/20261401/group_clip/best.pt \
  --sizes checkpoint default 25m --repeats 3 --threads 1 \
  --memory-count 512 --profile --out artifacts/performance/benchmark
python scripts/verify_runtime.py \
  --checkpoint artifacts/joint_stability/study/20261401/group_clip/best.pt \
  --out artifacts/performance/checkpoint_verification.json
```

The 25M setting is a capacity stress test initialized from a fixed seed. The checkpoint setting uses the trained eleven-node graph.
Add `--baseline-root /path/to/original/checkout` to compare directly with an untouched source tree; otherwise the benchmark uses the selectable reference backend.
Add `--baseline-backend optimized` when that checkout is a previously optimized release. The benchmark otherwise selects its reference execution path.

## Topology

| Region | Owned nodes | Forward inputs |
|---|---|---|
| R0 | A, B, C | Three causal branches of one byte stream, or explicitly masked independent views |
| R1 | U, P | U receives A+B; P receives U+A |
| R2 | V, Q | V receives B+C; Q receives V+C |

P and Q feed the shared byte readout. B has one state owner and publishes to both downstream regions. Delayed P/Q lateral messages and descending messages are read from a snapshot taken before the round starts. Forward edges read completed stages of the current round. The topology is configuration-driven; the seven-node layout is the default, not a runtime constraint.

## Development status

The [producer-state credit pass](docs/SHoEbox_State_Credit_Report.md) carries
conditional admission gradients through existing contextualizer and expert-block
MinKron state across gradient boundaries. It adds no trainable parameters or
inference state. The optional training path passes 571 unique checks, including
exact conditional-gradient agreement, source isolation and exact resume.
A matched 32-update continuation produces held-out answer NLL 1.7732 with state
credit versus 1.7688 with publication-only credit; exact accuracy beyond 96 bytes
is 32.1% in both arms. The paired fact-change probe shows weak content sensitivity
and no successful paired answer switches. Keep the feature ablatable for larger
shared-stream runs. [Protocol](docs/state_credit.md) |
[metrics](experiments/text_regions/results/state_credit/summary.json).

The [contextual-learning continuation](docs/SHoEbox_Context_Balance_Report.md)
trains language and memory at identical weights within each optimizer update,
then combines their gradients. Natural-text distractors now appear between facts
and queries, alongside competing updates. An optional gradient control caps
each source separately in the core, readout, recall and publication groups.
In the matched 128-update run, direct summation wins the held-out comparison:
language NLL improves from the parent's 3.3672 to 3.0814 and answer NLL from
2.0579 to 1.8560. Exact accuracy beyond 96 bytes remains 23.5%; useful durable
recall is still the next target. All 558 unique checks pass. See the
[protocol](docs/context_balance.md) and
[metrics](experiments/text_regions/results/context_balance/summary.json).

The [unified regional publication interface](docs/graph_publication.md) brings
task-trained write/hold policies, producer-state commits and delayed payload
credit into the ordinary graph. Each region can use downstream recall demand
from the preceding round. Natural text and memory tasks train the same graph,
with independent runtime states and credit histories for the two sources.
The new behavior is optional; existing graph configurations retain their defaults.
The [completed matched run](docs/SHoEbox_Graph_Publication_Report.md) trains a
312,440-parameter graph on both sources. Delayed credit lowers held-out natural
byte NLL from 3.5689 to 3.4339 and answer NLL from 2.5569 to 2.1084. The model
mostly reads fresh publications; longer useful retention is the next training target.

The [regional recall and mixed-learning pass](docs/SHoEbox_Regional_Recall_Report.md)
reuses the recall workspace at ordinary graph input ports, preserving forward and
delayed edge timing with receiver-specific feedback. A seven-node, seven-slot
integration completes training and exact state continuation. The mixed trainer
uses shared weights with independent language/memory state and delayed-credit
traces; both sources retain publication-gated producer state and payload credit.

Two matched 320-update continuations process 163,840 bytes each. Distinct slots
reach held-out natural-byte NLL **2.522 versus 2.570** for pooled slots;
exact-answer NLL is **1.142 versus 1.157**, and long-delay NLL is **1.161 versus
1.179**. Disabling feedback raises the distinct arm's language NLL by 0.135;
blocking reads raises language/all-answer NLL by 0.256/0.264. All **530 checks
pass**. [PDF](docs/SHoEbox_Regional_Recall_Report.pdf) |
[metrics](experiments/text_regions/results/regional_recall/summary.json) |
[usage](experiments/text_regions/README.md#regional-recall-and-mixed-languagememory-training) |
[handoff](STATUS.md).

The [learnable recall workspace](docs/SHoEbox_Recall_Report.md) adds configurable
item-or-null slots, learned overlap controls, slot-specific composer input maps,
and feedback from composition to retrieval and publication. Two-slot runs add
15,444 parameters; seven slots over nine owners also complete a combined state-hold
and payload-credit training check. A matched short continuation favors the pooled
control on held-out loss. The role-preserving architecture is retained for broader
training, with explicit ablations and comments marking potential training-threshold
dependence. All 520 checks pass. [PDF report](docs/SHoEbox_Recall_Report.pdf) |
[metrics](experiments/text_regions/results/task_recall/summary.json) |
[usage](experiments/text_regions/README.md#learnable-recall-workspace) |
[handoff](STATUS.md).

The [selective state-commit pass](docs/SHoEbox_State_Commit_Report.md) adds optional
holds for producer contextualizer and recurrent contents using the existing
publication decision. All 512 checks pass. Eight matched 600-update arms show a
small mean standard-delay improvement with stored reads (selected exact-answer
NLL 1.2008 versus 1.2079), but the long-delay advantage reverses at the final
checkpoint. Shuffling the protected producer contents during queries has almost
no effect. State holding remains optional; candidate compute and parameter count
stay fixed. [PDF report](docs/SHoEbox_State_Commit_Report.pdf) |
[protocol](docs/SHoEbox_State_Commit_Protocol.md) |
[metrics](experiments/text_regions/results/task_state_commit/summary.json) |
[handoff](STATUS.md).

The [delayed payload-credit update](docs/SHoEbox_Payload_Credit_Report.md) adds an optional identity payload adapter with owner/version-bound historical credit. It adds 16 trainable coefficients and 768 bytes of credit state at batch eight. All 502 checks and GitHub CI pass. Six matched 600-update runs show mixed behavioral results: selected test NLL averages 1.2177 with payload credit versus 1.2079 with gate-only credit, and older-content interventions do not establish a consistent retention gain. The feature stays optional. [PDF report](docs/SHoEbox_Payload_Credit_Report.pdf) | [protocol](docs/SHoEbox_Payload_Credit_Protocol.md) | [metrics](experiments/text_regions/results/task_payload_credit/summary.json) | [handoff](STATUS.md).

The [acquisition-to-retention update](docs/SHoEbox_Acquisition_Report.md) adds a three-stage acquisition curriculum, 64-byte live gradient windows, a matched 16-byte truncation control, source-window coverage accounting, and exact resume across stage changes. The model and its legacy defaults are unchanged. [PDF report](docs/SHoEbox_Acquisition_Report.pdf) | [pilot metrics](experiments/text_regions/results/task_acquisition/pilot_summary.json).

The completed 549,059-parameter pilot reaches exact-answer NLL 1.183 with the curriculum versus 1.199 for trained fresh outputs. Older-publication content interventions remain near zero and change sign at the final checkpoint. All 497 checks pass. Two additional seed pairs became inaccessible after update 300 when the execution workspace went offline; their confirmation results are not claimed. The completed pilot's checkpoints and raw evidence were preserved before the outage. See the [protocol](docs/SHoEbox_Acquisition_Protocol.md) for reproduction and the next development target.

The [retention update](docs/SHoEbox_Task_Retention_Report.md) adds publication-aware write decisions, optional one-owner-or-hold competition, cross-owner delayed credit, and whole-query content interventions. A separately trained current-output control distinguishes useful stored publications from ongoing recurrent computation. The harder stream includes four changing entities and intervening updates that can become relevant later. All additions are optional; legacy defaults and checkpoints are preserved. See the [retention protocol](docs/SHoEbox_Task_Retention_Protocol.md) and [reproduction commands](experiments/text_regions/README.md#task-driven-memory-retention).

Older publications are now measurable and manipulable independently of query-time writes. In the scratch pilot, competitive exact-answer NLL is 1.356 versus 1.361 for trained current-output reads, and changing older content has negligible effect. The common-parent continuation reaches 1.276 with competitive storage versus 1.258 for trained current-output reads. Useful longer-lived storage remains open; reduced write rates and older read ages do not count as success. The report separates the two cohorts and preserves the selected and final checkpoints of every arm. That result motivated the acquisition curriculum; the detached delayed trace still updates write heads only.

The new [task-driven memory mode](docs/SHoEbox_Task_Memory_Protocol.md) connects the ordinary byte stream to four owned publications, contextual reads and a sparse regional composer. Its bounded eligibility traces train earlier write policies across detached chunks. The 548,195-parameter pilot uses no field-address input, fixed latent teacher, replay store or retention threshold. See the [task-memory results](docs/SHoEbox_Task_Memory_Report.md) for behavioral outcomes and remaining phase exits.

```bash
python -m experiments.text_regions.task_memory_train \
  --out artifacts/task_memory/integrated --steps 300 --batch 16 --final-eval
# Paired controls: add --arm no_credit or --arm no_memory with separate output paths.
# Continue exactly: add --resume artifacts/task_memory/integrated/last.pt --steps 600.
python -m experiments.text_regions.task_memory_evaluate --run artifacts/task_memory/integrated
```

Across three paired initialization/stream seeds, the new integrated model reaches **78.9% held-out answer accuracy versus 70.3% without memory**, with mean answer NLL **0.524 versus 0.644** at matched expert count. Delayed credit improves mean accuracy over credit-disabled training (75.5%), while selected-loss gains are positive in only one of three pairs. Useful publication reads are demonstrated; selective retrieval over a complete query and longer-lived useful commits remain next targets.

Earlier milestone:

**The joint-stability continuation is complete. All three capped runs preserve the 90% held-out memory target; one of three meets the full joint text-and-memory target. M7 remains open.** Three parent/data pairs each completed two matched 60-update continuations: ordinary task-gradient summation and separate caps for each task within the core, readout, and producer optimizer groups.

These are the validation-selected checkpoints, evaluated on 4,096 new text targets and 512 fresh episodes in each of nine memory conditions:

| Pair | Capped text nats/byte | Capped memory minimum | Sum text nats/byte | Sum memory minimum |
|---|---:|---:|---:|---:|
| 20261400 | 4.5080 | 91.21% | 4.4221 | 86.13% |
| 20261401 | 3.5864 | 95.70% | 3.5193 | 96.29% |
| 20261402 | 6.5627 | 96.29% | 9.0613 | 87.50% |

Memory minimum is the weakest of all 27 recall, correction, and withheld-correction field scores. The cap preserves retention in the two pairs where ordinary summation loses it. Text learning remains uneven. Pair 20261401's capped arm selects its unchanged starting parent; its ordinary-sum control selects update 40. The report preserves selected and fixed-budget final results separately.

The graph remains at **510,958 parameters and 768 physical experts**. Each continuation adds 7,680 ordered natural-text bytes and 960 fresh procedural memory episodes; each final lineage has seen 28,160 natural-text bytes during the joint-text phases. All six run audits pass, including paired-data reconstruction, retained-owner state checks, and exact 128-byte generation after saving at byte 41. All **416 regression tests pass**.

See the [joint-stability report](docs/SHoEbox_Joint_Stability_Report.md), [locked protocol](docs/SHoEbox_Joint_Stability_Protocol.md), [audited results](experiments/text_regions/results/joint_stability), and [reproduction commands](experiments/text_regions/README.md#joint-text-and-memory-gradient-stability). Those retention guards apply to the historical diagnostic; the new task-memory mode selects on validation task loss. The [preceding joint-text report](docs/SHoEbox_Joint_Text_Report.md) records the three starting lineages and the resumable byte interface.

The preceding [multi-fact phase](docs/SHoEbox_Multifact_Report.md) passed its 90% target across all three parent/data pairs: mean recall was **99.93% shared, 99.80% left-private and 99.87% right-private**. Its four independently owned producers feed a frozen two-round composer. That published result remains the starting point for the joint-text study. Earlier [joint retention](docs/SHoEbox_Joint_Retention_Report.md), [retained composition](docs/SHoEbox_Retained_Composition_Report.md), and [selective publication](docs/SHoEbox_Selective_Publication_Report.md) established the memory interface and learned write/hold behavior.

## Earlier experiments

The full-rank decoder follow-up is complete: **72 deterministic refits on twelve frozen graphs**, with the same 960/3,840 training cohorts and fixed evaluations. The refits do not consistently improve on the retained compact identity correction. A new compiled readout folds normalization and affine corrections into a single matrix multiplication; all exported predictions are independently checked. See the [full-rank and inference-cost report](docs/SHoEbox_Full_Rank_Readout_Report.md), [aggregate results](experiments/text_regions/results/full_rank), and [reproduction and deployment commands](experiments/text_regions/README.md#full-rank-refits-and-compiled-readouts).

The missing cohort-diversity phase has been rebuilt and rerun across **twelve frozen graphs and 432 corrections**. The [reproduction report](docs/SHoEbox_Cohort_Reproduction_Report.md) compares the fresh results with the [preserved original findings](docs/SHoEbox_Cohort_Diversity_Report.md). Source, resumable fitting, independent audits and aggregate results are included; the downloadable artifact package contains all fitted tensors, expanded features and dependencies. See the [reproduction commands](experiments/text_regions/README.md#decoder-cohort-diversity-reproduction).

**All 80 reported accuracy values reproduce at their published precision.** Nonzero identity corrections accepted by validation rise from 3/36 to 33/36 when the fitting cohort grows from 960 to 3,840 cases. All 307 automated tests pass; artifact audits reproduce 2,880 new prediction groups and 720 retained legacy groups.

The first implementation includes reusable latent nodes, typed publication records, functional graph state, fixed synchronous schedules, a runnable byte language model, ordered stream training, exact checkpoint continuation, whole-region rounds, input/edge interventions, and CPU measurements. Baseline parity and engine behavior are covered by automated tests.

The probe harness trains recall/correction across chunk boundaries, with optional independent shared/private view objectives and explicit post-optimizer edge/lace utility commits. Both learning additions are disabled by default. Probe checkpoints resume the complete model, optimizer, utility buffers, stream cursor, RNG, and graph state.

The 246K-parameter, seven-node model reached **98.6% correction accuracy at an 18-byte delay and 99.6% at 34 bytes**, on 512 fresh test episodes per difficulty with 16-byte training segments. Resetting state reduced accuracy to 26.6%; withholding the correction preserved the old value in 100% of episodes. This is the compact four-value query probe. All 269 automated checks pass.

The subsequent shared/private benchmark holds out combinations of three independent facts and varies source formats and field order. In a matched four-way comparison, view objectives improved unseen-combination accuracy from **69.8% to 79.2%**. Private left/right answers improved from **65.6%/63.5% to 78.6%/75.5%**. Every variant started with identical weights and trained on 96,000 external events with 8,064,000 expert calls. Missing-view recovery remains uneven; an optional requirement for evidence before publication exposes interference from cold nodes. See the [shared/private report](docs/SHoEbox_Shared_Private_Report.md) and the [benchmark commands](experiments/text_regions/README.md#factorized-sharedprivate-benchmark).

Three paired fine-tuning seeds now separate publication-policy training from inference. Evidence-gated training raises mean **B-only shared recovery from 35.1% to 85.1%**, while reducing all-view held-out accuracy from **78.2% to 68.2%**. Private-fact accuracy falls in all three pairs, so the gate remains default-off. Regional readouts find the facts downstream; query-only inputs from other sources can still interfere with the final answer. The [evidence-policy report](docs/SHoEbox_Evidence_Policy_Report.md) records the matched results, diagnostics, and motivation for the baseline-gradient ablation. [Raw predictions and the portable comparison manifest](experiments/text_regions/results/evidence_policy) accompany the report.

The subsequent baseline-gradient ablation detaches the source-only auxiliary head from the encoder while preserving its training. Across the three gated seeds, mean **held-out accuracy improves from 68.2% to 77.6%** and private accuracy from **57.9% to 73.7%**, while **B-only recovery drops from 85.1% to 67.3%**. Seed 1 also loses left-private accuracy, so blanket detachment remains default-off. This motivated retaining baseline gradients for observed facts while masking unsupported targets. See the [baseline-gradient report](docs/SHoEbox_Baseline_Gradient_Report.md) and [matched raw results](experiments/text_regions/results/baseline_gradient).

The source-observability ablation now excludes only baseline targets that A cannot observe, with the original loss denominator and utility accounting preserved. The three-seed gated means are **68.6% held-out overall, 61.0% private, and 70.3% B-only**, versus 68.2%, 57.9%, and 85.1% for the original baseline. Independent audits verify every retained/excluded target. This does not resolve the tradeoff; the option remains default-off. The following experiment balances supported source exposure at fixed expert budgets. See the [source-observability report](docs/SHoEbox_Source_Observability_Report.md) and [three-arm comparisons](experiments/text_regions/results/baseline_observed).

The balanced-source experiment rotates supported whole-source combinations while preserving facts, inputs, loss settings and expert budgets. Across three paired seeds, **B-only shared recovery rises from 35.1% to 99.8% with original publication**, and from **85.1% to 96.5% with evidence gating**. Held-out private accuracy falls from **75.6% to 50.1%** and **57.9% to 49.5%**, respectively. Balanced exposure remains optional. See the [balanced-source report](docs/SHoEbox_Balanced_Source_Report.md) and [six matched comparisons](experiments/text_regions/results/source_schedule).

The subsequent frozen-readout study covers all twelve models. A common 256-byte decoder fitted to full P/Q outputs raises original-policy/dropout **B-only accuracy from 34.4% to 85.7%**, versus **70.1%** when refitting the compressed representation. Held-out overall accuracy stays near **78%**, while private accuracy moves from **76.3% to 73.7%**. Gated/dropout models improve both private accuracy (**57.2% to 60.9%**) and B-only recovery (**84.9% to 95.8%**). Fixed factor probes find more private information in both representations than the common answer decoder extracts, motivating a shared nonlinear emission adapter next. These scores use one fresh diagnostic cohort with every graph frozen and K=1. See the [frozen-readout report](docs/SHoEbox_Frozen_Readout_Report.md) and [audited results](experiments/text_regions/results/readout_diagnostic).

The shared emission-adapter study fits **144 heads** across those twelve frozen graphs, with three initializations and parameter-matched identity/GELU controls. Raw GELU raises private accuracy from **56.0% to 62.8%** on original/balanced graphs and **54.9% to 62.8%** on gated/balanced graphs; matched linear controls reach **56.8% and 54.5%**. Original/dropout B-only recovery rises from **85.7% to 97.6%**. Shared-answer accuracy falls on the held-out all-source task, motivating a nonlinear residual around the existing raw linear head. See the [emission-adapter report](docs/SHoEbox_Emission_Adapter_Report.md) and [aggregate results](experiments/text_regions/results/emission_adapter).

The residual follow-up fits **72 corrections** with 16 hidden units and 5,392 trainable parameters each. Validation selects their strength under shared-answer accuracy guards, choosing zero for **22 of 36 GELU heads**. Selected GELU raises B-only recovery from **85.7% to 90.1%** on original/dropout graphs and **95.8% to 97.3%** on gated/dropout graphs. Private accuracy barely changes: scaling back the correction also removes its private gains. All **284 tests pass** and saved weights reproduce every result. The default readout remains unchanged; next is shared-answer preservation during correction training. See the [residual report](docs/SHoEbox_Residual_Adapter_Report.md) and [audited results](experiments/text_regions/results/residual_adapter).

The shared-preservation phase adds a training-only margin loss and fits **144 matched corrections**, with validation selecting penalty and scale. Selected GELU raises private accuracy from **56.0% to 57.7%** on original/balanced graphs and **55.2% to 55.9%** on gated/balanced graphs. Shared accuracy changes from **80.5% to 79.4%** and **87.4% to 87.5%**, respectively. Zero is still selected for **19 of 36 GELU heads**. Four longer-fit diagnostics improve training objectives but all select zero afterward. All **295 tests pass**, and saved tensors reproduce every prediction and validation choice. The default readout remains unchanged; next is greater fitting-cohort diversity. See the [shared-preservation report](docs/SHoEbox_Shared_Preservation_Report.md) and [audited results](experiments/text_regions/results/shared_preservation).

```bash
python -m experiments.text_regions.probe_train --kind correction --steps 300 \
  --batch 8 --segment 16 --gap 16 --eval-gap 32 --eval-episodes 512 \
  --eval-batch 64 --out artifacts/probes/correction_300
python -m experiments.text_regions.probe_train --kind shared_private \
  --view-objective --utility --steps 100 --out artifacts/probes/views
python -m experiments.text_regions.probe_eval artifacts/probes/correction_300/checkpoint.pt \
  --gaps 16 32 --split test --out artifacts/probes/correction_test.json
```

See the [probe and learning report](docs/SHoEbox_Probe_Development_Report.md) for recorded accuracy, controls, and remaining acceptance work. The [initial report](docs/SHoEbox_Development_Report.md) preserves the first engine milestone.

- [Experiment API, reproduction commands, and state contracts](experiments/text_regions/README.md)
- [Authoritative development specification](docs/SHoE_Text_Development_Specification.md)
- [Development report](docs/SHoEbox_Development_Report.md)
- [Probe and learning development report](docs/SHoEbox_Probe_Development_Report.md)
- [Shared/private benchmark and publication diagnostics](docs/SHoEbox_Shared_Private_Report.md)
- [Three-seed evidence-policy comparison and regional diagnostics](docs/SHoEbox_Evidence_Policy_Report.md)
- [Baseline encoder-gradient ablation](docs/SHoEbox_Baseline_Gradient_Report.md)
- [Source-observed baseline supervision and target audits](docs/SHoEbox_Source_Observability_Report.md)
- [Balanced source exposure and paired publication-policy results](docs/SHoEbox_Balanced_Source_Report.md)
- [Frozen shared byte decoders and temporal factor probes](docs/SHoEbox_Frozen_Readout_Report.md)
- [Shared emission adapters with matched linear controls](docs/SHoEbox_Emission_Adapter_Report.md)
- [Residual byte decoders with validation-selected strength](docs/SHoEbox_Residual_Adapter_Report.md)
- [Shared-answer preservation during residual training](docs/SHoEbox_Shared_Preservation_Report.md)
- [Decoder cohort-diversity findings](docs/SHoEbox_Cohort_Diversity_Report.md)
- [Reproduced cohort study, 432 fresh fits and original-result comparison](docs/SHoEbox_Cohort_Reproduction_Report.md)
- [Recovery status, available artifacts, and reproduction](docs/SHoEbox_Recovery_Report.md)
- [Original SHoE overview and results](SHoE_BASELINE.md)
- [Preserved HAT/minKron baseline](experiments/hat_minkron)
- [Source lineage and third-party notices](THIRD_PARTY_NOTICES.md)

Checkpoints are generated locally and excluded from git. Source, protocols, measurements, and samples are committed. `SOURCE_MANIFEST.json` retains the original SHoE hashes; compatibility extensions to its HAT experiment preserve ordinary behavior and are checked for parity in the new suite.
