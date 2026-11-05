# SHoEbox

**Seven HAT-SHoE nodes, three state-owning regions, and recurrent exchange over persistent tensor ports.** SHoEbox builds on [Sparse Hierarchy of Experts](https://github.com/JRowe47/Sparse-Hierarchy-of-Experts) at revision `b4bb7cfea18c3833819f6ed08973fd014f15f107`, preserving its history, experiments, and recorded results.

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

## Topology

| Region | Owned nodes | Forward inputs |
|---|---|---|
| R0 | A, B, C | Three causal branches of one byte stream, or explicitly masked independent views |
| R1 | U, P | U receives A+B; P receives U+A |
| R2 | V, Q | V receives B+C; Q receives V+C |

P and Q feed the shared byte readout. B has one state owner and publishes to both downstream regions. Delayed P/Q lateral messages and descending messages are read from a snapshot taken before the round starts. Forward edges read completed stages of the current round. The topology is configuration-driven; the seven-node layout is the default, not a runtime constraint.

## Development status

**Retained facts now drive live two-step composition in one eight-node graph.** The previously trained producer joins the seven-node recurrent model through an ordinary latent edge. Across three seeds, held-out accuracy is **99.54% for recall, 98.31% after correction, and 99.35% when a correction is withheld**. Every seed exceeds the 90% target on all three conditions.

F keeps the fact while B/C supply new relation tables. The existing regional experts compute `g(f(x))`; the retained fact is never decoded into an intermediate input byte. Four successive queries leave F's contextual state, recurrent state and retained publication unchanged. Only the new edge's 1,024 weights are fitted in this component-integration phase. The combined model has 317,098 parameters and 480 physical experts.

The independent source-ownership option defaults off. The original seven-node checkpoint still produces exactly the same logits, states and execution counts, including **100% two-round composition on all 592 held-out cases**. The next step is joint local refinement of the weakest producer/connection pair. M7 remains open pending the broader integrated capability suite.

See the [retained-composition report](docs/SHoEbox_Retained_Composition_Report.md), [locked protocol](docs/SHoEbox_Retained_Composition_Protocol.md), and [audited results](experiments/text_regions/results/retained_composition).

The preceding [selective-publication phase](docs/SHoEbox_Selective_Publication_Report.md) established the isolated producer: learned publication reached 99.09% recall at four times the training delay versus 68.55% with unconditional writes. Two of three seeds met its stricter write-precision/recall target. The [gradient-horizon report](docs/SHoEbox_Gradient_Horizon_Report.md) records the unchanged composition parent.

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
