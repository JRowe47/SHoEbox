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

The first implementation includes reusable latent nodes, typed publication records, functional graph state, fixed synchronous schedules, a runnable byte language model, ordered stream training, exact checkpoint continuation, whole-region rounds, input/edge interventions, and CPU measurements. Baseline parity and engine behavior are covered by automated tests.

The probe harness trains recall/correction across chunk boundaries, with optional independent shared/private view objectives and explicit post-optimizer edge/lace utility commits. Both learning additions are disabled by default. Probe checkpoints resume the complete model, optimizer, utility buffers, stream cursor, RNG, and graph state.

The 246K-parameter, seven-node model reached **98.6% correction accuracy at an 18-byte delay and 99.6% at 34 bytes**, on 512 fresh test episodes per difficulty with 16-byte training segments. Resetting state reduced accuracy to 26.6%; withholding the correction preserved the old value in 100% of episodes. This is the compact four-value query probe. All 245 automated checks pass.

The subsequent shared/private benchmark holds out combinations of three independent facts and varies source formats and field order. In a matched four-way comparison, view objectives improved unseen-combination accuracy from **69.8% to 79.2%**. Private left/right answers improved from **65.6%/63.5% to 78.6%/75.5%**. Every variant started with identical weights and trained on 96,000 external events with 8,064,000 expert calls. Missing-view recovery remains uneven; an optional requirement for evidence before publication exposes interference from cold nodes. See the [shared/private report](docs/SHoEbox_Shared_Private_Report.md) and the [benchmark commands](experiments/text_regions/README.md#factorized-sharedprivate-benchmark).

Three paired fine-tuning seeds now separate publication-policy training from inference. Evidence-gated training raises mean **B-only shared recovery from 35.1% to 85.1%**, while reducing all-view held-out accuracy from **78.2% to 68.2%**. Private-fact accuracy falls in all three pairs, so the gate remains default-off. Regional readouts find the facts downstream; query-only inputs from other sources can still interfere with the final answer. The [evidence-policy report](docs/SHoEbox_Evidence_Policy_Report.md) records the matched results, diagnostics, and motivation for the baseline-gradient ablation. [Raw predictions and the portable comparison manifest](experiments/text_regions/results/evidence_policy) accompany the report.

The subsequent baseline-gradient ablation detaches the source-only auxiliary head from the encoder while preserving its training. Across the three gated seeds, mean **held-out accuracy improves from 68.2% to 77.6%** and private accuracy from **57.9% to 73.7%**, while **B-only recovery drops from 85.1% to 67.3%**. Seed 1 also loses left-private accuracy, so blanket detachment remains default-off. This motivated retaining baseline gradients for observed facts while masking unsupported targets. See the [baseline-gradient report](docs/SHoEbox_Baseline_Gradient_Report.md) and [matched raw results](experiments/text_regions/results/baseline_gradient).

The source-observability ablation now excludes only baseline targets that A cannot observe, with the original loss denominator and utility accounting preserved. The three-seed gated means are **68.6% held-out overall, 61.0% private, and 70.3% B-only**, versus 68.2%, 57.9%, and 85.1% for the original baseline. Independent audits verify every retained/excluded target. This does not resolve the tradeoff; the option remains default-off. The following experiment balances supported source exposure at fixed expert budgets. See the [source-observability report](docs/SHoEbox_Source_Observability_Report.md) and [three-arm comparisons](experiments/text_regions/results/baseline_observed).

The balanced-source experiment rotates supported whole-source combinations while preserving facts, inputs, loss settings and expert budgets. Across three paired seeds, **B-only shared recovery rises from 35.1% to 99.8% with original publication**, and from **85.1% to 96.5% with evidence gating**. Held-out private accuracy falls from **75.6% to 50.1%** and **57.9% to 49.5%**, respectively. Balanced exposure remains optional. The next diagnostic freezes the experts and compares one common answer decoder on P/Q's full outputs versus the existing compressed readout. See the [balanced-source report](docs/SHoEbox_Balanced_Source_Report.md) and [six matched comparisons](experiments/text_regions/results/source_schedule).

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
- [Original SHoE overview and results](SHoE_BASELINE.md)
- [Preserved HAT/minKron baseline](experiments/hat_minkron)
- [Source lineage and third-party notices](THIRD_PARTY_NOTICES.md)

Checkpoints are generated locally and excluded from git. Source, protocols, measurements, and samples are committed. `SOURCE_MANIFEST.json` retains the original SHoE hashes; compatibility extensions to its HAT experiment preserve ordinary behavior and are checked for parity in the new suite.
