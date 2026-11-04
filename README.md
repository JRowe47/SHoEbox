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

The 246K-parameter, seven-node model reached **98.6% correction accuracy at an 18-byte delay and 99.6% at 34 bytes**, on 512 fresh test episodes per difficulty with 16-byte training segments. Resetting state reduced accuracy to 26.6%; withholding the correction preserved the old value in 100% of episodes. This is the compact four-value query probe. All 168 automated checks pass.

The subsequent shared/private benchmark holds out combinations of three independent facts and varies source formats and field order. In a matched four-way comparison, view objectives improved unseen-combination accuracy from **69.8% to 79.2%**. Private left/right answers improved from **65.6%/63.5% to 78.6%/75.5%**. Every variant started with identical weights and trained on 96,000 external events with 8,064,000 expert calls. Missing-view recovery remains uneven; an optional requirement for evidence before publication exposes interference from cold nodes. See the [shared/private report](docs/SHoEbox_Shared_Private_Report.md) and the [benchmark commands](experiments/text_regions/README.md#factorized-sharedprivate-benchmark).

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
- [Original SHoE overview and results](SHoE_BASELINE.md)
- [Preserved HAT/minKron baseline](experiments/hat_minkron)
- [Source lineage and third-party notices](THIRD_PARTY_NOTICES.md)

Checkpoints are generated locally and excluded from git. Source, protocols, measurements, and samples are committed. `SOURCE_MANIFEST.json` retains the original SHoE hashes; compatibility extensions to its HAT experiment preserve ordinary behavior and are checked for parity in the new suite.
