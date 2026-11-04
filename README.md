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

The learning acceptance work comes next: delayed recall/correction beyond the training segment, shared/private view objectives, explicit inter-node utility commits, and multi-seed compositional comparisons. See the [development report](docs/SHoEbox_Development_Report.md) for the exact milestone status and measured results.

- [Experiment API, reproduction commands, and state contracts](experiments/text_regions/README.md)
- [Authoritative development specification](docs/SHoE_Text_Development_Specification.md)
- [Development report](docs/SHoEbox_Development_Report.md)
- [Original SHoE overview and results](SHoE_BASELINE.md)
- [Preserved HAT/minKron baseline](experiments/hat_minkron)
- [Source lineage and third-party notices](THIRD_PARTY_NOTICES.md)

Checkpoints are generated locally and excluded from git. Source, protocols, measurements, and samples are committed. `SOURCE_MANIFEST.json` retains the original SHoE hashes; the four compatibility extensions to its HAT experiment are documented and checked for parity in the new suite.
