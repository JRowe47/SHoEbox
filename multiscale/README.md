# Multiscale SHoE — selected routing D

`shoe_multiscale.Model()` and `fresh_model()` now create the tested **D** profile.
D uses exact prototype search, centered normalized routing, fading utilization
warmup, and stable predictive SDR cues. The Python API and CLI select the same
policy. The underlying native arithmetic is identical to the frozen D control.

## Quick start

Requirements: Python 3.10+, NumPy, a C++17 compiler, and a CPU. The first API use
compiles the native library into a temporary cache. Set `CXX` to select a compiler
or `SHOE_MULTISCALE_CACHE` to select a cache directory.

```python
from shoe_multiscale import Model, fresh_model

with fresh_model(levels=4, parameters=1_000_000, seed=7) as model:
    assert model.stats()["routing_arm"] == "D"
    model.train(b"An observed training sequence.\n")
    model.save("model.shoe")  # unfinished local blocks remain pending
    loss = model.evaluate(b"A separate held-out sequence.\n")
    print(model.generate("An ", length=256).decode("utf-8", "replace"))

with Model.load("model.shoe") as resumed:
    print(resumed.stats())
```

```bash
python -m multiscale.cli train --data input.txt --targets 1000000 --out model.shoe
python -m multiscale.cli inspect --checkpoint model.shoe
python -m multiscale.cli generate --checkpoint model.shoe --prompt 'An '
python -m multiscale.cli train --resume model.shoe --data next.txt --out next.shoe
```

Resume preserves the saved routing, objectives, stream state, and pending credit.
Supply the next input/target segment; use `Model.run(x, y)` to retain next-byte
pairs across API boundaries. Routing overrides are rejected with CLI resume.
The simple CLI applies fixed learning rates and does not reserve validation data;
it does not reproduce the FineWeb schedule by itself.

## D architecture and selected settings

| Scale | Bytes/event | Experts at the 1M budget |
| --- | ---: | ---: |
| Fast | 1 | 98 |
| L1 | 4 | 49 |
| L2 | 16 | 29 |
| L3 | 64 | 20 |

Width is 64, with one selected expert per event and four ordered children per
slow event. D has **1,011,903 fitted parameters** at the nominal 1M expert-budget
setting: the three 64-by-64 SDR prediction heads add 12,288 parameters above the
999,615-parameter base allocation. `parameter_count` reports the actual total.
The extra router-depth experiments are not part of D.

The learner includes context-dependent fast experts, separate fast/slow/query
learning rates, credit through retained constituents, addressed recurrent memory,
and retained-unit recall queried at each byte. Stable content representations
remain separate from context. D adds fixed ordered signed 16-of-64 content cues
and detached prior-state context to slow routing/expert inputs, plus a local
next-same-scale SDR prediction objective. The target is supplied only after the
next observed event. The byte decoder, contrastive-content and reconstruction
objectives remain active.

Centered normalized routing and the calibrated margin gate run at all scales.
Utilization warmup fades after every expert has 256 selections; subsequent
pressure only nudges starvation. Statistics and prototype seeding observe
training data only. Frozen evaluation learns neither weights nor routing means.

Default rates are fast/query 0.008, slow 0.08, head 0.16, and prototype attraction
0.002. Use `configure(fast_rate=..., slow_rate=..., query_rate=..., head_rate=...)`
for explicit schedules; omitted rates retain their current values.

## Historical policies

- `Model(routing_arm="A")` selects the former indexed routing and semantic-lace
  defaults. For the winning shuffled A comparator, explicitly use
  `router="scan", semantic_laces=False` and its recorded data order.
- `routing_arm="B"` and `"C"` keep their experimental definitions, exact search,
  and disabled semantic-lace dispatch. `semantic_laces=True` selects A.
- `memory_profile="legacy"`, `"addressed_v2"`, `"addressed_v3"`, and
  `"addressed_v4"` use their preserved native implementations.
- `Model.load()` dispatches versions 1–4 to those original backends. Versions
  5–9 retain their saved learning/routing policies in the current native backend;
  v8 C/D keeps the original continuing-balance pilot, v9 keeps fading warmup.
- `shoebox_reference` remains the independent historical cortical comparator.

## Evidence and verification

Three matched 15M-target FineWeb runs have mean frozen development loss
**1.772838555 nats/byte**. Previously measured serial median training throughput
is **30,670 bytes/s** on the recorded host. These are the selected experiment's
measurements, not a new training run during promotion.

[Selection, exact protocol and provenance](ROUTING_D.md) ·
[Promotion verification](results/routing_d_promotion_verification.json) ·
[Selected measurements](results/routing_d_selection.json)

```bash
python -m multiscale.tests.run_all
# Ptrace-managed hosts only: retain ASan/UBSan, explicitly disable LSan.
python -m multiscale.tests.run_all --no-leak-check
```

The suite covers native derivatives, memory/recall, semantic-lace compatibility,
D routing/forecast credit, frozen-state protection, public defaults, CLI/API
parity, and exact checkpoint continuation. GitHub CI runs it on the CPU.
Datasets, trained checkpoints, generated binaries, and temporary snapshots stay
outside git.

Earlier architecture and measurements are retained as historical records:
[Addressed v2 architecture](ADDRESSED_V2_ARCHITECTURE.md),
[addressed-memory results](ADDRESSED_MEMORY_REPORT.md),
[original rebuild results](REBUILD_REPORT.md), and [design](DESIGN.md).
