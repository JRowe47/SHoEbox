# Multiscale SHoE — selected corrected routing C

`shoe_multiscale.Model()` and `fresh_model()` create **corrected C**.
C uses exact prototype search, centered normalized routing, and coverage
warmup. The Python API and CLI select the same
policy. New models retain coverage warmup and retire its bias; the former
post-warmup absence penalty is removed. Saved policies remain reproducible.

## Quick start

Optional completed-constituent refinement is available with `Model(loops={})`:
extra pass budgets 0/1/2/2 at the 1/4/16/64-byte scales, shared expert weights,
64-value working state and convergence stopping. See
[selective multiscale loops](SELECTIVE_LOOPS.md) for training, inference and
checkpoint behavior. The ordinary corrected-C control remains reproducible.

Requirements: Python 3.10+, NumPy, a C++17 compiler, and a CPU. The first API use
compiles the native library into a temporary cache. Set `CXX` to select a compiler
or `SHOE_MULTISCALE_CACHE` to select a cache directory.

```python
from shoe_multiscale import Model, fresh_model

with fresh_model(levels=4, parameters=1_000_000, seed=7) as model:
    assert model.stats()["routing_arm"] == "C"
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
The simple CLI does not reserve validation data or reproduce the FineWeb schedule
by itself. It uses fixed rates unless route learning or an anchor ramp is selected.

## Route-local learning rates

Use `Model(route_lr={})` or `model.configure_route_lr()` to enable bounded cycles
around a shared, movable anchor. Each expert's phase advances with its own
selected training events. All parameter families inherit the anchor; expert
matrices, biases and routing keys also inherit their route's periodic multiplier.
Existing base-rate ratios and loss weights are preserved.

```python
with Model(parameters=5_000_000, route_lr={
    "lower": .5, "upper": 1.5, "anchor": 1., "period": 256,
}) as model:
    model.train(training_bytes)
    model.set_learning_anchor(.8, over_targets=1_000_000)
    model.train(next_training_bytes)
    model.save("route_model.shoe")
```

```bash
python -m multiscale.cli train --data input.txt --out route_model.shoe \
  --route-lr --lr-lower .5 --lr-upper 1.5 --lr-anchor 1 --lr-period 256
python -m multiscale.cli train --resume route_model.shoe --data next.txt \
  --out continued.shoe --lr-anchor-target .8 --lr-anchor-targets 1000000
```

Clocks and anchor ramps survive document resets and exact checkpoint resumption.
Frozen evaluation does not advance them. Full details, update ownership,
compatibility and comparison commands are in [ROUTE_LEARNING.md](ROUTE_LEARNING.md).

## C architecture and selected settings

| Scale | Bytes/event | Experts at the 1M budget |
| --- | ---: | ---: |
| Fast | 1 | 98 |
| L1 | 4 | 49 |
| L2 | 16 | 29 |
| L3 | 64 | 20 |

Width is 64, with one selected expert per event and four ordered children per
slow event. C has **999,615 fitted parameters** at the nominal 1M budget and
**4,999,743** at 5M (expert banks 571 / 285 / 171 / 116).
Explicit D adds three 64-by-64 SDR prediction heads, or 12,288 parameters.
`parameter_count` reports the actual total.

The current allocator accepts nominal budgets through 100,000,000 parameters.
The nominal 100M C configuration has 99,997,503 fitted parameters and expert
banks 11,816 / 5,908 / 3,544 / 2,365; width remains 64. Large exact-search banks
use a derived tiled prototype cache with AVX-512/AVX2 support and a portable
fallback. Every expert is still scored, with the original float accumulation
order and tie-breaking. The cache is rebuilt on load, is not a fitted parameter,
and does not change the checkpoint format or learning rules.

The learner includes context-dependent fast experts, separate fast/slow/query
learning rates, credit through retained constituents, addressed recurrent memory,
and retained-unit recall queried at each byte. Stable content representations
remain separate from context. Explicit D adds fixed ordered signed 16-of-64 content cues
and detached prior-state context to slow routing/expert inputs, plus a local
next-same-scale SDR prediction objective. The target is supplied only after the
next observed event. The byte decoder, contrastive-content and reconstruction
objectives remain active.

Centered normalized routing and the calibrated margin gate run at all scales.
Utilization warmup fades as the least-used expert approaches 256 selections.
After coverage, its accumulated bias decays on a 256-selections-per-expert
horizon. No continuing penalty pushes frequent contexts away from specialists. Statistics and prototype seeding observe
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
  5–10 retain their saved learning/routing policies in the current native backend;
  v8 C/D keeps the uniform-balance pilot, v9 keeps the residual absence rule,
  and v10 uses coverage warmup only.
- v11 stores explicit language-credit gains, and v12 additionally stores bounded
  retained-value producer credit and its pending gradients. Unit language gains
  and disabled retained-value credit preserve the original checkpoint bytes.
- v17 additionally stores an explicitly configured route-learning controller,
  its anchor ramp and per-expert clocks. Old checkpoints retain fixed rates
  until route learning is explicitly configured; private v13–v16 experiments
  are not accepted as v17.
- Explicit `Model(routing_arm="C")` / `"D"` retain v9 for historical comparisons.
  Add `balance_policy="warmup"` to select v10 explicitly. `Model()` selects
  C with v10. `set_balance_policy("warmup")` is an explicit migration of future
  updates; it preserves current biases, weights and pending stream state.
- `shoebox_reference` remains the independent historical cortical comparator.

## Evidence and verification

Corrected C is the selected baseline for the L2/L3 investigation. Three matched
20M-target runs at 5M reproduce mean development NLL **1.731222** and audit NLL
**1.688454**. C is the requested baseline; the historical D selection below is
not an assertion that D remains selected. See the
[L2/L3 investigation](../experiments/upper_credit/README.md).

Language gains and retained-value producer gradients are opt-in experimental
controls. `language_credit=(1,1,4,16)` scales only the language VJP entering native
states; output-head and auxiliary gradients are unchanged. `recall_credit=(0,0,1,1)`
trains learned retained values through their live producing constituents.
Producer credit ends at the bounded constituent epoch; older values remain
available for recall. Defaults stay `(1,1,1,1)` and `(0,0,0,0)`. These controls
are numerically verified, but are not a demonstrated specialization repair.

The 5M diagnosis selects v10 coverage-only balancing: three matched 20M-target
runs improve mean development NLL from **1.752425 to 1.728077** and separate
audit NLL from **1.712747 to 1.687145**. Corrected D retains its predictive cues;
corrected C remains an approximately 9% faster comparator. Upper expert
specialization remains weak. See the [experiment protocol](../experiments/routing_5m/README.md)
and [measurements](results/routing_5m_selection.json).

The historical v9 D selection used three matched 15M-target FineWeb runs with
mean frozen development loss
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
