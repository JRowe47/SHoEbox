# SHoEbox route-local learning rates

Implemented on corrected C · 2026-11-15

## What is implemented

Each `(level, expert)` route has its own bounded sine cycle. Its clock advances
with selected training events, so frequently used routes cycle faster in byte
time and rare routes retain their phase. A shared anchor shifts every rate.
Moving the anchor toward a bound squeezes the oscillation; moving it toward
the midpoint relaxes it.

Default enabled settings: **0.5–1.5× bounds, 1.0× anchor, 256 selected events
per cycle, full amplitude.** Bounds multiply each parameter family's base rate,
preserving the existing fast/slow/query/head/prototype ratios. Shared weights
use the anchor; expert matrices, biases and routing keys use the local cycle.
Gradients and credit passed to other levels remain unchanged.

```python
from shoe_multiscale import Model

model = Model(parameters=5_000_000, route_lr={
    "lower": .5, "upper": 1.5, "anchor": 1., "period": 256,
})
model.train(training_bytes)
model.set_learning_anchor(.8, over_targets=1_000_000)
model.train(next_training_bytes)
model.save("model.shoe")
```

For an existing checkpoint, call `finish()` and then `configure_route_lr(...)`.
`set_learning_anchor(value)` moves the anchor immediately; `over_targets=N`
creates a linear ramp. Both preserve every route's phase. Inspection is
available through `route_lr_stats(include_routes=True)`.

## Matched continuation result

Both arms started from the same **4,999,743-parameter corrected-C checkpoint
trained on 20M targets**, then consumed another **5M targets** in identical order.
This was a reshuffled replay of its FineWeb training pool, shuffle seed 1007;
development documents stayed separate. No hyperparameter search or audit-based
selection was performed. The cycle arm kept the anchor at 1 to isolate the
route-local behavior; anchor movement was checked numerically and through
interrupted training/resumption.

| Result after 25M total targets | Corrected C | Route cycles |
| --- | ---: | ---: |
| Development NLL, nats/byte | 1.722843 | 1.723436 |
| Training bytes/second | 31,642 | 31,238 |

**The first cycle settings did not improve loss:** route cycles finished
+0.000594 nats/byte above C. The implementation is available explicitly;
the established C optimizer settings remain the default.

The separate serial timing check used three alternating pairs of 200,000 targets on one pinned CPU, including resets and final gradient retirement. Route cycles changed throughput by -1.28%.

The common starting development NLL was 1.730870. Evaluation
uses 262,144 targets across 70 documents. The attached PNG shows the controller
and both measured loss curves; CSV and full JSON measurements accompany the code.

## Verification

- Native parameter-step checks confirm the expected per-expert update ratios,
  shared-anchor scaling across all rate families, and unchanged gradient credit.
- Hard bounds, sparse clocks, retargeted ramps, zero amplitude, pause/resume,
  frozen evaluation, malformed checkpoints and full-width 64-bit clocks pass.
- **37 Python tests pass.** The native suite and route-update checks pass
  AddressSanitizer and UndefinedBehaviorSanitizer; LeakSanitizer was explicitly
  disabled on the ptrace-managed host.
- A real v10 checkpoint and its 4,097-target continuation are byte-identical
  between the previous runtime and the unconfigured new runtime.
- Partial-block v17 checkpoints resume exactly with moving anchors and
  language/recall credit enabled. Context resets do not reset optimizer clocks.

The controller adds 9,144 bytes of expert counters at the 5M size, plus a small
global state; it adds no fitted parameters. Checkpoint v17 persists the full
controller and pending learning state. Old checkpoints keep their saved policy.

