# Route-local learning rates

The controller is implemented in the selected corrected-C runtime. It attaches
learning-rate cycles to `(level, expert)` identities and changes every rate
through one global anchor. It adds no fitted parameters, target-based router,
extra gradient pass or hot-loop monitoring.

## Rule

Let `b` be a parameter family's existing base rate, `L` and `U` the multiplier
bounds, `A(t)` the anchor, `P` the cycle period, `q` the amplitude in `[0,1]`,
and `n[l,e]` the selected training events already retired for an expert.

```text
radius(t) = q * min(A(t) - L, U - A(t))
factor(l,e,t) = A(t) + radius(t) * sin(2*pi*(n[l,e] mod P)/P)
expert_rate = b * clamp(factor, L, U)
shared_rate = b * A(t)
```

The anchor is the midpoint of the active oscillation. Moving it toward a hard
bound squeezes the symmetric oscillation; moving it toward the middle relaxes
it. `amplitude=0` collapses the cycle onto the anchor. `lower=upper=anchor`
also produces a constant multiplier. Zero is a valid lower bound.

Defaults on explicit enable are `lower=.5`, `upper=1.5`, `anchor=1`,
`amplitude=1`, and `period=256` selected events per expert. A first visit starts
at the anchor. After a quarter cycle the rate is highest, after half a cycle
it returns to the anchor, and after three quarters it is lowest. Frequently
selected routes advance faster in byte time; rare routes keep their phase.
Cycles are attached to route identity/use, without claiming that expert IDs
are supervised semantic labels.

The bounds are **multipliers**, preserving distinct fast, slow, query, head
and prototype base rates. For example, a slow rate of `0.002` with the default
controller ranges from `0.001` to `0.003`. Objective weights, credit gains,
gradient clipping, native execution periods and routing coverage statistics
are independent of this optimizer schedule.

## Update ownership

| Parameter | Multiplier |
| --- | --- |
| Fast expert matrix and bias | Selected fast expert's cycle |
| L1/L2/L3 expert matrix and bias | Selected expert's cycle at constituent retirement |
| Routing key language gradient | Owning key's expert cycle, including runner-up keys |
| Prototype attraction | Selected expert's cycle |
| Recurrence, composition, memory maps and prediction heads | Anchor |
| Shared byte embeddings, decoder and recall query/readout | Anchor |

Scaling occurs at the parameter step, after ordinary gradient aggregation and
clipping. It does not scale the derivative sent to a child or change a local
objective's relative weight. Shared parameters have multiple contributing
routes and therefore use the common anchor, not an arbitrarily chosen route.

Fast clocks advance after each eight-event fast update; upper clocks advance
after their bounded constituent epoch retires. Every selected event contributes
one clock tick, even when several events share an aggregated update. All
updates in that retirement use the clock value before advancement. Runner-up
key gradients do not count as selected events. No extra routes are evaluated.

## API and lifecycle

```python
from shoe_multiscale import Model

model = Model(parameters=5_000_000, route_lr={"period": 256})
model.configure(fast_rate=.0002, slow_rate=.002, query_rate=.0002,
                head_rate=.004, route_rate=.002)
model.train(training_bytes)

# Move the anchor now, preserving every route's phase.
model.set_learning_anchor(1.1)
# Or move linearly across the next million observed training targets.
model.set_learning_anchor(.8, over_targets=1_000_000)

# Inspection is read-only and outside the training loop.
print(model.route_lr_stats(include_routes=True))
model.save("model.shoe")

# Explicitly configure an existing checkpoint after retiring pending credit.
resumed = Model.load("old_corrected_C.shoe")
resumed.finish()
resumed.configure_route_lr(lower=.5, upper=1.5, anchor=1, period=256)
```

An anchor ramp starts at the current anchor and reaches its endpoint after
exactly `over_targets` additional training targets. Retargeting starts from
the current interpolated value. The first target uses the current value;
the last target's completion reaches the endpoint for subsequent updates.
Delayed gradients use the anchor at the time their parameter update retires.

Context resets retire pending gradients normally, but do not clear route
clocks or reset a ramp. Evaluation and inference never advance these clocks.
`configure_route_lr(enabled=False, ...)` pauses the controller and uses the
unscaled base rates; re-enabling retains visit counts. Full reconfiguration
requires a fresh/finished training boundary. Anchor changes can be made while
credit is pending, consistent with the existing mutable base-rate API.

The CLI preserves the entire controller on ordinary resume. Explicit
`--route-lr` or `--no-route-lr` on resume first retires pending credit; omitted
controller fields retain their saved values. `--lr-anchor` changes the anchor
immediately; `--lr-anchor-target` and `--lr-anchor-targets` create a ramp.

Checkpoint v17 saves bounds, amplitude, period, ramp endpoints/progress,
training-target count and all 64-bit expert clocks, alongside existing credit
and routing state. Models that never configure the controller retain their
previous checkpoint version and arithmetic. Private experimental v13–v16
formats are not reused. Historical native profiles are unchanged. Dynamic
semantic-lace cloning is rejected with this controller because those experts
do not yet have an optimizer-state migration policy; corrected C has stable
expert identities and supports it directly.

At the nominal 5M size, the four banks contain 1,143 experts: visit clocks use
9,144 bytes, plus a small global controller. The oscillator executes at sparse
parameter updates. No evaluation-loss feedback is used by the controller.

## Verification and comparison

```bash
python -m multiscale.tests.run_all
# For a ptrace-managed host only:
python -m multiscale.tests.run_all --no-leak-check

python experiments/route_learning/compare.py \
  --checkpoint SHoEbox_Corrected_C_5M_20M.shoe \
  --data /path/to/fineweb --out /path/to/results --targets 5000000 --cpu 2
```

Tests isolate sparse parameter updates from their gradients; verify constant
anchor scaling across every rate family; check neutral-policy equality, hard
bounds, sparse clocks, frozen evaluation, interrupted ramps, malformed inputs,
full-width counters, old policies and byte-identical checkpoint continuation.
Native checks run under AddressSanitizer and UndefinedBehaviorSanitizer.

The comparison starts both arms from the same seed-7 corrected-C checkpoint
after 20M targets. Both replay the same reshuffled training pool for another
5M targets, with document-contained chunks and the same held-out development
documents. C keeps its recorded rates; the route arm adds default local cycles
at anchor 1. Arms run serially in alternating 1M-target blocks on one pinned
CPU. This isolates the local cycle at a fixed anchor; anchor movement is
covered by lifecycle and rate-scaling checks. Results are a single-checkpoint
continuation screen, not a fresh multi-seed training comparison.
