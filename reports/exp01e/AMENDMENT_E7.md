# EXP01-E7: align local applicability with the observed answer request

Registered after E6's two development pilots, before E7 fitting. Both E6 arms
reached 50% initial binding exactness and 0% after generic language continuation.
E6 local gate weights exactly equal its query recognizer; it learned the supplied
label. The measured answer prefix is padding followed by the observed ANSWER
marker, with the query earlier in the stream. E6 labels that current window
negative, so its gate is 0.0000676 at answers. Gate-on recovers 99.41% exactness.
This is an applicability-target mismatch, not evidence that paired values were
lost. E6 and its checkpoint policy remain unchanged.

E7's sixth local target is whether the current, already observed byte is the
answer-request marker. It is constructed directly from the observed prefix,
without the task answer mask, future target, desired value or source index.
The classifier sees the same continuous existing two-byte stable feature.
No byte tests or parser enter inference. As in E6, generic training freezes
this fixture-supervised gate and local projections while learning the core.
Keep the existing normalized presence SGD rate/weight; no sweep.

Fresh seed-7 pilot: 250K warmup +2M binding +1M pinned language targets. Same
E2-family grouped train/dev folds, order and all development gates. Reuse E6's
already charged generic-gate control (same data/budget/initialization); verify
that the added ignored label leaves that control's numerical trajectory exact.
Charge 3.25M additional bytes; retain all 6.5M E6 pilot bytes and failures.

If development passes, lock a fresh three-seed formal comparison before the
still-unopened E2-family audit. Compare native v19, frozen D3, generic gate,
local gate and no-local-label control, with matched 250K+2M budgets; run all
binding gates before and after 1M generic targets. Generic production audits
must use fresh groups excluding the earlier language/copy/reconstruction audit.
Then 1M/5M capacity checks and isolated >=80% inference qualification.
All existing thresholds, ablations, causal/checkpoint/sanitizer requirements,
parameter/state accounting and direct-main publication authorization remain.
