# A2 pre-fit amendment: remove the binary overwrite shortcut

Registered before any A2 fitting. The requested shortcut audit of A1 exposes
an evaluation defect: two initial bindings have distinct binary values, and
flipping one makes both requested values equal. A query-independent predictor
that repeats the latest written value is therefore 100% whole-episode exact.
A1's 39.0625% native overwrite score and failed original gate remain unchanged,
but that fixture cannot qualify selective updates even if a later model passes.
This discovery does not affect the diverse-filler or competing-binding results.

For A2, use three possible values (0,1,2), two distinct initial values, and write
the third value to the changed key. The untouched key retains a different value.
Balance updated key, record order and query order; exclude no-op writes. Extend
local byte reconstruction to the observed third value without adding weights or
changing target scale, learning rates, gates or byte vocabulary. Use the same
three-value initial-state vocabulary across the matched E8/A1/A2 training arms.
All variants of a concrete initial state, including which key changes, share a
fold. Keys remain 0..63. Retain a separately reported fresh two-value canonical
binding regression at every original threshold, without reusing opened audits.

Use the already registered A2 salt/RNG, fresh 512/128/256 world counts, complete
historical audit exclusions and the reserved A1 audit exclusion. Pin these new
world hashes before fitting. The fixed 250K+2M per arm/seed and 20.25M total
budget remain; no additional pilot, warmup or repeated control is implicit.
Measure first/last/constant shortcuts and independent-interpreter agreement
before fitting. A perfect shortcut blocks a selective-update capability claim,
regardless of native accuracy; never weaken the original numeric gates.

The A1 binary fixture, source lock, scores, failed checks and every cost stay
immutable. The A2 architecture registration remains query-first selection before
truncation; this amendment changes only its prospective training/evaluation
fixture and explicitly preserves the original two-value regression. A2 has not
been implemented or fitted. Public E8 remains unchanged.
