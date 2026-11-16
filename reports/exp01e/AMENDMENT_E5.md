# EXP01-E5: condition applicability on already learned evidence presence

Registered after E4 development-only diagnosis. E4's decoded value and current
query are unchanged after language training: forcing its gate on restores
99.41% native exactness. Its learned answer gate falls from 0.52071 to 0.15078,
which makes the adapted native language decoder override recall. Restoring only
the gate intercept does not restore binding, so this is not just bias drift.
E4's conditional from-scratch Pilot B is not triggered because retention fails.

Extend the gate's current-pair features with three bounded causal features:
`tanh(record_presence_score_0)`, `tanh(record_presence_score_1)`, and
`tanh(query_presence_score)`. All are outputs of the existing learned D3
classifiers on its selected constituents. No new classifier, symbol test,
source oracle, parser or state slot is added. Three extra fitted coefficients
start at zero, preserving the explicit converted checkpoint's first prediction.
The sigmoid applicability learns online from generic mixture loss, with the
same normalized rate. The hypothesis is that evidence presence distinguishes
retrieval contexts from unrelated language and reduces destructive gate updates.

Pilot A: explicitly convert E4's PRE-language checkpoint into a new E5 format,
record both hashes and preserve its inherited E2 binding fit (2.25M bytes,
already charged). Fit exactly 1M pinned FineWeb targets with no local labels,
evaluate development every 250K and repeat all development binding gates.
Charge 1M new bytes. No E2–E5 audit access. Fitted readout: 7,673 weights, no new
persistent sequence state. Preserve every E1–E4 policy and failure.

A pass advances to a separately preregistered from-scratch, three-seed
qualification on the still-unopened E2-family audit and fresh generic audit
folds. A failure requires a new focused retention mechanism; do not silently
retune these coefficients or open that audit to choose it. Every original
binding, delay, intervention, throughput and production threshold remains.
