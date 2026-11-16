# D2: separate query recognition from occurrence recency

Registered after D1 development, before D2 fitting or any EXP01-D audit access.
D1 source snapshot: 41fdc6c. Its 750K pilot and 26.25M formal-training bytes
remain charged. No D1 held-out audit has been opened.

D1 final development exactness is 90.04% for each seed. First/second answers
are 99.41%/90.23%. Longer gate fitting increases query-dependent confidence
margins, allowing an older query to outweigh the fixed additive age preference.

D2 uses the classifier's natural zero-logit boundary: among positively
recognized query constituents, choose the largest existing occurrence timestamp.
When none is recognized, preserve D1's score-minus-age fallback. No symbol
test, hand-authored query parser, new weights/state, or threshold sweep.
All key/value readout and training code stays identical. D1 is a saved policy;
old checkpoints continue using it. D2 gets its own serialized policy value.

First inspect a frozen D1 checkpoint with only this query-routing intervention
on the same development data (zero fitting). Then fit D2 from scratch for
three seeds 1/7/19, each with 250K reduced-core warmup plus 2M head continuation:
6.75M additional charged bytes. The other D1 arms are the paired controls.
Reuse the registered untouched EXP01-D audit fold after locking D2 selection;
audit D1 and D2 together and do not adjust a mechanism from audit outcomes.
All original binding, counterfactual, phase, delay, correctness and cost gates
remain unchanged. Production regression qualification remains separately gated.
