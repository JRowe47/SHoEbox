# EXP01-E3: probability-mixture responsibility credit

Registered after E2's development-only retention test also fell from 99.41%
exact binding to 0% after 1M language targets. E2 never opens an audit. E1
binding audit is running; no E1 audit score informs this change. E2 preserves
independent readout logits, but logit blending still sends harmful generic
updates into the retrieval decoder.

Replace logit blending with a normalized 256-byte probability mixture:
`P(y)=(1-g) P_native(y) + g P_readout(y)`. The online posterior responsibility
`r=g P_readout(y)/P(y)` weights readout loss by r and native loss by (1-r).
The gate-logit derivative is `g-r`. This is the mixture-switch principle of
See et al., ACL 2017 §2.2, https://aclanthology.org/P17-1099/, applied to two
native byte distributions, with no source-pointer distribution. It avoids
training a badly mismatched retrieval expert as strongly on an unrelated byte.
Inference remains causal; target-derived responsibility exists only in training.

Same 9,965 fitted readout weights, no persistent sequence state, no slots,
no replay. Eight target-path nodes are inspected before the combined online
update; inference mixes all 256 probabilities. This extra CPU work is charged
and must meet the original >=80% throughput gate before promotion.

Pilot A: explicitly convert E2's already fitted seed-7 absolute-readout weights
into a new E3 checkpoint format, recording source/destination hashes. This is a
new policy evaluation, never a reinterpretation of a saved E2 checkpoint.
No fitting for conversion. Evaluate E2 development binding, then train 1M
FineWeb targets with no local labels and repeat development gates. Charge that
1M separately and disclose the inherited 2.25M binding fit from E2.

Pilot B, only if A retains >=90% development exactness: train E3 from scratch on
the same unopened E2 grouped folds, 250K native warmup plus 2M joint every-byte
continuation; then 1M FineWeb, total 3.25M new bytes. No audit access. Subsequent
three-seed confirmation and all unchanged production gates are required before
promotion. Failed pilots and all intermediate curves remain in the ledger.
