# EXP01-E8: next registered cold-start gate revision (not fitted)

E7 fixes applicability timing and retains 99.41% native exactness through 1M
generic language targets in the seed-7 development pilot. All 64 offsets average
98.05%, worst 95.31%, both query orders 99.22%, delays 100%, before and after.
However, three-seed generic development qualification fails: 2.746686 versus
2.227511 nats/byte native; copy 70.9635% versus 72.0052%. The cold local gate is
0.5 and receives no labels or generic gradients in these independent fits.
It reserves half the probability mass for an untrained retrieval distribution.
Reconstruction passes, 77.9297% versus 77.4089%. No E7 audit was opened.
Bypassing the gate after the generic fit still gives 2.504123 development NLL,
so the mixture also weakened native learning; output rescaling alone is not
a complete fix. This diagnostic used no additional fitted updates.

## Focused hypothesis and fixed implementation choice

Initialize only the local applicability bias to logit(1/256), using the byte
alphabet size as the fixed cold-start prior. Keep all other E7 weights, rates,
budgets, loss rules and bounded state unchanged. Gate weights remain learned
from observed request labels on the binding curriculum and frozen when those
labels are absent. No inference parser, byte equality test, oracle, slot or
exact store. This is a fixture-supervised classifier with a generic fallback;
do not claim generic byte loss learns its applicability.

Give E8 a new policy identity and checkpoint magic; do not reinterpret E7 or
change its initialization in place. Prove that only the registered prior changes.
Inspect gate activation on generic development text, including after binding
training: a successful cold-start prior need not fix out-of-distribution gate
activations after local fitting. Measure that separately before promotion.

## Locked next execution budget and controls

First repeat the fresh seed-7 250K warmup +2M binding +1M pinned language pilot
(3.25M additional charged bytes), using E2-family train/dev and the original
retention corpus. Compare retained E7 at identical budgets and fixed data order.
Then run fresh native, D3 joint and E8 generic fits across seeds 1/7/19 on the
already pinned E7 corpus and synthetic folds: 1M language +250K copy +250K
reconstruction per arm/seed (13.5M bytes). This is a fixed prior, not a sweep.
Charge repeated controls; retain all failures. No local syntax labels on generic
fits. Track bytes including padding, objective/local update counts, CPU time,
actual fitted budget, state, native full-alphabet output and capability retention.

If development passes, register a final three-seed native/frozen-D3/generic-gate/
local-gate/no-local comparison and its post-language retention, lock final
snapshots, then open the still-unused E2-family binding audit and E7 generic
audit. Repeat 1M/5M capacity qualification. Keep the original 95% mean /90%
minimum-seed primary and per-position requirements, 90%/85% query-order and
delay requirements, all-offset mean 95% /worst-offset seed mean 90%, all query/
state/pairing lesions, and >=80% native inference throughput. Do not inherit
E1 audit failures or E7 pilot successes as E8 qualification.

If any gate fails, preserve v19 and preregister the measured next revision.
Preserve all E1–E7 code, checkpoint meanings and evidence. Relevant Python/native,
causality, exact-cursor continuation and ASan/UBSan checks remain mandatory.
Verified commits and research results may be pushed directly to main without
another publication confirmation; preserve remote history and never force-push.

Primary grounding remains Miller et al. (EMNLP 2016,
https://aclanthology.org/D16-1147/) and See et al. (ACL 2017 §2.2,
https://aclanthology.org/P17-1099/). The cold prior is this experiment's design
choice, not a result attributed to those papers.
