# EXP01-E6: next registered applicability-learning study (not yet fitted)

Registered after E1–E5's completed fits. Production remains v19. E1's three-seed
fresh audit reaches 98.83% native exactness, but all-offset mean is 94.11%, below
the unchanged 95% threshold. Frozen D3 on this same fresh fold is also 94.14%.
The registered 1M/5M comparisons retain high primary exactness and padding-delay
success but do not repair this offset mean. No opened audit world may guide
which example, phase or coefficient to modify.

E4 fixes full-alphabet readout calibration: after 1M language targets its query /
value diagnostics remain identical, and gate-on restores 99.41% development
native exactness. Its natural gate falls from 0.52071 to 0.15078 and binding
falls to zero. E5 adds learned evidence-presence inputs but still loses binding
by 250K generic targets. This isolates applicability plasticity as the next
measured retention problem. E3's transferred-policy pilot failed before language
fitting and is not evidence against a probability mixture trained from scratch.

## Hypothesis and scoped change

Compare generic-loss applicability (E4/E5 controls) with a locally supervised
applicability classifier. For the supervised arm, the training-only annotator
marks an answer-request context using ONLY the already observed local window;
never an answer byte, correct source, desired lookup value or future label.
Use the established local presence SGD recipe, without a rate sweep. This is
fixture-specific supervision and must be named as such. The learned gate sees
continuous existing state at inference. No symbol tests or parser enter it.

Keep the calibrated generic eight-bit decoder and paired source machinery.
With no local labels, update the native core under the mixture next-byte loss;
freeze the already locally supervised recognizers/projections/applicability.
This explicitly separates supervised extraction from generic byte learning.
Do not describe it as learning general applicability from generic next-byte
objectives. Include the generic-loss-gate control, no-local control and gate
interventions so this tradeoff remains visible. No new sequence state or slots.

## Registration and budgets

Use the E2-family grouped training/development folds and its still-unopened
audit (E2–E5 pilots never opened it). They exclude all D2/D3 and E1 primary audit
worlds. Freeze a source/protocol manifest before fitting. Pilot seed 7: 250K
warmup +2M continuation, all charged spacers and 64 phases, then 1M pinned
language targets with no local labels. Evaluate complete DEVELOPMENT gates
before and after; compare the fixed generic-loss-gate control at the same
budget, not an uncharged frozen intervention. Total pilot budget: 6.5M new bytes.
Do not continue an E1 audit-trained choice. Diagnose any phase failure on
DEVELOPMENT worlds before revising local supervision or coverage.

If development passes, register a final three-seed comparison (native, frozen
D3, generic-loss gate, selected local gate, no-local control), same 250K +2M
per arm, and lock before the untouched audit. Repeat fresh document-grouped
language, copy and reconstruction regression audits excluding previously opened
generic audit groups; then 1M/5M checks. Preserve original sample sizes, thresholds,
all 64 offsets, both query orders, full 256-byte autoregression, independent
+64/+128/+256 delays, query/state/pairing lesions and >=80% native throughput.
All production gates remain mandatory. Any further failed mechanism gets a
new preregistration; no audit tuning, no default promotion from pilot results.

Cite Miller et al. (EMNLP 2016) for paired representations and See et al.
(ACL 2017 §2.2) for learned mixture switches. Preserve all E1–E5 kernels and
checkpoint meanings. Run numerical, causal, exact continuation and ASan/UBSan
checks. Direct verified publication to main remains explicitly authorized.

### Pre-fit implementation lock

The local applicability target is the existing query-presence label: whether
the just-observed two-byte window is a request. This label uses no future byte.
It gets the same normalized presence SGD (rate 0.16, positive weight 4) as the
existing recognizer. The generic-loss arm retains E4's gate objective. Both arms
start fresh; transferred E4/E5 weights are not used for this comparison.
E6 has its own checkpoint magic and explicit serialized policy. Both policies
cache eight bit log-probabilities while expanding the full 256-byte readout,
eliminating duplicate softplus calls without changing the arithmetic result.
Numerical parity is checked before fitting. This addresses the measured E4/E5
prototype inference shortfall without relaxing the 80% threshold.
