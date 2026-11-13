# SHoEbox: neural producer retention

13 November 2026 | First recommended fix | Base: `13bd9f0`

Completed events now retain their original neural producer for as long as a live consumer or pending update needs it. The repair is enabled in the default completion/reference path. It adds no learned parameters: the test model remains **1,031,952 parameters**.

## Implementation

- The existing 64-entry regional ring is a recent-record cache. Displaced producers move into an owned archive keyed by **region and immutable execution tick**. The old ring index is only a lookup hint; stale ticks never resolve to a replacement occupant.
- Each archived record retains the selected expert, activations, inputs, routing cue, input/feedback derivatives, accumulated credit, and matching bottom-state temporal eligibility. Neural input records also keep their semantic DAG alive.
- Collection starts from held and relayed publications, pending completed frames, retained inputs, forecasts, recent eligibility and outstanding credit. Dependency traversal obeys the configured neural credit depth; transparent relays preserve provenance. Sampled bottom temporal Jacobians retain their required intermediate records without recursively extending the temporal chain.
- Delayed task, local issuer and feedback updates use the saved original record. Input and neural credit flushes include archived records in chronological order. Reset and document-reset paths reclaim the archive. Bottom expert revival defers replacing a generation still owned by retained eligibility.
- The reference smoke now fails on missing neural parents, feedback producers, retained-input links, lost semantic handles and overwritten pending credit.

## Checkpoints

New checkpoints use **version 35 / composition ABI 9**, including the complete neural archive. Save/load, snapshot copies and differently chunked continuation preserve exact state and outputs.

Version-33 and version-34 completion checkpoints require explicit fresh-stream conversion. Their already-expired producers cannot be reconstructed. The converter preserves fitted weights, dictionaries, payload gates, saved policy and preprocessing means, while clearing event/working/associative memory, forecasts and pending updates. Versions 31 and 32 retain their historical kernels and formats.

```bash
python experiments/identity_semantics/src/convert_to_completion.py OLD.npz NEW.npz
```

## Verification

**165 distinct regression tests passed:** the 164-test package run plus one additional semantic-ownership regression. The final eight-test neural-retention suite passed; seven of those checks also passed with 64 experts. The native ASan/UBSan harness passed training, evaluation, draining, reset, serialization and semantic-pool growth.

Controlled tests hold the original producer across **256 subsequent source executions**. Language, local issuer and feedback updates match the immediate-update reference exactly and change only the intended expert. Coverage includes repeated roles, relay provenance, all ownership roots, source freezing, expert-generation protection, credit-depth reclamation, malformed snapshot rejection, semantic-pool collection and checkpoint continuation.

| Matched repeated-text stress | Before | After |
| --- | ---: | ---: |
| Failed neural-parent lookup attempts | 30,464 | 0 |
| Median training bytes/s | 437.7 | 159.8 |
| First measured round NLL | 2.219747 | 2.219750 |
| Second measured round NLL | 1.768917 | 1.768730 |

The repaired run also has **zero** missing feedback producers, expired retained-input links, missing live dependencies, expired semantic handles and expired local forecasts. Snapshot continuation and diagnostic-kernel parity are exact. At the measurement endpoint, 29,674 records remain archived and 15,587 have been reclaimed.

Both timings use seed 7, one pinned CPU, 2,048 warmup targets and two rounds of 4,096 next-byte targets, with LR 0.04 and local LR 0.04/3. The measured rate is 63.5% lower with retained credit. This repeated-sentence mechanism test includes the additional delivered updates and their changed training trajectory; it is not a held-out quality comparison. No trained model or corpus is included in the patch.

## Scope and reproduction

This completes retention fix 1 and its missing-credit assertions. Propagating local prediction error into causal child neural experts remains fix 2; the local acceptance test here verifies correct teaching of the saved issuing expert.

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s experiments/identity_semantics/tests -v
python experiments/identity_semantics/tests/run_completion_sanitizer.py
python experiments/identity_semantics/src/reference_smoke.py --kernels completion --targets 4096 --repeats 2
```

Compact measurements: `experiments/identity_semantics/results/neural_retention_validation.json`.
