# SHoEbox: reference defaults and reviewed fixes

13 November 2026

All produced repairs for the current 18-region native design are included in the
default reference path. The review also corrected primary-clock semantics and
removed live semantic-ancestry eviction from the fixed eligibility pool.

## Source reconciliation

GitHub main was `ea878a3`; the completion repair was on `3e5dd4d` in draft PR #2.
The recovered completion package and latest audit source match that branch byte
for byte. Its ancestry includes the earlier composition, multi-source binding,
identity/vector, relay-preservation, contribution-check and learned-message fixes.
The separate `feat/hierarchy-pilot` branch is an older six-region PyTorch study,
not an additional repair to this native runtime, and remains on its own branch.

## Enabled reference behavior

| Mechanism | Default behavior |
| --- | --- |
| Constituent formation | Multi-source binding, stable identity semantics, learned completion and recursive learned exports. |
| Teaching and transport | Relay recognition/provenance, semantic credit, retained-input credit and credited feedback activation/routing. |
| State and routing | Learned bottom state, task routing, EMA prototypes and revival. |
| Memory | Both EC recall channels and key-trace mixing. |
| Auxiliary learning | DFA gain 1, separate A1 residual objective and local innovation targets. |
| Credit coverage | Three neural hops; all 64 spatial coordinates; every-byte, 64-coordinate bottom temporal credit; unsampled third-hop credit. |

Both `fresh_model()` and low-level `Model()` select completion with this policy.
`shoebox_reference.py` supplies a repository-level entry point. Explicit older
kernels and saved checkpoint policies remain available for comparisons.
`AGENTS.md` records that completed architectural repairs belong in the default
development path and throughput regressions do not justify leaving them opt-in.

## Two additional corrections

**Primary clock:** atoms, sequences, role bindings and relays propagate primary
occurrence spans separately from coverage of all constituents. Held side context
can no longer force a split at an otherwise adjacent primary boundary. The
confidence-1.0 regression now joins adjacent events; genuine primary gaps and
overlaps still split. `completion_clocks()` exposes this distinction.

**Live semantic ancestry:** a longer repeated-pattern stress exposed overwriting
when the old 16,384-record pool filled with live dependencies. The pool now grows
only when collection cannot free a record. Reverse traversal has no fixed work
cutoff. Overflow records are serialized and restored, and stream reset clears
them. This retains the saved learning graph instead of silently dropping credit.

<!-- pagebreak -->

## Verification and measured cost

- 156 package tests pass, including 18 completion tests and six reference-default tests.
- Six additional default, checkpoint and migration tests pass with 64 experts per region.
- ASan/UBSan pass with enabled reference learning paths and forced pool growth, credit traversal, export/import and reset.
- A 16,416-record live dependency chain grows to 32,768 slots, retains its earliest producer, traverses every record and resumes identical updates after save/load.
- An actual checkpoint made by the unchanged version-33 source converts with exact fitted neural/semantic weights, gates and dictionaries, and a clean stream.
- The repeated-pattern smoke passes finite training, no lost semantic handles, no omitted semantic work, no live capacity pruning, no expired/duplicate forecasts, exact snapshot continuation and production/diagnostic parity.

The 1M fixture has 1,031,952 parameters. Timing uses one pinned CPU core, seed 7,
2,048 warmup targets and three interleaved rounds of 8,192 next-byte targets per
runtime at learning rate 0.04. Compilation, initialization, final draining and
verification are outside timed calls. The workload is repeated synthetic
sentences; these are mechanism and timing measurements, not held-out language
quality results.

| Runtime | Median training bytes/s |
| --- | ---: |
| Earlier actual default: composition | 12,725 |
| Earlier recommended runtime: relays | 3,404 |
| Current full reference: completion | 400 |

The current reference is 88.2% slower than relays on this workload.
Its semantic pool reached 65,536 slots. Performance optimization remains
subsequent work; these features stay enabled.

## Use and checkpoint contract

```python
from shoebox_reference import fresh_model
model = fresh_model()
print(model.capabilities())
```

New checkpoints are **version 34 / composition ABI 8**, including variable-length
semantic overflow. Version-33 completion checkpoints require explicit conversion
at a fresh stream boundary. From `experiments/identity_semantics`:

```bash
python src/convert_to_completion.py old.npz new.npz
python -m unittest discover -s tests -v
python tests/run_completion_sanitizer.py
python src/reference_smoke.py
```

The converter accepts versions 31, 32 and 33, preserving their fitted policy.
Version-31/32 checkpoints still load directly in their original kernels. Explicit
`profile="historical"` retains the previous completion training settings while
using the repaired clock and pool. Fresh development models use the full policy.

## Remaining design work

The audit's remaining targets are reusable upper pattern classes, exported
content driven by constituent-specific expert computation, local neural credit
to child experts, and neural producer eligibility for long completed events.
Native hippocampal recall still uses two independent EC associative transition
channels; shared content/context episode identity remains to be implemented.

Eight historical experimental control families are absent from this kernel and
are reported as unavailable. Three previously accepted no-op port controls now
reject nonzero values. `capabilities()` reports effective switches, memory
contract and semantic capacity. These absent implementations are not represented
as enabled features. Source hashes and compact test/measurement files accompany
the code; trained models, corpus files and compiled binaries are excluded.
