# EXP01-C / EXP22: query-time access over existing state

This isolated research build preserves the production minimal-v19 ABI and the
EXP01-B implementation. It compares three frozen-core, online, rank-16 heads:

| View | Current query | Key/value evidence |
| --- | --- | --- |
| `l3` | Completed L3 state | Completed L3 state (EXP01-B numerical control) |
| `split` | Provisional composition of valid unfinished constituents | Completed L3 state |
| `pool` | Own learned source pool | Four independently learned source pools |

All heads have exactly 9,535 parameters deducted from the requested core budget.
The final query/key soft selection and native 256-byte residual decoder match
EXP01-B. No task parser, target label or selected-record index enters prediction.
Prefix-byte reconstruction labels are explicit *training-only* supervision.

The pool reads retained stable ordered pairs, the current completed stable
constituent at each scale, and valid children of unfinished constituents. Each
source has 60 stable-content coordinates and four structural coordinates:
scale, bounded occurrence age, completed/retained/partial type and child role.
Existing stable codes depend on observed content, not fitted labels. This arm
tests those fixed content channels; it does not read the retained learned-value
half. Each lane ties its first projection row to its source-attention scorer.
Both gradient paths are included. Projecting after pooling is deliberately
bounded but can mix distinct records; the experiment tests that limitation.

No write/admission rule changes, new persistent sequence state, growing tape,
optimizer moments or inference iteration are introduced. At most 80 scratch
source slots are allocated (77 can be populated by this four-scale inventory).
Five online normalizers occupy 5,160 bytes; all freeze during inference. The
shared-L3 arm's normalizers are identical copies, retaining numerical parity
while using the same research layout as the new views. All memory is reported,
including this redundant allocation. Existing core state is separately counted.

`provisional()` composes only valid child prefixes and never writes them back.
At a completed boundary it uses the completed content. It never reads stale
entries beyond `children_count`. It is an ordered composition probe, without
executing additional expert transitions or modifying model clocks.

## Reproduce

The [protocol](../../reports/exp01c/PROTOCOL.md) predates fitting. Each run writes
its own source/protocol hashes before training, refuses to overwrite its plan,
and separates development selection from the audit phase.

```bash
python -m unittest experiments.query_access.test_adapter -v
python -m experiments.query_access.check_native
python -m experiments.query_access.run pilot --out /tmp/exp01c-pilot --workers 1
python -m experiments.query_access.run train --out /tmp/exp01c-formal --workers 3
python -m experiments.query_access.run audit --out /tmp/exp01c-formal --workers 3
python -m experiments.query_access.positive_control --out /tmp/exp01c-controls
python -m experiments.query_access.inspect_sources --runs /tmp/exp01c-formal --out /tmp/exp01c-formal/source_presence.json
# After all other computation completes, measure isolated inference:
python -m experiments.query_access.benchmark --runs /tmp/exp01c-formal --out /tmp/exp01c-formal/benchmark.json
python -m experiments.query_access.report --runs /tmp/exp01c-formal --pilot /tmp/exp01c-pilot --controls /tmp/exp01c-controls --out reports/exp01c
```

Use `--no-leak-check` only where the host prevents LeakSanitizer; ASan/UBSan
remain active. Correctness tests cover finite differences, native and EXP01-B
parity, read-only views, frozen-core exactness, boundary continuation, causal
labels, ignored targets and independently checked per-query scoring.

`inspect_sources` is an external diagnostic: it reconstructs fixed signatures
for known record spans and the observed query, then inspects retained vectors.
Its oracle span locations never enter training or native inference. Presence
does not prove a trained native decoder can recover the bytes.

Fresh folds group all query/record-order/offset siblings. Training cycles
through 64 leading-padding offsets; delays remain out of training. Native
scoring feeds generated answers into later state. Counterfactuals fork the same
record-prefix state, preserve padding, and change query order only. Seed means
share audit worlds and are not independent-world sample counts.

## Primary literature

- [Miller et al., EMNLP 2016](https://aclanthology.org/D16-1147/), section 3:
  separate address and value representations; adapted to existing neural state.
- [Dai et al., ACL 2019](https://aclanthology.org/P19-1285/), sections 3.2–3.3:
  completed-context access across segment boundaries, without a new cache here.
- [Sun et al., 2023](https://arxiv.org/abs/2307.08621), section 2.1:
  within-chunk and recurrent contributions motivate the provisional comparison.
- [Yang et al., ICLR 2025](https://openreview.net/forum?id=r8H7xhYPwz):
  distinguish read/access failure from erase/write correction before changing
  memory. No gated delta-rule mechanism is implemented in this experiment.
