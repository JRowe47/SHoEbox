# EXP00: calibration before architectural comparisons

This namespace implements the first item of the
[capability development plan](../../docs/capabilities/SHoEbox_Capability_Research.md).
It leaves the nine historical `experiments/mad` tasks and the production runtime
unchanged. Results belong in `reports/exp00/`; weights, generated datasets and
compiled libraries do not belong in git.

## Reproduce

Use Python 3.10+, NumPy, CPU PyTorch and a C++17 compiler. The recorded run uses
the exact versions in its provenance. For a CPU-only PyTorch installation:

```bash
python -m pip install numpy
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m unittest experiments.mad.test_protocol experiments.mad_next.test_exp00 -v
python -m experiments.mad.check_native --sanitize
python -m experiments.mad_next.run
python -m experiments.mad_next.run --execute --out /tmp/exp00-results.json
python -m experiments.mad_next.run --phase audit --execute --out /tmp/exp00-results.json
```

The first runner invocation only prints the registered plan. Training refuses
to overwrite an existing result file. Audit requires completed training and a
locked development selection; test results never choose the task setting.
Temporary checkpoints survive between phases, then are deleted after a
successful audit. An interrupted train is not automatically resumed.
The recorded report can be rebuilt without training via
`python -m pip install -r reports/exp00/requirements.txt` followed by
`python -m reports.exp00.build_report`.

## Tasks and splits

| Task | Fixed setting | What must be recovered |
| --- | --- | --- |
| Binding | Two distinct values; 48 possible key names; value vocabulary 2/4/8 | Both key-specific answers in independent query order |
| Initialization | Four registers, eight values, distinct initial contents | All four initial values |
| Set | Same state plus one non-noop assignment | Changed destination and preservation of other registers |
| Copy | Same state plus one copy between distinct registers | Destination receives the source's old value |
| Swap | Same state plus one swap between distinct registers | Both exchanged values and preservation of other registers |
| One hop | Three named nodes from a pool of 16; arbitrary non-self outgoing edges | All three directly addressed successor nodes |

Every sampled problem emits a Cartesian product of cyclic record-order and
query-order rotations. This balances every record position against every query
position, with randomized starting permutations. It is not an exhaustive
permutation suite for three/four items. Requested episode counts round up to a
complete family; the actual counts are recorded.

Protocol v2 hashes the task and sorted **initial mapping** into 60/20/20 folds.
All operations applied to the same initial mapping share a fold. Swap(a,b)
and swap(b,a) have one semantic identity for deduplication. Explicit query,
padding and renaming variants preserve their parent's lineage. Folds hold out
concrete initial states, not all isomorphic renamings or novel program
structures. Each result records both episodes and distinct initial-state
groups; family members are correlated observations.

Byte framing follows MAD's observed/scored distinction. Only answer bytes are
supervised. Normal training places answers at offsets divisible by 64;
diagnostics sweep every offset 0..63. The independent interpreter parses
observed bytes and executes array-based semantics; generation uses a separate
dictionary implementation. Neither reads the scored answer payload. Tiny
oracle checks include repeated values and no-op cases as well as the harder,
distinct-value/non-noop training fixtures.

## Learned positive controls

These are learnability controls with their own resource budgets:

- **Attention lookup:** a learned query/key compatibility table, learned value
  embeddings and an output projection. No symbolic key equality or lookup is
  applied by the input adapter.
- **State-gated GRU:** an ordinary 128-unit GRU consumes observed instruction
  fields; learned register output heads are selected by the observed query.
  The adapter exposes opcode, register and literal identities but never
  executes set, copy or swap.

Both receive parsed record syntax, omit padding and predict each query without
feeding gold answers back. Register-addressed output heads are an explicit
inductive bias. Thus a successful control establishes learnability through
this adapter, not a raw-byte language capability or a resource-matched win
over SHoEbox. Padding invariance in these controls is by construction.

| Training per seed | Native C | Learned control | Finite training episodes |
| --- | ---: | ---: | ---: |
| Each binding vocabulary | 250K bytes | 250K bytes | 2,048 |
| Initialization | 250K | 8M | 8,192 |
| Set | 250K | 32M | 65,536 |
| Copy | 250K | 128M | 65,536 |
| Swap | 250K | 128M | 65,536 |
| One hop | 250K | 250K | 2,052 after family rounding |

Seeds are 1, 7 and 19. C uses the existing MAD learning policy and no extra
loops. Attention uses Adam at 0.05; GRU uses Adam at 0.003, with batch size 32
and gradient-norm clipping at 1. Training orders are seeded, shuffled finite
epochs, independent of model seed. Planned and actual encoded input-byte
budgets include padding and repeated presentations. A final partial episode
is charged but supplies no labels to the record-based control. Its discarded
partial bytes are reported. Optimizer/model state round-trips are verified.

The complete formal run is 48 fits and **897M charged bytes**: 6M for C and
891M for the learned controls. Earlier development-only pilot costs are kept
separately in `reports/exp00/pilots.json`; they are not hidden in the binding
tranche's original 31M planning estimate. No EXP01 architecture sweep is run
here. CPU and wall timers exclude evaluation and initialization. Native timers
cover byte-level processing; reference timers cover input encoding and batched
updates. Do not use their ratio as a matched efficiency result.

## Selection and audit

The primary gate is **whole-episode exactness**. For each setting, let `B` be
the strongest registered shortcut's exact score and `A` be C's mean exact
score across seeds. Use `(A-B)/(1-B)`, never chance-normalization alone. The
comparative band is 0.1..0.9, with at least three seeds, 512 development
episodes, a learned control at >=95% mean and >=90% on every seed. A shortcut
ceiling of 1 invalidates the setting.

Registered shortcuts emit input-order values, the last supplied value, the
majority supplied value, or the original values while ignoring updates.
Independent uniform guesses and non-start-node guesses are also reported.
These are diagnostic controls, not a proof that every possible shortcut is
excluded. In balanced binding, input-order copying is exactly 50% exact; an
ignore-update baseline is 0% exact on atomic updates despite high token scores.

Selection uses final **autoregressive** C scores, not its teacher-forced
learning curves. Eligible settings closest to normalized 0.5 are selected.
If none qualifies, the first/easiest declared setting is audited and explicitly
marked diagnostic-only. Final audits use at least 1,024 episodes, without
changing the selected recipe.

Reported endpoints distinguish restricted-alphabet autoregressive exactness,
full-256-byte autoregressive exactness, teacher-forced token/NLL scores, and
teacher-forced changed/unchanged-register accuracy. Category masks change only
what is counted, not the byte stream. References have task-alphabet outputs;
they do not have a native-byte score.
Both native decode modes receive the observed query and framing bytes;
autoregression replaces earlier scored answer bytes with the model's predictions.
This is not a free-form syntax-generation evaluation.

Counterfactuals reverse query order or change an argument within the same
trained operator family: new set value, reversed copy direction, or a new swap
partner. Every offset is evaluated on up to 32 distinct held-out initial-state
groups, and query/operation changes on up to 128 groups. These are frozen
diagnostics, not extra training. No-query, zero-state and erased-operation
lesions test the learned controls' dependence on their inputs.
