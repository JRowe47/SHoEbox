# EXP01: fast online query/readout diagnosis

This package tests whether the current state contains usable bindings and whether
a small query-conditioned output can recover them. Both `minimal` v19 and
historical `full` corrected C are named explicitly. The public runtime is unchanged.

**Result:** [report, literature review, and measurements](../../reports/exp01/SHoEbox_EXP01_Query_Readout.md).
The generic online heads do not solve binding. A factorized diagnostic can decode
the first queried value from L3, which directs the next step toward learned
address/value selection and credit rather than larger state or deeper loops.

## Online implementation

`Adapter(kind="bilinear")` adds a rank-16 residual to the existing hierarchical
256-byte decoder. Its query-side input is the four current 64-value states; its
memory-side input is the four current 32-value retained reads. No task parser or
explicit query index enters this model. The projection uses
`tanh(Uq+bq) * tanh(Vm+bm)`, followed by one output projection. The additive
control replaces the product with the mean of the same two projected vectors,
with identical fitted capacity. The original decoder remains in the residual path.

The factorization follows [Kim et al. (ICLR 2017)](https://arxiv.org/abs/1610.04325).
The separation of address and value follows
[Miller et al. (EMNLP 2016)](https://aclanthology.org/D16-1147/).
These are SHoEbox adaptations, not replications of those full networks.

- 10,511 extra fitted parameters, deducted from expert-bank capacity.
- 42,044 weight bytes; 2,748 bytes of scratch input/activation/error cache.
- No additional persistent sequence state, optimizer moments, replay, attention
  history, refinement pass, or observation-clock advance.
- Per-target normalized SGD. `joint=True` passes exact head VJPs into current
  live states and the existing retained-read gradient path. Native local credit
  still detaches at retired boundaries; this is not full-sequence backpropagation.
- `joint=False` trains only the added head on a frozen pretrained core. The core
  continues its ordinary streaming memory writes without fitting its parameters.
- Zero-initialized output weights make a fresh adapter match its adopted core.

The isolated research build inserts checked, scoped hooks in a compiled copy of
`native.cpp`. It does not modify production files or their ABI. `kind="native"`
has exact native training/checkpoint parity. Wrapper checkpoints include core,
head, policy, and unfinished stream state, with checksum verification.

```python
from experiments.query_readout.adapter import Adapter

with Adapter(core="minimal", kind="bilinear", parameters=500_000) as model:
    model.run(b"observed", b"bserved!")  # next-byte online learning
    checkpoint = model.dumps()
    with Adapter.loads(checkpoint) as continued:
        continued.run(b" bytes", b"bytes.")
```

## Reproduction

Run from the repository root with NumPy and a C++17 compiler. Matplotlib is used
only to regenerate the report figure. The existing EXP00 positive-control tests
also use CPU PyTorch.

```bash
python -m experiments.query_readout.run train --out /tmp/shoe-exp01 --workers 2
python -m experiments.query_readout.run audit --out /tmp/shoe-exp01 --workers 2
python -m experiments.query_readout.single_copy --out /tmp/single-copy.json
python -m experiments.query_readout.probe \
  --checkpoint /tmp/shoe-exp01/minimal_1/native.qr \
  --out /tmp/probe-minimal-1.json --audit
# Repeat probes for both cores and seeds 1, 7, 19.
python -m experiments.query_readout.benchmark \
  --runs /tmp/shoe-exp01 --out /tmp/inference-cost.json
python -m unittest experiments.query_readout.test_adapter -v
python -m experiments.query_readout.check_native
python -m multiscale.tests.run_all
python -m experiments.query_readout.report
```

Use `--no-leak-check` only where the host prevents LeakSanitizer operation; ASan
and UBSan stay enabled. Do not run throughput measurements alongside training or
other benchmarks. Saved `.qr` models and generated datasets are temporary and
must not be committed.

`run.py` writes the plan before training and exposes a separate audit phase.
The fixed screen is 250K warmup bytes plus 500K continuation bytes per final arm,
three seeds, 500K nominal capacity, and six arms per core. The source, splits,
training order, charges, and every seed are retained in the compact evidence.
This is the first diagnostic screen, not the broader six-policy promotion sweep.

## Diagnostic boundaries

`probe.py` stops before the first scored answer. It fits ridge heads and optional
normalization on training examples only. Raw, centered, and standardized probes
use the same solver, regularization, dimensions, and folds. It inspects fast and
slow states, retained reads, and an explicitly labeled **read-only requery** of
the address matrix; that requery never mutates its source or advances its clock.

The factorized probe learns five intermediate labels: query identity, two record
keys, and their two values. It then compares **predicted** key identities and
selects a **predicted** value. Its strict score additionally requires a unique
predicted-key match. Parsed validation keys never choose the primary answer.
A separate, explicitly labeled oracle grants the true record index.

This demonstrates accessible representation under intermediate supervision and
an offline diagnostic fit. It is not the native online decoder, native two-answer
rollout, or language-generation performance. All explicit presentation siblings
stay in the EXP00 world split; the audit is held-out concrete worlds, not unseen
identifier alphabets or all isomorphic renamings.

Task motivation: [Zoology/MQAR](https://arxiv.org/abs/2312.04927) and
[MAD](https://proceedings.mlr.press/v235/poli24a.html).
