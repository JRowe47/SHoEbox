# Cortical indexed expert execution

The `fused` CPU backend now evaluates selected experts directly from their
weight banks. It keeps the existing MinKron fusion, architecture, routing,
GELU, gate weighting and `1/sqrt(K)` aggregation. Parameter and checkpoint
schemas are unchanged. `SHOEBOX_EXPERTS=0` selects the previous expert path
while retaining fused MinKron, providing an isolated comparison.

## Indexed arithmetic and bounded gradient groups

For input `x`, selected row `e` and routing weight `g`, the native loop computes
`z = up[e] @ x`, `h = GELU(z)` and `down[e].T @ h`. It combines the selected
outputs with the existing routing weights. It reads bank storage in place;
there are no gathered copies of the selected matrices. Training saves the
input, selected IDs, gates, bank references and small preactivation tensor.
Inference allocates no preactivation tape.

Backward computes input and gate gradients plus compact selected-row
derivatives. Each bank provides groups of eight independent autograd views.
These are aliases of the same parameter storage, not copies. Each selected
expert invocation uses a different edge. Once the group's derivatives arrive,
one native scatter-add writes them into a newly allocated dense bank gradient.
Repeated row IDs accumulate correctly. Unused edges remain `None`, avoiding
materialized zero gradients for incomplete groups.

This removes repeated sparse concatenation/coalescing and reduces whole-bank
gradient allocations by up to eightfold. The group size bounds pending row
derivatives. It is an execution grouping, **not a gradient truncation**: the
full recurrent gradient graph remains connected. A full wide B4/T32 call has
6,528 individual up/down bank uses and 816 dense gradient groups.

The existing per-forward context owns the view pools and releases them on
return or exception. Separate model calls have separate pools. The backward
does not write parameter `.grad` or a shared mutable accumulator. Parameter
hooks, `autograd.grad`, retained graphs, gradient accumulation, clipping and
dense AdamW see ordinary dense gradients. Inactive rows therefore retain
normal optimizer moment and weight-decay behavior.

## Numerical and runtime contract

Linux CPU FP32/FP64 uses the same locally compiled extension as MinKron.
Unsupported devices/dtypes, missing compilers, disabled native execution and
CPU autocast use the portable expert path. No prebuilt binaries are shipped.
As with fused MinKron, native training supports first-order reverse-mode;
use `optimized` for higher derivatives.

Floating-point reduction order changes. FP64 output and derivative comparisons,
finite-difference gradcheck and a multi-update AdamW comparison check the
equations independently. Tests cover duplicate IDs, noncontiguous inputs,
frozen parameters, incomplete/multiple accumulation groups, retained graphs,
two CPU threads, disabled native execution, and live state crossing model
calls. Matched full-size FP32 runs also compare logits and greedy predictions.

## Reproduction

Use the exact 1,004-byte `profile_train.bin` included with the results. The
benchmark alternates old/new order across three pairs of isolated processes.
Both paths have one CPU thread, four streams, eight warmup observations and a
32-observation gradient window. Compilation and warmup are excluded.

```bash
SHOEBOX_REQUIRE_FUSED=1 python scripts/check_all.py

python scripts/benchmark_expert_fusion.py \
  --data artifacts/expert_fusion/profile_train.bin \
  --out artifacts/expert_fusion/matched --repeats 3

python -m shoebox.profile_wide --mode train --backend fused \
  --data artifacts/expert_fusion/profile_train.bin \
  --out artifacts/expert_fusion/train --targets 1000 --batch 4 --context 32
```

The harness requires an empty comparison directory. Training protects existing
complete and partial logs. Reports record requested expert execution separately
from the MinKron backend. The source digest includes the native kernel.

Measured results and endpoint audit are in `cortical_expert_fusion_results.json`.
