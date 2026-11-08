# Cortical MinKron fusion

The `fused` execution backend replaces the HAT's pairwise chunk calculation
with a native CPU scan and analytic backward. It also uses that recurrence
for the three single-step MinKron cells inside each regional SHoE. Projections,
normalization, routing and byte heads remain ordinary PyTorch operations.

```python
from shoebox import CorticalModel, cortical_wide
model = CorticalModel(cortical_wide()).set_execution_backend('fused')
```

## Same recurrence, smaller intermediates

For one head and incoming internal byte, the implementation computes

\[
D_t=\operatorname{diag}(\exp(g_t))S_{t-1},\qquad
n_t=v_t^\top v_t,\qquad r_t=1/\max(n_t,10^{-18}),
\]
\[
u_t=c_t n_t r_t-\beta_t r_t(k_t^\top D_t v_t),\qquad
S_t=D_t+u_t k_t v_t^\top.
\]

These are the original compact MinKron state equations. Instead of building
pairwise time matrices and solving their triangular system, a compiled loop
applies the state updates directly. The native backward walks the saved states
in reverse and accumulates derivatives for keys, values, decay, erase/write
gates and the incoming state. Every write remains differentiable.

For terminal HAT reads, recurrence work is O(BTHKV), and the saved state tape
is O(BTHKV). There are no T-by-T recurrence intermediates. No tape is allocated
during inference. The returned final state owns compact storage, so detaching
it cannot keep the history tape alive. Query/read normalization and output
projection remain outside the fused recurrence.

The model dimensions, byte ports, HAT objective, 32-observation TBPTT,
learning-rate schedule and parameter count are unchanged from the wide preset.
The operation changes floating-point evaluation order; it does not change
the recurrence or its intended derivatives. FP64 tests compare values and
all gradients with the original independent PyTorch recurrence, and a
numerical finite-difference check verifies the custom backward.

## Build and portability

The optional kernel supports Linux CPU FP32/FP64. A local C++ compiler builds
the source once into PyTorch's extension cache. The cache identity includes
source, loader, PyTorch version, ABI and CPU flags. It uses local ISA tuning;
compiled binaries are not shipped. No Ninja package or downloads are needed.

Other platforms, unsupported dtypes/devices and missing compilers fall back
to the portable optimized path. Set `SHOEBOX_FUSED=0` to disable native
execution. Inspect `shoebox.fused_kron.native_status()` to see which path is
available. Profiling with `--backend fused` requires the native kernel, so a
silent fallback cannot be reported as a fusion measurement.

Native autograd supports first-order reverse-mode training. Use the portable
backend for higher-order differentiation. Checkpoints keep the same parameter
and state schema and can be loaded with either backend.

## Reproduce the measurement

Use the exact packaged `profile_train.bin` or the existing wide-profile file.
The training command consumes exactly 1,000 byte targets in eight updates.
Run reference/optimized/fused benchmarks in separate processes. Compilation
and initial warmup happen before the timed interval.

```bash
SHOEBOX_REQUIRE_FUSED=1 python scripts/check_all.py

python -m shoebox.profile_wide --mode benchmark --backend optimized \
  --data artifacts/cortical_wide_profile/profile_train.bin \
  --out artifacts/cortical_minkron_fusion
python -m shoebox.profile_wide --mode benchmark --backend fused \
  --data artifacts/cortical_wide_profile/profile_train.bin \
  --out artifacts/cortical_minkron_fusion
python -m shoebox.profile_wide --mode train --backend fused \
  --data artifacts/cortical_wide_profile/profile_train.bin \
  --out artifacts/cortical_minkron_fusion --targets 1000 --batch 4 --context 32
```

For operator profiles, use `--mode operators --backend optimized` and then
`--backend fused` against that same output directory/checkpoint. Neither
operator profile updates model weights. Use a new directory for another
training run: existing complete or partial logs are protected from overwrite.

The run contains the exact corpus, native-build status, complete training and
port diagnostics, endpoint checkpoint and operator profiles. Source fingerprints
include the C++ kernel as well as the Python model files. Original MinKron
attribution remains in `experiments/hat_minkron/references/` and
`THIRD_PARTY_NOTICES.md`.
