# Selective multiscale refinement

Completed upper-scale constituents can take additional shared-expert passes
before publication. Enable the preset with `Model(loops={})`; corrected C remains
the control returned by `Model()`. This is a new computation-depth experiment,
not a replacement for the existing L2/L3 specialization investigation.

| Scale | Input bytes per event | Preset extra passes |
| --- | ---: | ---: |
| Byte | 1 | 0 |
| L1 | 4 | 1 |
| L2 | 16 | 2 |
| L3 | 64 | 2 |

Each upper budget accepts 0–4 passes. Zero skips that scale. The preset adds at
most 26 expert matrix evaluations per 64 input bytes (to the base 85), without
increasing parameter count or expert-search count. This is a matrix-operation
count, not a prediction of whole-runtime overhead.

## Cell and causal timing

Let `a` be the base content from the selected expert, `W,b` its matrix and bias,
and `g` the original routing margin gate. A single 64-value working vector is
initialized with `h[0] = a`. Each extra pass computes:

```text
proposal = tanh(a + b + g * W @ h)
h_next   = (1 - strength) * h + strength * proposal
```

The immutable anchor is the completed constituent. Strength defaults to 0.25.
The final vector replaces its content for parent composition, native state,
local objectives, and retained memory. The selected expert and gate remain
fixed throughout refinement. No additional routing search is performed.

Refinement runs only after all four ordered children exist. The parent receives
one final output. Native recurrence, MinKron/addressed-memory writes, retained
unit admission, coverage statistics and route-learning clocks advance once per
real constituent. No pass consumes a future target or publishes into earlier
byte positions. There is no extra byte-level decode or emitted thinking token.

With `tolerance > 0`, stop after an applied pass when
`sqrt(sum((h_next-h)^2)/(sum(h^2)+64e-8)) <= tolerance`.
The preset tolerance is 0.01. `tolerance=0` always exhausts the selected budget.
This is convergence-based stopping. A learned hard-constituent selector or
learned halting head is not implemented; neither is claimed by this interface.

## Learning and memory

The existing constituent epoch retains the executed loop states and proposals.
Reverse credit covers every pass, the repeated anchor injection, tied matrix
and bias, routing gate, base expert, composer and live child constituents. Hard
route selection and stopping decisions are treated as fixed for differentiation.
Updates use the existing route-local learning multiplier once per real event.
Memory and auxiliary objectives see the refined content once.

No parameters are added. At four levels, the default training tape reserves
394,668 bytes (about 385 KiB), bounded by the existing 512-byte publication
epoch. Only executed steps are serialized. Frozen inference releases the tape
and retains the 64-value working state plus ordinary stack temporaries during
each completion. `loop_stats()` reports budgets, pass counts, convergence,
budget exhaustion and tape usage. Context reset clears these stream counters.

## API and CLI

```python
from shoe_multiscale import Model

with Model(parameters=1_000_000, loops={}) as model:
    model.train(b"Observed training bytes. " * 100)
    model.save("loops.shoe")  # pending loop gradients survive exactly
    print(model.loop_stats())

with Model.load("loops.shoe") as model:
    model.finish()  # retire pending credit before changing training policy
    model.configure_loops(steps=[0, 0, 2, 4], strength=.25, tolerance=0)
    with model.inference_copy() as frozen:
        frozen.configure_loops(steps=[0, 1, 3, 4], tolerance=.02)
        print(frozen.generate("An ", length=128))
```

```bash
python -m multiscale.cli train --data input.txt --out loops.shoe \
  --loop-steps 0 1 2 2 --loop-strength .25 --loop-tolerance .01
python -m multiscale.cli train --resume loops.shoe --data next.txt --out next.shoe
python -m multiscale.cli generate --checkpoint next.shoe \
  --loop-steps 0 1 3 4 --loop-tolerance .02 --prompt 'An '
```

Resume preserves the saved policy. An explicit CLI training override retires
pending credit before applying the requested policy. Inference overrides operate
on a frozen copy. Historical backends and semantic-lace reassignment are not
supported by the loop feature. A/B/C/D routing on the current backend works with
semantic laces disabled. The byte path always has zero extra passes.

## Checkpoints and verification

Version 18 wraps an unchanged prior-format payload and appends the loop policy,
statistics and bounded pending tape. Loading old checkpoints keeps loops off.
Explicit all-zero budgets produce the identical base payload and training
trajectory as corrected C. Existing historical checkpoint formats remain valid.

`python -m multiscale.tests.run_all` covers independent double-precision
derivative oracles, integrated language credit, completion clocks, bounded
storage, causal target handling, corrupted tapes, CLI controls, frozen inference,
and exact continuation at byte/chunk/publication boundaries. On ptrace-managed
hosts, `--no-leak-check` explicitly disables LeakSanitizer only.

## Research mapping

[SpiralFormer](https://arxiv.org/abs/2602.11698) applies recurrence at different
sequence resolutions and explicitly preserves causal chunk timing. Here,
SHoEbox's existing completed 4/16/64-byte constituents supply those resolutions.
[Think-at-Hard](https://proceedings.mlr.press/v306/fu26t.html) uses a learned
decider to select additional latent iterations. The present implementation
establishes bounded refinement and convergence stopping; learned selection is a
separate subsequent change. These are SHoEbox-specific cells, not reproductions
of either paper's Transformer architecture.
