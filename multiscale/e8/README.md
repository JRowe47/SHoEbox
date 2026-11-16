# Public E8 policy, format e8-public-1

The arithmetic is the qualified EXP01-E8 policy, copied without numerical edits
from `experiments/joint_cold_gate`. The readout header differs only in checkpoint
magic. `backend.py` applies the original checked hooks to the unchanged native
core; missing or repeated anchors abort the build. Production imports no
experiment modules. `public.cpp` supplies lifecycle validation and the public
C ABI. Regression tests pin the copied arithmetic and original C ABI exactly.

```python
from shoe_multiscale import Model

with Model(policy="e8", parameters=500_000, seed=7) as model:
    loss = model.train(b"Generic UTF-8 training bytes.")
    print(model.stats())
    model.save("model.shoe")
    output = model.generate(b"A prompt", length=64)
```

E8 uses four minimal corrected-C scales and deducts 7,670 readout weights from
the nominal expert budget. The local applicability intercept is `-log(255)`.
There are no new persistent sequence slots or optimizer moments. Full-byte
inference always produces 256 normalized log probabilities.

## Constructor and checkpoint compatibility

- `Model(policy="e8")` explicitly selects E8.
- `Model(policy="v19")` and `Model(core="minimal")` retain the original v19 core.
- Explicit architectural arguments retain their historical behavior. For
  example, `core="full"` selects v10, and explicit C/D arms retain v9 unless
  another historical policy is selected.
- `Model()` / `fresh_model()` with only parameters/seed use E8 after the completed integration qualification.
  `multiscale.model.Model` remains the low-level historical constructor.
- Loaded checkpoints select their saved policy, irrespective of the default.
  E7 and every research checkpoint require their original loader or the explicit
  E8 importer below. No automatic v19-to-E8 conversion exists.

The public E8 envelope is `SHOEME01`, SHA-256, then the native public payload
(magic `0x3850454f4853`, format 1). Its nested core remains v19. Stats expose
`checkpoint_version="e8-public-1"` and `core_checkpoint_version=19`.
Cloning, save/load, partial groups, pending credit and frozen flags persist.
`observe()` remains inference-only. `run()` on a frozen copy scores byte
objectives; it rejects local labels. `clone()` preserves frozen status.
`evaluate()` and `generate()` leave the source model unchanged. External
`bind/lookup/payload` remains a separate explicit API; E8 inference never uses
it as an automatic task solver.

## Local labels and generic objectives

`run(inputs, targets, *, mask=None, local_labels=None)` retains the historical
mean-loss return value. `run_objectives` returns the exact `loss_sum`, number
of objective `targets`, and charged `input_bytes` for accounting.

`local_labels` is an optional int32-compatible array shaped `(n,6)`, describing
only the observed prefix ending at each input byte:

| Column | Meaning | Valid values |
| --- | --- | --- |
| 0 | Observed query key | -1 absent, or byte 0–255 |
| 1–4 | Four completed-window key/value phase labels | -1 absent, or 256 × key + value |
| 5 | Observed answer request | 0/1; -1 is historical absent/negative |

The public API does not derive these labels. Fixture helpers remain in the
research harness. Labels contain no selected record or future answer. Masked
next-byte objectives and local updates are separate; padding and masked bytes
still consume state and CPU. Generic language/copy/reconstruction must omit
labels. Generic byte loss then freezes local extraction/applicability weights
while preserving E8's native mixture-responsibility credit.

`reconstruction_features()` and `apply_reconstruction_credit(gradient)` expose
the qualified separately budgeted final-L3 decoder path. Consume its last byte
with a zero-mask `run_objectives` before requesting the completed L3 state.
No external decoder is silently initialized, budgeted or checkpointed for you.

## Explicit research checkpoint import

```python
from pathlib import Path
from shoe_multiscale import Model

model, receipt = Model.import_e8_research(Path("local_gate.qr").read_bytes())
with model:
    model.save("public.shoe")
print(receipt)  # original and converted SHA-256; exact payload parity
```

For a checkpoint plus its exact outer data cursor:

```bash
python -m multiscale.e8.import_checkpoint local_gate.qr public.shoe
```

This writes the public checkpoint, unchanged `public.cursor.json`, and
`public.import.json` containing both hashes. It refuses existing outputs.
Reconstruction containers retain their exact `decoder.npz` member, changing
`adapter.qr` to `model.shoe`. Only minimal joint E8 kind 4 is accepted; E7,
generic-gate controls and full-core research checkpoints are rejected.

The cursor is essential: it preserves pending episode/document positions,
remaining shuffled order, both RNG states and scheduled finish events.
`experiments/e8_public/bridge.py` runs the exact historical drivers through the
public API. The importer itself performs no training, reset, finish or replay.

Primary grounding remains Miller et al. (EMNLP 2016),
https://aclanthology.org/D16-1147/, and See et al. (ACL 2017 §2.2),
https://aclanthology.org/P17-1099/. The 1/256 cold prior is the registered E8
design choice. This implementation qualifies explicitly fixture-supervised
binding and generic fallback; it does not claim generic byte loss learns the
local heads. Opened audits must never guide new numerical revisions.
