# SHoEbox: relay teaching, recognition and identity semantics

This package now implements the first three repairs: multi-source binding, consistent identity/vector semantics, and preserved teaching/recognition through relays. Regional neural contribution checks are next. The `relays` kernel shares the tested runtime with the earlier comparison kernels.

## Run

Use Python 3.10+, NumPy, and g++ with C++17. The native module compiles automatically.

```python
import sys
sys.path.insert(0, "runtime")
from runtime import fresh_model, Model
m = fresh_model(kernel="relays")
# Exact continuation of a repaired checkpoint you save:
# m = Model.load("my_relay_checkpoint.npz")
```

Available kernels: `relays` (current relay repair), `semantics` (previous learned representation), `semantics_fixed` (new structural encoding with its learned residual held at zero), `binding` (prior repair), `composition` (earlier comparator), and `native` (legacy numerical kernel). The runtime's historical default remains `composition`; explicitly select `relays`.

## Contract

- Exact 128-bit IDs bind namespace, kind, source/recipe role, child identity, order, presence and relative geometry. Absolute offsets and neural context are excluded.
- Each event carries an immutable structural descriptor and a separate learned semantic code. Parent descriptors use child structural descriptors, preventing held embeddings from accidentally relabeling a symbol.
- Sixteen regional 64-by-8 learned matrices provide bounded semantic residuals: 8,192 extra coefficients, 1,031,936 total. The local exact ID remains independent of expert choice and dictionary eviction.
- Semantic updates use saved publication features and derivatives, never the arriving target's features. Local prediction targets are detached.
- Relays preserve intrinsic structure, recognition, confidence and actual-producer provenance. `local_arity` and `relayed` distinguish the receiving action from the inherited content.
- A separate forecast record teaches the saved producer on the next observed regional content input, including a completed singleton. Idle ticks do not consume forecasts. Evaluation/frozen targets retire forecasts without weight updates.
- `relay_teaching_status()` reports scheduling, consumption, expiry and pending provenance; `relay_event_status()` reports local relay metadata, confidence and origin.

## Checkpoints

Repaired checkpoints use version 31 / composition ABI 5. `Model.load()` resumes pending forecasts and all stream state exactly. Older kernels retain their original checkpoint formats.

`python src/convert_to_relays.py OLD.npz NEW.npz` explicitly converts version-30 learned semantics at a fresh stream boundary. It preserves neural/semantic weights, dictionaries, fixed transports, policies and fitted preprocessing means. It resets recurrent context, pending forecasts, eligibility, online adaptation state and hippocampal associations. The old event blob is rejected rather than reinterpreted. The destination must be new.

For older version-29 binding checkpoints, `src/convert_to_semantics.py` remains available as the prior conversion step.

## Verification and reproduction

```bash
python -m unittest discover -s tests -v
python tests/run_relays_sanitizer.py
python src/run_relays_study.py pilots
python src/run_relays_study.py full
python src/relays_performance.py
python src/report_relays.py
```

The relay suite is discovered by the existing `scripts/check_all.py` integration. The study uses the exact Tiny Shakespeare data and validation-only rate selection on seed 3. Seeds 1 and 2 each train for one million next-byte targets, then evaluate on a disjoint 78,771-target test segment. The [relay report](deliverables/SHoEbox_Relay_Preservation_Report.md) and [results](deliverables/SHoEbox_Relay_Preservation_Results.json) contain the measured loss, throughput and forecast accounting.

Test models and intermediate relay runs are excluded from Git. Small results and final reports are included. `SOURCE_MANIFEST.json` identifies the current source; historical comparison records remain in `provenance/` and the earlier reports. Earlier study scripts remain available for reproduction.

## Restore data and trained checkpoints

The Git checkout contains the tested runtime and fixed initialization fixtures. Raw corpus bytes and trained checkpoints stay in run artifacts, following repository policy. Restore all nine exact data/checkpoint files from `SHoEbox_Identity_Semantics_Source_and_Checkpoints.zip`:

```bash
python src/restore_artifacts.py --archive /path/to/SHoEbox_Identity_Semantics_Source_and_Checkpoints.zip
```

Every artifact is verified against `ARTIFACT_MANIFEST.json` before installation. Existing files with different bytes are rejected. To reproduce training without the checkpoint archive, restore the exact corpus split with `python src/restore_artifacts.py --download-data`; `DATA_MANIFEST.json` pins the source hash and split.

The [multi-source binding report](provenance/binding_study/SHoEbox_Multisource_Binding_Report.md) records the first repair. The [identity/vector report](deliverables/SHoEbox_Identity_Semantics_Report.md) records the second. The [relay report](deliverables/SHoEbox_Relay_Preservation_Report.md) records the third. These are frozen study records; their statements about publication describe when the studies finished.
