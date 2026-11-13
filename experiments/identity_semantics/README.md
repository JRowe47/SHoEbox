# SHoEbox: consistent identity and vector semantics

This package implements the second repair in the sequence: multi-source binding, consistent identity/vector semantics, relay teaching/recognition, then regional contribution checks.

## Run

Use Python 3.10+, NumPy, and g++ with C++17. The native module compiles automatically.

```python
import sys
sys.path.insert(0, "runtime")
from runtime import fresh_model, Model
m = fresh_model(kernel="semantics")
# Exact continuation of the included 1M-target checkpoint:
m = Model.load("results/full_semantics_seed1/last.npz")
```

Available kernels: `semantics` (new learned representation), `semantics_fixed` (new structural encoding with its learned residual held at zero), `binding` (prior repair), `composition` (earlier comparator), and `native` (legacy numerical kernel). The runtime's historical default remains `composition`; explicitly select `semantics`.

## Contract

- Exact 128-bit IDs bind namespace, kind, source/recipe role, child identity, order, presence and relative geometry. Absolute offsets and neural context are excluded.
- Each event carries an immutable structural descriptor and a separate learned semantic code. Parent descriptors use child structural descriptors, preventing held embeddings from accidentally relabeling a symbol.
- Sixteen regional 64-by-8 learned matrices provide bounded semantic residuals: 8,192 extra coefficients, 1,031,936 total. The local exact ID remains independent of expert choice and dictionary eviction.
- Semantic updates use saved publication features and derivatives, never the arriving target's features. Local prediction targets are detached.
- The current singleton relay cancellation and recognition behavior remains for the next repair.

## Checkpoints

New checkpoints use version 30 and composition ABI 4. The prior binding version-29/ABI-3 state cannot be imported directly. `src/convert_to_semantics.py OLD.npz NEW.npz` preserves neural weights and fixed transports while starting fresh semantics, dictionaries, recurrent eligibility and episodic memory. It requires a new destination.

## Verification and reproduction

```bash
python -m unittest discover -s tests -v
python tests/run_semantics_sanitizer.py
python src/encoding_probe.py
python src/run_semantics_study.py pilots
python src/run_semantics_study.py full
python src/semantic_contribution.py
python src/semantics_performance.py
python src/report_semantics.py
python src/render_semantics_report.py
```

The study uses the included exact Tiny Shakespeare data and independently selects each arm's learning rate on seed 3 and a reserved validation segment. Seeds 1 and 2 train for one million next-byte targets, then evaluate on a disjoint 78,771-target test segment. Reports contain the complete protocol, result, speed cost and continuation instructions.

Source/configuration hashes, logs, structured measurements and reports are tracked here. The six trained checkpoints and corpus bytes are in the separate downloadable source/checkpoint archive. Source ancestry and recovered-package verification are recorded in `provenance/`. The report's scope is this repair; no claim about repaired relay teaching or long-trained 100M checkpoints is implied.

## Restore data and trained checkpoints

The Git checkout contains the tested runtime and fixed initialization fixtures. Raw corpus bytes and trained checkpoints stay in run artifacts, following repository policy. Restore all nine exact data/checkpoint files from `SHoEbox_Identity_Semantics_Source_and_Checkpoints.zip`:

```bash
python src/restore_artifacts.py --archive /path/to/SHoEbox_Identity_Semantics_Source_and_Checkpoints.zip
```

Every artifact is verified against `ARTIFACT_MANIFEST.json` before installation. Existing files with different bytes are rejected. To reproduce training without the checkpoint archive, restore the exact corpus split with `python src/restore_artifacts.py --download-data`; `DATA_MANIFEST.json` pins the source hash and split.

The [multi-source binding report](provenance/binding_study/SHoEbox_Multisource_Binding_Report.md) records the first repair. The [identity/vector report](deliverables/SHoEbox_Identity_Semantics_Report.md) records the second. These are frozen study records; their statements about publication describe when the studies finished.
