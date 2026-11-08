# SHoEbox

**18 ordinary HAT-SHoE regions, four parallel cortical pathways, and one shared
hippocampus accessed exclusively through two ordinary EC regions.**

The active model implements the Cortical Input Map v2. Every region uses the
same framed HAT, three sparse expert layers, and ordinary minKron state.
Source sets, sequence spans, port widths and feedback connections define the
intended specialization. Names do not select semantic algorithms.

~~~bash
python -m pip install -r requirements.txt
python scripts/check_all.py
python -m shoebox.shakedown --out artifacts/cortical_skeleton
~~~

The default command runs 24 small CPU learning updates and writes initial.pt,
last.pt, results.json, trace.json, and the complete topology.json.
The fixed fixture tests optimization; it is not a language benchmark.

~~~bash
# Train ordinary byte streams, carrying state across chunks.
python -m shoebox.shakedown --data your_text.txt --context 8 --batch 2 \
  --steps 100 --out artifacts/cortical_stream

# Exact continuation; preserve the original output directory.
python -m shoebox.shakedown --resume artifacts/cortical_stream/last.pt \
  --data your_text.txt --context 8 --batch 2 --steps 20 \
  --out artifacts/cortical_stream_continued
~~~

~~~python
import torch
from shoebox import CorticalModel, SkeletonConfig

model = CorticalModel(SkeletonConfig())
observations = torch.tensor([[ord(c) for c in "red "]])
result = model(observations)
state = result["state"].detach()
next_byte_logits = result["logits"][:, -1]
~~~

- Four pathways: language L2/L3, content S2/S3, relations R2/R3, operations F2/F3.
- E4/F4/V4/M4/G5 compose situations, operations, relevance and broader context.
- EC-C and EC-X are ordinary regions with additional framed hippocampal input.
- Y receives language, operations, local content, EC content and preceding output.
- Complete learned ports preserve source and frame order: 1/2/4 bytes in the
  original presets, 4/8/16 bytes in `cortical_wide()`.
- Feedback reads a previous committed round. A1/B1 ingest each observation once.
- Two recent transport occurrences are retained per port; durable episodes exist
  only in the shared hippocampus. Repeated rounds revise the current publication.
- All regions own separate state. B1 starts with a slower learned time constant.

See [implementation and shakedown](docs/cortical_skeleton.md) and
[current status](STATUS.md).

For the **9,982,290-parameter** version and the one-pass **100,000-byte-target
FineWeb** run, see [configuration, data and training](docs/cortical_fineweb.md).
Use `from shoebox import cortical_10m` to construct the larger configuration.

The **14,976,882-parameter** `cortical_wide()` preset concentrates capacity at
orders 1 and 2, widens internal ports to **4/8/16 bytes**, and selects **4 of 32**
experts per regional layer. Its completed **1,000-byte FineWeb profile** uses a
**32-observation gradient horizon**, logs port entropy and complete-frame
diversity, and compares reference with optimized execution. See
[configuration and profiling commands](docs/cortical_wide_profile.md).

The **fused MinKron CPU backend** uses a native recurrence and analytic
backward, replacing pairwise temporal intermediates in the HAT and fusing
the regional single-step recurrence. The matched 32-observation benchmark
is **25.2% faster** than the previous optimized path, with **35.4% less peak
RSS**. Enable it with `model.set_execution_backend('fused')`; see
[implementation, checks and commands](docs/cortical_minkron_fusion.md).

The fused backend also reads **selected expert weights in place** and combines
their gradients in bounded groups of eight calls. Against fused MinKron with
the previous expert path, three matched B4/T32 pairs show **6.5% higher median
throughput**, **23.8% less peak RSS**, and **41.0% less saved storage**. Dense
AdamW and temporal gradients retain their behavior. See
[expert execution and reproduction](docs/cortical_expert_fusion.md).

The active model has no question/answer classifiers, conflicting-query heads,
regional fact archives, retention controllers or replicated retrieval APIs.
The older seven/eleven-node implementation remains under experiments/text_regions
for historical reproduction; it is not imported by shoebox and its checkpoints
are not loaded into the new topology. The --legacy option on scripts/check_all.py
also runs those historical suites. [Historical overview](docs/legacy_project_overview.md).

SHoEbox reuses the established HAT/minKron/sparse expert primitives in
experiments/hat_minkron. Original attribution and experiment provenance remain
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [SHoE_BASELINE.md](SHoE_BASELINE.md).
