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
- Complete 1-, 2- and 4-byte learned ports preserve source and frame order.
- Feedback reads a previous committed round. A1/B1 ingest each observation once.
- Two recent transport occurrences are retained per port; durable episodes exist
  only in the shared hippocampus. Repeated rounds revise the current publication.
- All regions own separate state. B1 starts with a slower learned time constant.

See [implementation and shakedown](docs/cortical_skeleton.md) and
[current status](STATUS.md).

The active model has no question/answer classifiers, conflicting-query heads,
regional fact archives, retention controllers or replicated retrieval APIs.
The older seven/eleven-node implementation remains under experiments/text_regions
for historical reproduction; it is not imported by shoebox and its checkpoints
are not loaded into the new topology. The --legacy option on scripts/check_all.py
also runs those historical suites. [Historical overview](docs/legacy_project_overview.md).

SHoEbox reuses the established HAT/minKron/sparse expert primitives in
experiments/hat_minkron. Original attribution and experiment provenance remain
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [SHoE_BASELINE.md](SHoE_BASELINE.md).
