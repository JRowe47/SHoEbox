# SHoEbox

**A predictive six-region HAT-SHoE pilot with continuous messages, a 1/4/16-byte
temporal hierarchy, and a joint byte decoder.**

The current learning path is `HierarchyModel`: A1/B1 run every byte, L2/S2
consume four completed local spans, L3 consumes four completed middle spans,
and small Y combines local detail with the latest upper representations.
There are 32 experts per layer, four selected, 64-dimensional messages, and
3,050,848 total parameters. Regional scheduling is fixed until the hierarchy
earns learned regional selection. The local-only control has one HAT-SHoE.

```python
from shoebox import HierarchyModel, HierarchyConfig

model = HierarchyModel()
# observations[:, t] is the known byte; targets[:, t] is its next byte.
result = model(observations, targets=targets)
result['loss'].backward()
state = result['state'].detach()  # only at the optimization boundary
```

See [pilot architecture, training and evaluation](docs/predictive_hierarchy.md)
for corpus preparation, B8/T128 training, exact resume and the local comparison.
The optional [regional training cycle](docs/regional_training.md) alternates
upper development, lower alignment and joint learning, preserving recurrent
execution and gradients through frozen regions. Joint AdamW remains the default. See the
[matched continuation results](docs/regional_training_report.md).
The [fresh-data continuation](docs/fresh_joint_training.md) starts from the
stronger joint checkpoint, doubles the credit window to B8/T256, and trains
once through 524,288 targets from previously unused FineWeb documents.
It reaches 1,048,576 lifetime targets and improves full held-out NLL from
2.392272 to 2.236783; see the [completed phase report](docs/fresh_joint_training_report.md).
The native CPU backend reuses the AVX512 MinKron and sparse expert kernels;
`set_execution_backend('optimized')` selects portable PyTorch.

The optional [Phase A latent objective](docs/hybrid_learning.md) predicts each
upper region's next incoming message bundle with causal document-aware pairs,
training-only normalization, live diversity penalties and named Adam migration.
It adds 83,200 training parameters and leaves inference unchanged. The
[matched 262,144-target pilot](docs/hybrid_learning_report.md) scores 2.185723
nats/byte for legacy versus 2.188464 for Phase A. Legacy remains the default;
L2/S2 show useful latent prediction, while L3 remains close to shuffled futures.

## Preserved cortical expansion model

The 18-region model and measured checkpoints remain available as `CorticalModel`:
four parallel cortical pathways and one shared hippocampus accessed exclusively
through two ordinary EC regions. The six-region pilot does not activate episodic
memory; EC remains the only planned hippocampal interface when it returns.

The expansion model implements the Cortical Input Map v2. Every region uses the
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

The **14,989,597-parameter** `cortical_cpu()` preset extends learned sparsity to
regions and their allowed connections. Each stream activates **7–8 of 18 regions**
and selects incoming edges; skipped regional rows keep their state and do no
forward work. The CPU backend combines variable-length HAT sequences and active
SHoE rows, uses indexed byte transport, AVX512 expert arithmetic and OpenMP native
loops, and offers independent regional worker threads. At B4/T32, repeated
forward/backward timing improves **8.72 -> 12.19 bytes/s**, with **36% less peak
RSS**. A matched 2,048-byte training shakedown improves **8.56 -> 12.63 bytes/s**;
held-out NLL is **4.51 sparse vs 4.41 dense** after the same budget.
See [CPU configuration and commands](docs/cortical_cpu.md) and
[measurements](docs/cortical_cpu_results.json).

For longer runs, `python -m shoebox.train_cpu` adds periodic last/best checkpoints,
held-out validation, exact restart and explicit continuation from existing
weights. See [100k-run commands and checkpoint semantics](docs/cpu_training.md).

The active model has no question/answer classifiers, conflicting-query heads,
regional fact archives, retention controllers or replicated retrieval APIs.
The older seven/eleven-node implementation remains under experiments/text_regions
for historical reproduction; it is not imported by shoebox and its checkpoints
are not loaded into the new topology. The --legacy option on scripts/check_all.py
also runs those historical suites. [Historical overview](docs/legacy_project_overview.md).

SHoEbox reuses the established HAT/minKron/sparse expert primitives in
experiments/hat_minkron. Original attribution and experiment provenance remain
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [SHoE_BASELINE.md](SHoE_BASELINE.md).

The completed **100,000-byte CPU continuation** averages **61.54 training bytes/s**
at B50/T32. Held-out NLL falls **4.434 -> 3.269**, remaining at unigram level
(3.256 baseline; greedy predictions are spaces). See
[training results and checkpoint audit](docs/cpu_training_results.json).
