# SHoEbox

## Current reference design

The default CPU runtime now enables the implemented reference design, including
all published composition, binding, semantic, relay, message and completion
repairs. Architectural completeness is the development priority; performance
optimization follows. Use the repository entry point:

```python
from shoebox_reference import fresh_model, Model
model = fresh_model()
print(model.capabilities())
```

This selects `completion` with recursive learned exports, credited feedback,
three-hop lace and retained-input credit, learned bottom state, task routing,
EMA/revival, both EC recall channels, key-trace mixing, DFA, A1 residual and local
innovation objectives. Temporal credit covers all 64 coordinates every byte;
third-hop credit is unsampled. Historical kernels require an explicit selection.

Primary occurrence clocks are now propagated separately from all-source content
coverage. Held side context can no longer force an otherwise adjacent learned
sequence to split. Semantic eligibility grows when all records remain live, so
longer completed units do not lose ancestry to the old fixed pool limit. New
checkpoints are version 34 / composition ABI 8.
Version-33 completion checkpoints require explicit fresh-stream conversion;
versions 31 and 32 retain their original kernels when loaded.

[Current implementation and usage](experiments/identity_semantics/README.md) |
[Review and verification](experiments/identity_semantics/deliverables/SHoEbox_Reference_Defaults.md).
The sections below record earlier studies; their historical recommendations do
not override the current default.

## Learned-message and feedback repair

The upper-region repair is available in
[experiments/identity_semantics](experiments/identity_semantics/README.md) as
`fresh_model(kernel="messages")`. Parents consume learned constituent residuals;
bounded delayed feedback influences activation and sparse expert selection.
Semantic adapters and neural experts receive credit through separate saved
producer records. IDs, binding and relay forecasts remain intact.

199 repository checks, 17 additional checks at 64 experts, and native
ASan/UBSan pass. Two matched 1M-target runs produce mean reserved-test NLL
**1.944734**, compared with **1.940301** for `relays`. The repair removes the
mechanical restrictions but does not improve average test loss in this study;
`relays` was the production recommendation for that study. The new runtime is selectable
for continued message/credit work, with version-32 / ABI-6 checkpoints and an
explicit fresh-stream converter.

Same-core median training throughput is **3,275 -> 2,215 bytes/s** (32.4%
lower). All 120 regional interventions complete; the 100 controls requiring
stable content events preserve exact identity/provenance traces.

[Implementation and measurements](experiments/identity_semantics/deliverables/SHoEbox_Learned_Message_Fix.md) |
[PDF](experiments/identity_semantics/deliverables/SHoEbox_Learned_Message_Fix.pdf).

## Regional contribution checks

The fourth composition-repair step is implemented in
[experiments/identity_semantics](experiments/identity_semantics/README.md).
Seven independent evaluation controls separate neural publication, learned
expert transform, content descriptor, singleton relay context, retained input,
local semantic residual, and whole-source removal. Memory evaluation explicitly
selects cold/warm stores and frozen/adaptive admission.

Two existing 1M-parameter checkpoints reproduce their published losses exactly.
226 full-test interventions and eight production-parity comparisons complete.
All 190 interventions that should preserve content events preserve their full
identity/provenance traces and execution counts. Removing learned expert
transforms increases mean NLL by **0.038686 for L3** and **0.023036 for F4**;
E4/V4/G5 remain weak despite realized teaching and credit.

That study used `fresh_model(kernel="relays")`; the optional diagnostic
kernel compiles its hooks out of production. Neural parameters and checkpoint
version 31 / ABI 5 are unchanged. Test models and intermediate runs are excluded.

[Contribution report](experiments/identity_semantics/deliverables/SHoEbox_Contribution_Checks_Report.md) |
[PDF](experiments/identity_semantics/deliverables/SHoEbox_Contribution_Checks_Report.pdf) |
[Full regional measurements](experiments/identity_semantics/deliverables/SHoEbox_Contribution_Checks.csv).

## Production relay teaching and recognition runtime

Use `fresh_model(kernel="relays")` from
[experiments/identity_semantics](experiments/identity_semantics/README.md).
A singleton can teach the pending forecast without canceling it or teaching
against the relayed public activation. Updates use the saved actual producer.
Relays preserve recognition, confidence, intrinsic structure and provenance;
separate metadata identifies the receiver's transport action.

All 100 package checks and ASan/UBSan pass. In matched two-seed million-byte
runs, mean test NLL improves **1.946529 -> 1.940301**. Training throughput is
**5,243 -> 5,142 bytes/s** (1.9% lower).

The 1M configuration remains 1,031,936 parameters. Version-31 / ABI-5
checkpoints preserve pending teaching exactly. Explicit fitted conversion from
version-30 semantics is available at a fresh stream boundary. Earlier kernels
remain available, and the historical default at that time was `composition`.

[Relay report](experiments/identity_semantics/deliverables/SHoEbox_Relay_Preservation_Report.md) |
[PDF](experiments/identity_semantics/deliverables/SHoEbox_Relay_Preservation_Report.pdf) |
[Measurements](experiments/identity_semantics/deliverables/SHoEbox_Relay_Preservation_Results.json).

## Previous multi-source binding and identity/vector study

The preceding binding/semantics repair is in
[experiments/identity_semantics](experiments/identity_semantics/README.md).
It includes explicit multi-source constituent roles, consistent structural IDs
and descriptors, and learned regional semantic residuals. Required distinctions
reach routing and local teaching; exact identities stay separate from context,
expert selection and learned-vector updates.

Two fresh 1M-model seeds, each trained on one million Tiny Shakespeare byte
targets, improve mean reserved-test NLL **1.959272 -> 1.946529 nats/byte**
against the preceding binding runtime. One seed is slightly worse and one
improves. Same-core median throughput is **6,218 -> 5,284 training bytes/s**
(15.0% lower), with **1,023,744 -> 1,031,936 parameters** (0.80% more).

```bash
cd experiments/identity_semantics
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tests/run_semantics_sanitizer.py
```

Select `fresh_model(kernel="semantics")`, or use `Model.load()` for a version-30
/ ABI-4 checkpoint. The fresh-model default at that time was `composition`.
All **62 checks** and ASan/UBSan pass. Data and trained checkpoints have
[verified restoration instructions](experiments/identity_semantics/README.md#restore-data-and-trained-checkpoints).

[Report](experiments/identity_semantics/deliverables/SHoEbox_Identity_Semantics_Report.md) |
[PDF](experiments/identity_semantics/deliverables/SHoEbox_Identity_Semantics_Report.pdf) |
[Curves](experiments/identity_semantics/deliverables/SHoEbox_Identity_Semantics_Curves.png).

The relay teaching/recognition repair and regional contribution checks are now
available above, with content and relay transport separated from local learning.
The audit narrowed the earlier L3 lesion interpretation: whole-source removal
measured transport as well as learned computation.


## Earlier composition and credit runtime

The repaired native runtime is in
[experiments/composition_repair](experiments/composition_repair/README.md).
It separates reusable constituent identities from compute experts and contextual
state, learns completion boundaries, consumes each child event once, and assigns
credit through the upper regions' retained input. Checkpoint version 28 includes
the constituent dictionaries, pending events and temporal credit state.

Two matched fresh 1M-parameter models, each trained on 1M Tiny Shakespeare byte
targets, reduce mean held-out loss from **2.060284 to 2.035142 nats/byte**.
Same-core interleaved timing measures **22,564 -> 17,098 training bytes/s**:
1.22% lower loss at 24.2% lower throughput. The 19 structural and numerical checks,
address/undefined-behavior checks, and clean-build exact-resume check pass.
Removing L3 hurts prediction substantially more after repair; E4/G5 still show
little contribution after this short run.

```bash
cd experiments/composition_repair
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tests/run_sanitizer.py
```

This implementation uses width 64 and uniform expert banks. The 1.02M, 4.99M and
100.20M configurations build and run; the full matched language comparisons use
the 1.02M model. The directory includes tested source, fixed initialization
fixtures, raw measurements, the [report](experiments/composition_repair/deliverables/SHoEbox_Composition_Repair_Report.md)
and [PDF](experiments/composition_repair/deliverables/SHoEbox_Composition_Repair_Report.pdf).
Corpus and checkpoint restoration instructions and checksums accompany it.
Legacy weights require explicit conversion to the new message semantics.

## Earlier width-scaling runtime

The width-aware native runtime and matched 25M-byte comparison are in
[experiments/scaling_cpu](experiments/scaling_cpu/README.md). It supports widths
64, 128 and 256, packed per-region expert banks, multiword feature masks, exact
legacy checkpoint replay, version-20 checkpoints, and guarded resume.

The three comparison arms are fresh 4.99M / 19.64M / 19.57M models, two seeds
each on identical data. Nine 500K-byte rate pilots selected peak LR **0.008**;
the six full 25M runs have not started. The proposed **99,727,616-parameter**
configuration is implemented and smoke-tested. Eleven publication checks pass.

Measured single-core FP32 training on an AMD EPYC 9V74, with 16,384 warmup
and 250,000 timed byte targets per model, sequentially on one pinned CPU:

| Model | Parameters | Training bytes/s | Projected 25M training time |
|---|---:|---:|---:|
| 5M control (D=64, E=64) | 4,990,720 | 20,526 | 20.3 min |
| 20M bank (D=64, E=256) | 19,638,016 | 18,756 | 22.2 min |
| 20M width (D=128, E=64) | 19,565,824 | 6,426 | 64.8 min |
| 99.73M (D=256, regional banks) | 99,727,616 | 1,887 | 220.8 min |

Timing excludes initialization, compilation, validation and checkpointing.
The 25M times are projections, not completed runs. See the
[raw benchmark](experiments/scaling_cpu/reports/throughput.json) and
[reproduction command](experiments/scaling_cpu/README.md#throughput).

```bash
cd experiments/scaling_cpu
python -m pip install -r requirements-data.txt
python restore_data.py
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python comparison.py launch --dry-run
python comparison.py launch --workers 3
```

The repository includes source, tests, fixed fixtures, calibrated settings and
a compressed manifest for exact data restoration. Trained checkpoints and raw
corpus binaries stay in run artifacts. Resume with `--resume`; run the synthetic
checks with `python -m unittest discover -s tests -v` from this directory.

## Previous fixed-width online CPU runtime

The preceding fixed-width CPU-first 18-region implementation is in
[experiments/online_cpu](experiments/online_cpu/README.md). It includes the selected
centered routing and decoder policy, learned byte state, temporal lace credit,
the EC/hippocampus interface, exact SIMD optimizations, and asynchronous
validation/checkpoint snapshots.

The architecture-preserving profile improved training from **14,238 to 19,721
bytes/s** for the 1M model and **12,606 to 19,180 bytes/s** for the 5M model.
A second core for snapshot work reduced measured end-to-end elapsed time by
**33–37%**. See the [full report](experiments/online_cpu/reports/SHoEbox_CPU_Profile_Report.md)
for protocols and exact-equivalence results.

```bash
python -m pip install -r experiments/online_cpu/requirements.txt
SHOEBOX_EXPERTS=12 python experiments/online_cpu/check_packaging.py
SHOEBOX_EXPERTS=64 python experiments/online_cpu/check_packaging.py
```

## Earlier PyTorch implementation
**18 ordinary HAT-SHoE regions, four parallel cortical pathways, and one shared
hippocampus accessed exclusively through two ordinary EC regions.**

The earlier PyTorch model implements the Cortical Input Map v2. Every region uses the
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




