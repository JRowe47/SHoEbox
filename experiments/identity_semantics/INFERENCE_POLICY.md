# Forward-only runtime: forward-only-v1

Action 7 adds explicit inference to the current completion runtime. All 18
regions, content encoders, primary occurrence clocks, source/role binding,
MinKron recurrence, delayed feedback and paired episodic retrieval remain live.
The training defaults and selected Action-6 profile remain available.

```python
import numpy as np
from shoebox_reference import fresh_model, Model

m = fresh_model(profile='epochs', pipeline_workers=2,
    bank_epochs=dict(epoch_bytes=4, lag_epochs=1),
    prediction=dict(compact=True, rank=16, slow_stride=8))
# Train or load as usual, then make an independent inference stream:
i = m.inference_copy(memory='warm_frozen')
x = np.frombuffer(b'The river', dtype=np.uint8)
i.run(x)  # only mode='eval' is accepted
rng = np.random.default_rng(7)
byte = i.draw(rng, temperature=.8)
i.run([byte])
i.save('inference.npz')
i = Model.load('inference.npz')
# Training is an explicit new-context transition:
i.resume_training(reset_context=True)
```

`fresh_model(profile='inference')` creates the full reference in inference mode.
`fresh_model(profile='event', inference=True)` and the corresponding epochs
configuration select another established forward policy. `run(mode='eval')`
on an ordinary model retains training eligibility as before; selecting the
forward-only runtime is an explicit lifetime decision.

## Forward state and eligibility

Forward publications own their pattern-class assignment and role-aware
prediction vector. Lightweight intermediate frames additionally own their raw
role sum until completion. These values used to reside in derivative records;
they now survive independently without retaining the child derivative graph.
Published structural identities, occurrence metadata and payload/context
separation are unchanged. Semantic slab indices are internal credit handles;
new inference publications use index -1, while exact structural IDs, semantic
serials, sources and clocks retain their meanings.

Inference does not allocate semantic derivative slabs, content-expert
eligibility, MinKron eligibility, neural/temporal archives, event replay or
reverse plans. It skips training collection, bottom temporal Jacobians and
learning-worker preparation. The large semantic slab and gradient workspaces
are allocated lazily for ordinary training. Historical composition and neural
wire layouts remain unchanged. Fixed legacy compatibility buffers still impose
a memory floor; the report separates process RSS from live derivative storage.

## Entry boundary

`enter_inference()` is in-place and preserves the observed prefix. It calls
`finish_training()`, which commits already-materialized slow-Y gradients,
reverse epochs, directional replay and sparse bank jobs in that order, with a
final reverse/bank drain. Workers are joined and their arenas released. This
boundary can update weights; compare against an ordinary model after the same
`finish_training()` boundary.

No artificial future event is supplied to a pending next-event teacher. Such
teachers, unscheduled retained credit and their eligibility are retired at this
explicit end-of-training boundary. `finish_training()` alone does not retire
those records and does not drain a document or advance byte clocks.

`inference_copy(memory='continue')` performs the same transition on an independent
snapshot, leaving the original model and its incomplete jobs untouched. The
four fresh-stream policies below instead use the existing frozen-evaluation
contract and its visible parameter version; original pending training remains
parked in the original model.

| Memory policy | Initial episodes | Causal admission during inference |
| --- | --- | --- |
| `cold_adaptive` | Empty | Enabled |
| `cold_frozen` | Empty | Disabled |
| `warm_adaptive` | Copied from source | Enabled |
| `warm_frozen` | Copied from source | Disabled |

Adaptive admission is forward episodic memory, not parameter training. Queries,
recurrent context, counts and clocks evolve in all four policies. Query values
cannot depend on future targets. Evaluation and autoregressive generation are
qualified separately.

## Checkpoints and return to training

Version 47 requires `forward-only-v1`, `drain-materialized-v1` and
`reset-context-v1` policy markers. Its dedicated forward-state blob contains
current/pending publications, pattern/frame metadata, recurrent state, clocks,
slow-Y values and memory-write controls. Fitted composition, pattern, context
and prediction parameters and the paired episodic store are separate sections.
Derivative archives and pending learning jobs are absent. Imports validate the
mode, dimensions, lengths and forward references before installing stream state.
Irregular chunks, snapshots and save/load preserve inference continuation.
The sampler RNG remains caller-owned; preserve it separately when reproducing
a sampled continuation.

Training calls are rejected by both the Python and native run/document APIs.
`reset()` and `reset_context()` alone leave inference mode active. Returning to
training requires `resume_training(reset_context=True)`: it clears the prefix,
starts fresh eligibility and bank/reverse epoch clocks, and reconstructs bottom
prototype accumulators from fitted parameters. Fitted parameters, routing
counts, learned dictionaries and the episodic store are preserved. Memory-write
policy is preserved; use `freeze_writes(False)` explicitly if needed. No attempt
is made to reconstruct credit for earlier inference bytes. Learning workers may
be enabled again after this reset.

Legacy training versions 40, 45 and 46 import bit-for-bit. Cross-build
continuation is checked for numerical drift as well as exact identities and
clocks; within this build, frozen versus forward-only outputs are bitwise equal.

## Inspection and reproduction

`inspect_inference()` derives storage inventories at API boundaries. No timers,
profiling counters or test harnesses were added to the byte loop. `capabilities()`
reports derivative learning inactive in inference while retaining saved
training policies for an explicit reset back to training.

```sh
SHOEBOX_EXPERTS=12 python -m unittest discover \
  -s experiments/identity_semantics/tests -p test_inference.py -v
ASAN_OPTIONS=detect_leaks=0 SHOEBOX_EXPERTS=12 python \
  experiments/identity_semantics/tests/run_inference_sanitizer.py
SHOEBOX_EXPERTS=12 python experiments/identity_semantics/src/measure_inference.py \
  --output experiments/identity_semantics/results/action7_1m_measurements.json
python experiments/identity_semantics/tests/check_inference_legacy.py \
  --baseline /path/to/checkout-at-085e0a6
```
