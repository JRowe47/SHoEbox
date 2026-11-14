# Sparse bank epochs, policy v1

Action 6 adds `sparse-epochs-v1`. It freezes **upper content-expert W, bias and
router keys** across a configured training interval. It does not freeze the
whole model. Unchanged banks remain immutable references to the committed
arrays. Changed banks own private result tiles; router-only changes own only
vectors. Two reusable arenas own detached inputs and already clipped factors.

```python
from shoebox_reference import fresh_model

model = fresh_model(
    seed=7,
    profile='epochs',
    bank_epochs=dict(epoch_bytes=4, lag_epochs=1, storage='sparse'),
    pipeline_workers=2,  # caller plus one learning worker
    prediction=dict(compact=True, rank=16, slow_stride=8),
)
```

`profile='epochs'` without an explicit mapping selects 4 bytes and zero lag.
This API default is not an empirical recommendation. The Action-6 report
records which settings qualify. Existing `reference`, `event`, and `pipeline`
profiles keep their schedules and checkpoint policies.

## Schedule

Epoch length is one of 1, 4, 8 or 16 training bytes. Worker lag is independently
zero or one **epoch**. Worker count never selects a numerical schedule. Jobs
store their derivative read version, application base, commit sequence,
training-byte interval, and task/local objective mask. Task/local scope order,
per-request clipping, factor reduction order and per-scope router normalization
are preserved. Different epochs intentionally change parameter visibility.

For full epoch k, numbered from 1, with initial bank version V0:

| Lag | Derivative reads | Application base | Result becomes visible |
| --- | --- | --- | --- |
| 0 | V(k-1) | V(k-1) | End of epoch k, after its final prediction and teacher |
| 1 | V(max(0,k-2)) | V(k-1) | End of epoch k+1, after that epoch's predictions and derivatives |

For lag 1, the prior job is committed before constructing the successor's
application base. The caller alone commits results. The learning worker owns
the immutable job and never accesses Model. No future byte target can affect
the current byte's prediction. A one-byte epoch with one-epoch lag is exactly
the historical compact pipeline's numerical schedule.

Ordinary reverse roots retain their existing scope-union lifetime; they are
consumed into immutable factors before crossing an API boundary. Roots are
not merged across bytes. Local replay retains its byte-boundary budget and
ordering, then synthetic credit remains terminal. This preserves nonlinear
scope boundaries while batching the expensive factor-to-weight operations.

## Online learning and supported combinations

Bottom A1/B1 expert learning, MinKron context, composition adapters, payload
gates, directional credit predictors, actual-next-event forecasting, recognition
and decoder learning retain their online schedules. Slow Y retains its own
explicit Action-4 stride; it is independent of the bank epoch. Paired episodic
memory retains its existing causal publication/commit rules.

One or two MinKron context heads are required. Both local event credit and full
ancestry credit are supported (`event={'learning': 'ancestry'}`). This is the
sparse alternative for ancestry learning with MinKron context. The historical
dense-context ancestry epoch remains an unchanged comparison policy; combining
its `reverse.epoch_bytes > 1` with this policy is rejected. Here the reverse
policy must be scope-union with `epoch_bytes=1`.

Compact forecasting, fast/slow Y and lightweight frames can be combined with
this policy. A new policy can be selected only on an empty stream. The
`storage='dense'` comparison option copies every upper bank but uses identical
factors and arithmetic. It exists to isolate storage cost. `fused=False` also
retains the scalar materialization oracle. Neither is a different learning rule.

## Boundaries and checkpoints

`run()` waits for worker quiescence before returning but does **not** commit a
pending version or flush an incomplete epoch. Chunk size therefore cannot
change learning. Inspection and saving also preserve pending work.

`finish_pipeline()` explicitly submits a partial factor job, commits all jobs,
and materializes visible bank arrays. Document drains finish credit before
draining banks. A deliberate reset uses the same ordered boundary. An empty
training byte still advances the interval; frozen evaluation does not.

Checkpoint version 46 requires the explicit policy. Pipeline wire extension 3
stores the finished pending result, partial epoch position, captured factors
and shared inputs, objective metadata, and schedule versions. Import validates
the complete payload before replacing state. Inputs remain valid even if their
original records have since been recycled. Restarts can switch worker counts.
Historical pipeline wire extensions 1 and 2 retain their existing loader.

Frozen evaluation uses the committed version and parks pending jobs. It never
applies them because of an evaluation target. A cold evaluation copy contains
the current committed fitted state; it does not drain its source model.

## Inspection and validation

`inspect_bank_epochs()` reports the training-byte position, partial arena,
pending byte interval and objectives, and complete checkpoint payload size.
`inspect_pipeline()` reports version numbers, selected bank tiles and reserved
arena storage. These functions read state at API boundaries; there are no new
timers, observers or cumulative profiling counters in training.

Use `tests/test_bank_epochs.py` for exact schedule, restart, chunk, worker,
storage, frozen-evaluation, document, long-stream and malformed-state checks.
`tests/run_bank_epoch_sanitizer.py` exercises every epoch/lag combination under
ASan/UBSan. `src/measure_bank_epochs.py` runs isolated rotated quality/timing
comparisons; its fixtures and measurement scope are included in each result.
