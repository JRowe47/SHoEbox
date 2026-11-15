# Multiscale SHoE

This is the separate implementation requested in `DESIGN.md`. It imports none
of the cortical reference's scheduler, publication graph, retained ancestry,
replay machinery or checkpoint compatibility code. `shoebox_reference` remains
the historical comparator; `shoe_multiscale` is the new public interface.

## Quick start

Requirements: Python 3.10+, NumPy, a C++17 compiler, and a CPU. The Python API
builds the native shared library into a temporary cache on first use. Set `CXX`
to choose the compiler or `SHOE_MULTISCALE_CACHE` to choose its cache location.
No Torch dependency or GPU is required.

```python
from shoe_multiscale import Model, fresh_model

model = fresh_model(levels=4, parameters=1_000_000, seed=7)
model.configure(learning_rate=.08, head_rate=.16,
                auxiliary_weight=.1, route_rate=.002)
model.train(b"A complete observed training sequence.\n")

# Explicit x/y pairs avoid losing a next-byte target across Python call edges.
model.run(b"abc", b"bcd")
model.save("training.shoe")        # retains every unfinished local block
resumed = Model.load("training.shoe")

loss = model.evaluate(b"A disjoint held-out sequence.\n")
sample = model.generate("A ", length=256, temperature=.8)
print(model.stats(), loss, sample.decode("utf-8", "replace"))
model.close()
resumed.close()
```

Command-line training, generation and inspection:

```bash
python -m multiscale.cli train --data input.txt --targets 1000000 --out model.shoe
python -m multiscale.cli generate --checkpoint model.shoe --prompt 'ROMEO:'
python -m multiscale.cli inspect --checkpoint model.shoe
```

`--resume model.shoe` restores the stream exactly. Supply the next input/target
segment, not a replay of the previous segment. CLI training does not reserve a
validation split; the benchmark command below does. Weights and datasets stay
outside git. Existing cortical checkpoints cannot be loaded into this runtime.

## Executed architecture

| Scale | Bytes/event | 1M bank size | Responsibility |
| --- | ---: | ---: | --- |
| Fast | 1 | 104 | Byte transform, immediate prediction and context |
| Level 1 | 4 | 52 | Four ordered completed byte contents |
| Level 2 | 16 | 31 | Four ordered Level-1 contents |
| Level 3 | 64 | 21 | Four ordered Level-2 contents |

All widths are 64. `levels=1` and `levels=2` redistribute the same requested
parameter budget over their banks. Parameters are rounded down by whole experts;
the default four-scale model has 997,183 fitted scalars. Every selected expert
has a 64x64 matrix, bias and learned routing prototype. Shared fitted parameters
include embeddings, recurrence/read maps, slow composition/conditioning maps,
native prediction heads, addressed-memory queries/read maps, constituent
reconstruction heads, and the shared 255-node binary byte decoder.
Derived router centroids and stored episodes are not fitted parameters.

The ordered adapter uses distinct invertible coordinate permutations and
learned coordinate weights for each child role, plus child lengths and an input
bias. Its expert produces canonical content using those observed inputs only.
It never uses surrounding recurrent state or recalled records for routing or
canonical content. The content feeds a separate cheap context update:

```
content = tanh(input + expert_bias + margin_gate * W_selected @ input)
state = tanh(content + recurrent @ previous_state + read_map @ memory_read
             + addressed_read_map @ addressed_memory_read)  # slow scales only
decoder_input = tanh(fast_state * (1 + cached_gain) + cached_bias)
```

The compact memory is an 8x8 MinKron-style normalized delta recurrence with
fixed signed pooling, read before write, decay .98 and write strength .5.
Its read and write both receive exact derivatives inside the local block.
The expert transform, normalized prototype attraction and top-two margin gate
reuse the reference arithmetic, reimplemented without its regional lifecycle.

## Stable addresses, native clocks and local reconstruction

New models use `memory_profile="addressed"` by default. Each slow scale adds a
32-by-16 associative matrix; the three active matrices occupy 6 KiB in float32.
The original 8-by-8 MinKron state remains at every scale. No new expert execution,
all-history scan, or cross-level backward traversal is introduced.

Each address has three separately normalized parts: eight coordinates for the
current stable input sketch, eight for the previous sketch, and sixteen for
eight sine/cosine clocks. Byte sketches use a fixed signed codebook; larger
sketches use ordered fixed permutations of child sketches. They do not depend
on learned embeddings, expert weights or surrounding history. Learned canonical
content is still published separately and is independent of time.

The eight clock periods are geometrically spaced from 4 to 256 **native events**.
Their corresponding byte periods are 16-1,024 at Level 1, 64-4,096 at Level 2,
and 256-16,384 at Level 3. Fixed 2D rotations advance the clocks once per event;
double-precision phase is renormalized every 256 events. There are no trigonometric
calls in the event loop. This is a rotary/Fourier time address, not transformer
RoPE applied to the learned content vector. Frequencies remain fixed for the
lifetime of a checkpoint policy.

The sixteen stored values combine eight pooled learned-content features with
eight fixed input features. Read queries learn from current canonical content,
the previous contextual state and the clock. A fixed current-sketch term gives
the content query an initial useful orientation. The query and write key are
independent: learning a better read does not move old write coordinates.

For row `a`, the operations are:

```text
read[b] = sum_a query[a] * memory[a,b] / sqrt(1 + mass[a])
memory[a,b] = decay[a] * memory[a,b] + key[a] * value[b]
mass[a] = decay[a]^2 * mass[a] + key[a]^2
```

Reading precedes writing. Fixed retention half-lives range from 32 to 2,048
native events; each temporal sine/cosine pair shares its decay. Mass tracks
write energy to normalize reads. These are bounded recurrent summaries, not
an exact record store. Local derivatives include the learned values, query,
read projection and carried memory matrix; fixed sketches, clocks and mass
have no trainable ancestors.

Each slow event also reconstructs one of its four detached child contents from
the same canonical content used for composition. The selected child role rotates
across events. A shared 64-by-64 decoder uses a role-specific input permutation
and bias. This costs one reconstruction per slow event during training, rather
than four; inference does not run the reconstruction head. Default weight is
0.05, applied to mean squared error per coordinate and averaged across the local
block. The existing contrastive next-unit objective remains weighted at 0.1.
Both losses update the selected expert and ordered composer. Predictions are
trained only after the next unit is observed; reconstruction uses observed
constituents. No future input is supplied to the forward path.

Together the new query/read/reconstruction parameters cost 9,728 scalars per
slow level. The fitted-budget allocator reduces expert counts to compensate.
The addressed four-scale topology needs at least 152,383 scalars; use 200,000
or more for small four-scale tests.

```python
model.configure(reconstruction_weight=.05)  # zero is an objective ablation
stats = model.stats()                      # profile, clocks, losses, sizes
loss = model.evaluate(test_bytes, memory="no_time")
# Other frozen controls: "no_read" and "no_content".
```

Memory interventions act on all slow scales. Set them before consuming a cold
evaluation prefix so the removed channels are absent throughout that prefix.
They are rejected on training models. `no_content` removes content **address
rows**, not the content-derived query or the values stored in temporal rows.
Changing outputs under an intervention demonstrates contribution, not improved
long-dependency accuracy; the comparison runner measures both.

All slow gain/bias matrix products are cached at their native event rate.
The decoder is a single balanced binary tree: training touches the target's
eight binary decisions, and `logits()` returns normalized log probabilities of
all 256 bytes. Thus output scores and nats/byte are exact, not sampled-softmax
approximations. There is no separate full byte-vocabulary head at each scale.

The indexed router searches 16-way derived centroid layers and then at most
16 leaf experts (banks of 32 or fewer scan directly). It is approximate relative
to a full-bank argmax. Prototypes and affected centroid paths update only at
local block retirement. `router="scan"` provides the exhaustive control.
The currently supported fitted-budget range is 50,000 to 6,000,000, subject
to topology overhead; this is not a qualified 100M runtime.

## Learning and lifecycle contract

1. Each scale's parameters remain frozen across eight native events. Records
   accumulate the task adjoints of every byte prediction they serve.
2. Before the ninth event, the last output retires and one reverse sweep updates
   the eight records. Recurrent state and both memories receive local temporal
   derivatives. Published child content and the entering state/matrix detach.
3. Task gradients average over served byte predictions; native contrastive
   gradients average over available targets and have an independent weight.
   Constituent reconstruction gradients average over observed native events.
   Updates use clipped normalized block gradients and selected expert buffers.
4. The native head predicts the next same-scale canonical content, normalized
   against up to sixteen immutable older same-scale alternatives. The forecast
   exists before its target. Duplicate near-identical negatives are excluded.
5. At a block edge, the final next-unit forecast is truncated. It is not trained
   later with changed parameters or a new hidden parameter version. There are
   at most seven native forecast losses in each eight-event block.
6. The shared byte head updates online after computing that prediction's state
   derivative with the exact readout weights used for the prediction.

`finish()` explicitly applies a partial block and truncates its eligibility.
Its held output is thereafter constant until replaced. `reset_context()` first
finishes pending updates, then starts fresh composition, state and eligibility;
rotary clocks and both recurrent memories reset together. Explicit association
episodes survive unless `clear_memory=True` is passed.

`inference_copy(cold=True)` snapshots currently fitted weights, clears context
and episodes, and freezes all learning/admission. `cold=False` retains the causal
prefix and records. It does **not** commit pending training in the source.
Call `finish()` explicitly if evaluation should include those pending updates.
Forward-only inference does not populate derivative rings. Training checkpoints
retain partial child groups, every local record, negative targets, readout,
recurrent matrices, clocks, address normalization, stable sketches, episode
payloads and counters. A version/ABI header and a
SHA-256 envelope reject mismatched or corrupt saves; exact continuation is
verified on this native ABI.

Version-1 multiscale checkpoints automatically select the unchanged runtime in
`legacy/`; they retain their original learning and state semantics. New addressed
checkpoints use native version 2 within the existing checksum envelope. Use
`Model(memory_profile="legacy")` or CLI `--memory-profile legacy` for fresh
comparisons. Resume selects the saved policy automatically. Old cortical
checkpoints remain a different format.

## Association store

`associations=True` enables the neural association path: once per Level-1 event,
query current content plus slow context, scan at most sixteen indexed candidates,
return at most two, and cache their mean contribution to decoder conditioning.
Query precedes admission. The previous key is paired with the newly completed
observed content and its four-byte payload. Recalled vectors are detached.
The fixed store has 256 buckets with four ways (1,024 records); collisions evict
the oldest record within a bucket.

Exact payload access is explicit and works even when neural recall is disabled:

```python
handle = model.bind(b"new-key", b"an already observed exact payload")
assert model.lookup(b"new-key") == handle
assert model.payload(handle) == b"an already observed exact payload"
```

Exact keys have 1-64 bytes and payloads 0-256 bytes. Full keys are compared after
hashing; handles include a slot generation, so overwritten/evicted handles fail
instead of returning another record. This API preserves payload bytes; it does
not claim the neural byte generator has learned to emit arbitrary stored strings.
Admission of an explicit API binding is a caller action, including in inference.
Automatic neural admission is frozen in inference copies. Neural recall remains
off by default until delayed-association and language controls justify enabling it.

## Verification and evidence

```bash
python multiscale/tests/run_all.py
# Hosts managed under ptrace cannot run LSan; explicitly retain ASan/UBSan:
python multiscale/tests/run_all.py --no-leak-check
python -m multiscale.benchmark --data input.txt --out /tmp/shoe-benchmark --checkpoints
python -m multiscale.dependencies --out /tmp/shoe-dependencies
python -m multiscale.compare_memory --data input.txt --out /tmp/shoe-memory-comparison
```

The language runner reserves the final 10%, trains the first million targets
exactly once, and compares seeds 7/17/29 at one, two and four scales. It reports
full-test loss, fixed-subset curves against both bytes and training seconds,
frozen per-level interventions, generation rate and native inference rate.
Initialization, compilation, evaluation and checkpoint IO are outside training
timing; terminal local updates are included. A single CPU is pinned explicitly.

The dependency runner uses AB versus BA followed by identical distractors and
an answer query. It includes gaps and grouping shifts absent from training.
This is a separate acceptance test from mere changed-output interventions.
The memory comparison alternates legacy/addressed execution order across three
seeds, at the same one-million-parameter budget, pinned to one CPU. It records
language curves, full-test loss, memory interventions, throughput and the ordered
distant-cue task. Use `--memory-profile legacy` with either original runner to
reproduce the original algorithm.

See `ADDRESSED_MEMORY_REPORT.md` for the latest change and measurements, and
`REBUILD_REPORT.md` for the original runtime's passed mechanisms and failed
architecture gates. Learned boundaries, mixed-corpus selection, systematic
held-out recombination and larger-scale router qualification remain gated work.
