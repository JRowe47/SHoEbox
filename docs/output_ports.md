# Persistent output ports

**Status: interface design.** `experiments/hat_minkron/events.py` implements persistent byte slots, version increments and event snapshots. The typed port lifecycle described here is not yet an executable runtime in this repository.

An output port exposes the latest completed signal from a module. The payload stays available until replaced or invalidated; the routing network can inspect a compact header without running the producer again or moving its full output.

## Header and payload

| Field | Meaning |
|---|---|
| `link`, `slot` | Exact identity of the producing module connection and output slot |
| `generation` | Monotonically increasing identity of a completed publication |
| `payload` | Stable handle to the published storage |
| `layout` | Mechanical representation: byte length, dtype, byte order, shape, strides or record offsets |
| `valid` | Whether the port currently exposes a readable publication |
| `published_at` | Publication time or logical clock for age policies |
| `gain`, `decay_policy` | Availability or routing strength and its regional/global lifetime policy |
| semantic header | Cheap routing evidence supplied by the producer's semantic interface |

The payload can remain in its native tensor storage with a byte view. A generic layout descriptor explains how to locate records without interpreting whether the tensor describes an image, sound, text or a sensor. Learned semantic evidence remains separate from exact identity and mechanical layout.

**A local record index is not a byte offset.** A consumer requesting record 17 resolves that record through the layout before reading bytes. Fixed-size tensors can use shape and strides; variable-size records require offsets. The signal's relative local index travels with routing and the resulting read.

## Publication lifecycle

1. **Stage privately.** A producer writes a new candidate output to staging storage. The previous complete publication remains available.
2. **Complete and commit.** The producing expert decides that its output is complete. Publication atomically swaps the payload and matching header, incrementing `generation`.
3. **Probe cheaply.** Connected semantic heads inspect routing evidence and availability. No producer or expert execution is required to inspect a published header.
4. **Read on activation.** A selected downstream node acquires a consistent generation and reads its requested record or range. In-flight readers retain that generation until they release it.
5. **Retain or replace.** The latest payload persists. An older payload is retained only while an in-flight reader still needs it, then can be reclaimed.

A diffusion expert, for example, keeps its intermediate denoising states private and commits only when its completion criterion is met. That criterion is expert behavior; the port only enforces the completed-publication boundary.

## Consumption and memory

Publication identity and activation order have different meanings. A fixed dictionary entry can stay at generation 0 while being consumed thousands of times. minKron advances when an occurrence is consumed, not merely when its source publishes a new generation. A generation change tells a consumer that the payload changed; it does not substitute for a consumed-event index.

Ports belong to a module invocation or stream. Two children using the same physical expert weights still own separate output ports and runtime state. Weight sharing must not accidentally alias their signal buffers.

## Decay and reset

Decay acts on availability or routing gain while preserving payload bytes. A reader or probe can compute effective gain from publication age lazily, so every clock tick need not sweep all buffers. If an expert needs numeric fading, it applies gain after decoding the numeric representation; multiplying serialized bytes would corrupt the representation.

Regional or global reset invalidates the affected publications. Invalidity is explicit: a zero byte is legitimate data and cannot double as a missing signal. Reclamation waits for pinned readers to finish. Clearing ports and resetting recurrent state are separate operations that a regional reset policy can invoke together.

## Relationship to the current demo

`SlotDictionary.publish(link, slot, bytes)` replaces the slot's current byte string and increments its version. `SlotDictionary.events(...)` materializes immutable byte events for the consumer. This demonstrates exact addressing, persistent current values and snapshot reads.

Typed layouts, native tensor handles, private staging, reader pinning, lazy decay and reset coordination are the next implementation layer. They do not require adaptive event merging, a learned modality adapter, or moving specialized computation into HAT.
