# HAT + minKron SHoE

> Public release: run `python scripts/prepare_data.py` from the repository root before training. Recorded metrics and samples are included; checkpoints are created by training and are excluded from git. Commands below run from this experiment directory.

HAT is a shared byte interface for fixed dictionary entries and streaming module outputs. Every input occurrence carries an exact `(link, slot)` address and a byte value. The model learns contextual symbols and routes them through the existing parent / two-child SHoE hierarchy. minKron replaces the children's attention mixers and their historical key/value storage.

## Implemented architecture

- **One 256-byte alphabet.** Text uses UTF-8 bytes; arbitrary binary sources use their existing bytes. There is no modality switch or modality-specific neural encoder.
- **Static and live dictionary entries.** Link 0 contains immutable byte values in slots 0–255. Other links expose byte payloads in stable slots. Publishing changes a slot's current value and version; already materialized events retain their original bytes.
- **Exact provenance.** Link and slot identities accompany the learned representation. Approximate overlap in learned symbols never replaces the exact address.
- **Two HAT levels.** Each level uses the same minKron weights across sources, maintains separate state for active links, and selects two entries from a shared 64-entry learned symbol dictionary. The second level consumes the first level's contextual representation. Values have a residual path alongside selected symbols.
- **One semantic parent, two recurrent expert SHoEs.** The parent probes child semantic heads without executing their recurrent memories or experts. It dispatches each event to one child. Low-confidence routes probe both heads; established matches can use the learned fast path.
- **Three minKron blocks per child.** Each block updates a fixed recurrent matrix, then routes to its sparse MLP experts. There is no softmax attention layer, historical-key index, positional embedding table, or token KV cache in the live model.
- **The prior expert sharing is preserved.** Each child can select from 1,024 logical experts, with 512 shared and 512 dedicated. There are 1,536 physical experts, 1,228,800 expert weights, and 48 expert activations per input event. Shared matrices are the same Parameter objects in both children.
- **Bottom-up feedback is retained.** Executed child laces supervise the parent's translated semantic prior. Signed downstream usefulness updates routing associations explicitly after the optimizer step, for the next pass. Forward calls never mutate these model buffers.

The experts are the existing tiny MLPs. Their sequence context is supplied by a recurrent mixer at each child block, replacing the attention that previously supplied it. There is not a separate memory allocation for each of the 1,536 MLPs.

## Counts and memory

The default model has **1,624,588 unique parameters**, including **136,512 HAT parameters**. Width is 100. Each minKron has four 8×8 heads and uses FP32 persistent accumulation.

| Runtime memory | Per sequence |
| --- | ---: |
| Both children's six recurrent blocks | 6,144 bytes |
| Both HAT levels, per active link | 2,048 bytes |
| One link, all recurrent matrices | 8,192 bytes |
| Eight links, all recurrent matrices | 22,528 bytes |

Integer clocks add eight bytes per active link and eight bytes per sequence. With one active link the complete returned state is 8,208 bytes; with eight it is 22,600 bytes. Batches multiply these counts. Link state is allocated lazily at batch granularity, so a link used by one row has a state row for every item in that batch. Inactive links retain their matrices without decay or updates. Each child advances only on its routed events.

These figures describe inference state, not weights, gradients, optimizer state, the caller's current input payloads, or training activations. Training uses bounded 32-event recurrence chunks internally; its computation graph still spans the supplied training window. No growing sequence tensor is retained in the returned inference state.

## Input and streaming API

```python
import torch
from events import ByteBatch, SlotDictionary
from model import HATConfig, HATSHoE

model = HATSHoE(HATConfig()).eval()
bank = SlotDictionary(max_links=8, max_slots=256)
bank.publish(link=1, slot=3, data=b'\x00\xff\x41')

# A literal byte and a live source slot use the same API.
events, versions = bank.events([(0, 65), (1, 3)])
with torch.no_grad():
    first = model(events)
    bank.publish(1, 3, b'\x42\x00')
    more, _ = bank.events([(1, 3)])
    second = model(more, state=first['state'])

# Tensor forms support a batch of equal-length event streams.
events = ByteBatch.text('ROMEO:\n')
# ByteBatch.static(byte_tensor): link 0, slot = byte value.
# ByteBatch.stream(byte_tensor, link=1, slot=0): live output bytes.
# ByteBatch(values, links, slots): mixed addressed inputs, all [B,T] int64.
```

`SlotDictionary` versions identify snapshots for callers. Versions are not interpreted by a modality-specific parser. A packet's bytes are consumed in supplied order. The model is causal and the result does not depend on how that same ordered event sequence is divided into API calls.

Each live slot may contain an arbitrary nonempty byte string. Serialize floating point or multibyte numeric outputs explicitly using the producer's chosen byte convention; HAT consumes the resulting bytes. This is a byte boundary, so arbitrary external serialization is not an end-to-end differentiable connection to a producing network.

`state.detach()` truncates the training graph between carried windows without clearing memory. `model(events, state=state, reset=boolean_batch_mask)` resets selected independent sequences before consuming the call. Reset whole sequences when reusing batch rows for unrelated episodes. State passed into a call is never modified in place. Use `torch.no_grad()` or inference mode for serving.

Use `state.to_dict()` and `StreamState.from_dict(...)` to serialize/restore memory alongside a model checkpoint. Keep batch-row identity stable across calls. The `context` configuration is the training-window size, not an inference cache limit.

## Training and checks

```bash
python3 verify.py
python3 train.py --steps 600 --threads 2 --out results/language
python3 learning_checks.py --steps 500 --threads 2
python3 generate.py --checkpoint results/language/checkpoint.pt --prompt 'ROMEO:'
python3 example_live.py
```

Only PyTorch is required for inference and tests. Report generation also uses matplotlib. The default reference executes on CPU and can be moved to a PyTorch device; fused accelerator kernels are not included.

The language shakedown trains on raw Tiny Shakespeare bytes. Each batch contains two static-dictionary rows, one live link-1/slot-0 row, and one live link-2/slot-7 row. It uses 600 × 4 × 128 = 307,200 byte targets, fixed validation windows, and an untouched test split. It saves weights, optimizer, RNG, protocol, and a sample stream state.

The two additional learning checks isolate the mechanisms:

1. HAT plus a small readout must recall a random byte for the requested source and slot after distractor events.
2. The complete small SHoE model performs that task with HAT history disabled and child 0 forced, so historical information must come from the expert blocks' minKron state.

Frozen interventions remove history, source identity, or slot identity. These are mechanism checks; they are not independently trained benchmark baselines. Validation loss from this run is also not a matched comparison with the earlier 65-character tokenizer experiment.

## Scope of this version

HAT emits **one contextual symbol occurrence per input byte**. Its hierarchy is representational: learned top-k symbols at multiple levels. Adaptive event merging and variable-rate emission are not implemented. Static/live slot access, source-conditioned symbol construction, recurrent expert context, sparse execution, and delayed routing reinforcement are implemented.

The two child memories integrate inputs that actually reach each child. HAT preserves each input link's own continuity, while a child's recurrent state supplies fusion across routed sources. There is no replay pass or retrospective recomputation of old expert outputs.

## Source files

- `events.py`: static/live slot dictionary and byte-event contract.
- `kron.py`: bounded-chunk normalized minKron with permanently packed projections.
- `hat.py`: hierarchical symbols and state indexed by source link.
- `model.py`: recurrent two-child SHoE, top-level routing, feedback, generation, and carried-state API.
- `semantic_core.py` / `hierarchy_reference.py`: preserved prior SHoE source used for its lace controller, parent and shared-bank implementation. Their original attention classes are not instantiated by the new model.
- `verify.py`: independent-reference, gradient, provenance, causality, dispatch and checkpoint checks.
- `train.py` / `learning_checks.py`: reproducible language and delayed-recall runs.
- `references/`: prior compact minKron source, the authors' original reference, the extracted tokenwise reference used by tests, and MIT notices.

The minKron recurrence preserves normalized SiLU Q/K/V, separate write/erase gates, vector forgetting, gated RMS normalization, and the 1e-18 value-norm floor. Its output projection is initialized at standard deviation .02 for this residual model. The tested compact recurrence is adapted from the October 21 bundle; source identities and hashes are recorded in `PROVENANCE.json`.
