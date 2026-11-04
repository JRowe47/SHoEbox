# Recurrent text regions

This is SHoEbox's event-major reference implementation of the [development specification](../../docs/SHoE_Text_Development_Specification.md). It reuses the baseline's SymbolLevel, RecurrentChild, minKron, lace controller, and split expert banks. No old experiment is copied into this directory.

## Components

| File | Responsibility |
|---|---|
| `config.py` | Serializable topology, ownership, sharing, rounds; forward-DAG validation |
| `node.py` | Existing two-level symbols over latent inputs; existing three-block SHoE |
| `ports.py` | Identity, semantic namespaces, dependencies, stage/commit, snapshots, invalidation |
| `state.py` | Ingress/node state, clocks, reset, detach, serialization, byte accounting |
| `graph.py` | Directed value/context projections, learned gates, diagnostic substitutions |
| `model.py` | Event-major stages, delayed feedback, shared byte decoder, generation |
| `data.py` | Ordered per-row cursors, document boundaries, resumable positions |
| `train.py` | AdamW, inherited schedule/clipping, isolated validation, full checkpoints |
| `probes.py` | Seeded recall/correction, shared/private, and relation-chain fixtures |
| `verify.py` | Baseline parity and engine invariant tests |
| `evaluate.py` | Common measurements for baseline and graph |

## API

```python
import torch
from experiments.text_regions import GraphConfig, GraphState, RegionLanguageModel

torch.set_num_threads(2)
model = RegionLanguageModel(GraphConfig(rounds=2, recurrent_feedback=True))
x = torch.tensor([[65, 66, 67]], dtype=torch.long)
y = torch.tensor([[66, 67, 68]], dtype=torch.long)
result = model(x, y, trace=True)
result['loss'].backward()
state = result['state'].detach()  # after a training segment
more = model(torch.tensor([[68]]), state=state)
restored_state = GraphState.from_dict(state.to_dict())
generated, final_state = model.generate('AB', count=40)
```

The default is K=1 with recurrent edges disabled. Override `rounds` and `feedback` per call. To compare K=1,2,4 with the same recurrent model and observations, reuse the same unmodified prefix state and keep `feedback=True` for all calls. Feedback snapshots continue across external events, so K=1 with feedback enabled differs from the initial feedforward integration configuration.

Every activation updates both contextualizers and all three minKron blocks. Each node owns its state even when expert weights are shared. Forward calls do not update parameters or learning buffers. The current objective is byte cross-entropy plus routing/symbol balance. M6 association commits and representation objectives are not implemented.

## Views and interventions

Ordinary input is a byte tensor or the baseline `ByteBatch`, preserving exact link/slot addresses. One HAT occurrence supplies the three branch adapters. Controlled inputs provide all configured views:

```python
views = {'A': x, 'B': x + 1, 'C': x + 2}
available = {'B': torch.zeros_like(x, dtype=torch.bool)}
result = model(views, availability=available)

# Information mask; node/expert execution counts stay unchanged.
edge_mask = {'A__U__forward': torch.zeros_like(x, dtype=torch.bool)}
result = model(views, edge_masks=edge_mask)

# Execution mask; skips node B while preserving its state/publication.
active = {'B': torch.zeros_like(x, dtype=torch.bool)}
result = model(views, node_masks=active)
```

Views have aligned `[B,T]` tensors and separate ingress state under shared HAT weights. Unavailable bytes never pass through HAT or advance its source clock. An available zero byte is an ordinary input. Source nodes can still execute with retained state or feedback when their view is absent; use node/edge masks to isolate those paths too.

`EdgeSubstitution(payload, memberships)` replaces both parts of a message, with shapes `[B,T,width]` and `[B,T,3,groups]`. Replacing both prevents original semantic evidence from leaking into an intervention. Dependencies mark substituted messages. Masks are boolean `[B,T]`; reset accepts `[B]` or `[B,T]`.

## State contracts

Publications contain payload, validity, source/port, per-row version, stream epoch, external occurrence, round, local memberships, selected group/lace IDs, and consumed dependency IDs. Ingress also retains link/slot IDs. Local coordinates are namespaced by `(node, block)` and translated by each edge's learned map.

Stages commit by replacing references. Existing readers keep their old tensors. Runtime code never mutates supplied state or publications. Frozen records do not make PyTorch storage read-only: callers must avoid in-place edits too. Dependencies are bounded IDs and masks, never recursive references to historical payloads. Batch-row position is part of stream identity and remains stable until reset.

`detach()` severs every carried tensor at a segment boundary while preserving values. Crossing a region or round does not detach. `reset(rows)` clears ingress matrices, node matrices, ports, memberships, validity, clocks, and dependency rows, and advances stream epochs. Inactive rows retain state and publications. Switching between text and controlled-view ingress requires fresh state.

`storage_bytes()` counts deduplicated allocated tensor storage; `tensor_bytes()` counts logical bytes including repeated references. Both exclude weights, gradients, optimizer tensors, and retained training activations. Last-occurrence provenance is copied so it cannot retain an entire input window.

## Training and exact resume

Run from the repository root:

```bash
python -m experiments.text_regions.train --tiny --context 8 --batch 4 \
  --steps 100 --out artifacts/overfit

# Preserve the 20-step schedule while stopping at step 5.
python -m experiments.text_regions.train --tiny --context 8 --batch 4 \
  --steps 20 --stop-after 5 --out artifacts/resume_check
python -m experiments.text_regions.train --tiny --context 8 --batch 4 \
  --steps 20 --resume artifacts/resume_check/checkpoint.pt --out artifacts/resume_check

python -m experiments.text_regions.train --data experiments/hat_minkron/data/tiny_shakespeare.txt \
  --rounds 2 --context 128 --batch 4 --steps 100 --out artifacts/language_k2
```

Data uses the existing contiguous 90/5/5 split. Validation owns fresh state, separate cursors, and an isolated RNG context; test data remains untouched. Each split is partitioned into nonoverlapping row shards. Chunks continue at the next byte; completed shards start new independent streams and reset that row. Targets never cross shard boundaries. Explicit document lists are also supported, including resets inside a chunk.

Checkpoints include configuration, weights, optimizer, graph state, ports, validity, clocks, stream positions, dataset fingerprint, Python/PyTorch RNGs, step, and schedule horizon. Exact resume rejects changes to data, batch, context, threads, rounds, LR, or horizon. Loading uses tensor/primitive-only serialization and `weights_only=True`.

Delayed message gates initialize at 0.05 and forward gates at 0.5; both remain trainable. This damps initial closed-loop transport while retaining gradients through every round. The original minKron recurrence and baseline initialization are unchanged.

## Verification and measurement

```bash
python scripts/check_all.py
python -m experiments.text_regions.evaluate --baseline --batch 4 --context 128 \
  --out artifacts/baseline_profile.json
python -m experiments.text_regions.evaluate --rounds 1 --batch 4 --context 128 \
  --out artifacts/graph_k1_profile.json
python -m experiments.text_regions.evaluate --rounds 2 --batch 4 --context 128 \
  --out artifacts/graph_k2_profile.json
python -m experiments.text_regions.probes --kind recall --split validation \
  --count 512 --distractors 32 --out artifacts/recall_validation.jsonl
```

Profiles separate unique parameters, repeated expert calls, and unique physical experts touched in the declared window. They measure CPU time, routing/expert forward components, state/optimizer bytes, and process peak RSS. Random-byte profiles test execution and gradients; learning runs use separately declared data. Full autograd over 128 bytes consumes substantially more memory than persistent inference state.

The probe generator supplies fixtures and fixed split identities. Probe training/scoring, 512-episode evaluations, three-seed comparisons, and matched-compute controls remain later work. The transport test verifies that A cannot affect Q in round one and can in round two.

See the [development report](../../docs/SHoEbox_Development_Report.md) and [raw results](results).
