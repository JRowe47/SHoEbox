# First-class memory context

The architecture-review update gives every regional node and the final composer
explicit, versioned evidence slots. The legacy model and checkpoints remain
loadable; `MemoryContextConfig(enabled=True)` installs the new interface.
`migrate_memory_context` enables the event cue clock and bounded CA1 policy,
disables extra lace credit, and preserves inherited weights and AdamW moments.

## Computation

Each consumer keeps a separate current query and at most eight evidence slots.
The query combines fresh ingress, the regional cue trace, a learned node role,
and that node's committed working state. Query features never compete with
candidate values for capacity. The common slot record contains raw content,
learned key and role, validity, source identity/version, source-event time,
and a separate consumption strength.

Hippocampal candidate discovery and local publication transport keep their own
storage. Both feed the same context. Incoming publications arrive through a
direct content lane alongside the existing transformed graph path. Active
forward edges receive protected slots; other evidence competes through a
learned relevance/hold score. Slot identity deduplication happens before top-k.
No answer labels, entity parser, or replay store is added.

Owner decoding is `content_projection(capsule) + contextual_adjustment(...)`.
The current query, stored key, capsule and owner's current working state occupy
separate arguments. The content projection starts as identity. Source encoding
also preserves a direct latent lane beside its small learned adjustment.

Cached raw capsules are interpreted under the current consumer query. Bounded
CA1 can accept, null or perform a local reread; a reread accepted by the terminal
comparison is the one delivered. Conditional content and acceptance remain
separate. Reading never attenuates the stored capsule.

One shared query-conditioned slot reader runs before sparse routing and again
at configured block depths (default: blocks 0 and 2). The same fetched context
is reused; block access does not perform another global lookup. The composer
gets a late context access before the byte head. Its query sees the complete
P/Q publications, together with the reserved question lane.

## Storage and work

| Concern | Implementation |
| --- | --- |
| Processor vs. publication | Working state advances on active input even when a durable publication holds. `decouple_working_state=False` retains the hold-mode ablation. |
| Event identity | Source-event time is separate from the internal visibility round. Source and abstract anchors share one regional event binding and the existing link budget. |
| Repeated rounds | An unchanged source offer reuses its anchor. Revised same-event content increments its version in the same slot. |
| Provenance | Mixed anchors record the recalled handles and direct publication versions available to the computation; pure ungrounded recall cannot create a new event offer. These are causal source records, not truth labels. |
| Cached reads | Raw slots survive until invalidation or learned replacement. Namespace, generation, version, owner availability and semantic version are checked before use. |
| Demand | Changed requests, stale handles, an age limit or a learned refresh decision request more evidence. Round-robin admission limits directory requests to four per stream per round by default. |
| Shared fetch | Requests share capsule fetches within a world and causal cut, while owner interpretation remains consumer-specific. |
| Gradients | Ordinary task/language losses train requests, hold/refresh, reads, composition and writes through the live credit window. Detached old content trains current consumers without invented historical writer derivatives. |

The source/abstract pair uses two anchor slots per admitted regional event.
With capacity 64 this permits 32 such events; additional internal rounds do
not allocate another pair. Admission and learned semantic revision still
determine which offers survive. This reference scheduler bounds directory work;
it does not add background workers or distributed RPC.

## Continuation and evaluation

```bash
python scripts/run_memory_context.py \
  --parent artifacts/memory_context/parent.pt \
  --out artifacts/memory_context/pilot --steps 4

python scripts/evaluate_memory_context.py \
  --checkpoint artifacts/memory_context/parent.pt \
  --cohorts artifacts/memory_context/cohorts.json \
  --out artifacts/memory_context/parent_evaluation.json --lock

python scripts/evaluate_memory_context.py \
  --checkpoint artifacts/memory_context/pilot/last.pt \
  --cohorts artifacts/memory_context/cohorts.json \
  --out artifacts/memory_context/final_evaluation.json
```

Use `--resume` with the same parent, output directory and declared step budget
for an interrupted run. A new continuation can use `pilot/last.pt` as its
parent with a new output directory. Every optimizer update writes an atomic
checkpoint, including optimizer, RNG, language stream, context and episodic
state. Source/protocol guards reject an incompatible exact resume.

The compact acceptance set uses one paired template at each of +0/+32/+64
bytes, a full 4x4 value crossing, single/correction retention at +64 and 128
language targets. All compared models receive identical locked inputs. This
is an integration check; it does not replace the established promotion banks.
One fixed-history intervention blocks only source A's new context lane.
The unit-level source intervention independently verifies that changing one
selected fact changes its requested output while preserving the other query.

Relevant checks:

```bash
python -m unittest experiments.text_regions.test_memory_context \
  experiments.text_regions.test_hippocampus \
  experiments.text_regions.test_hippocampus_binding \
  experiments.text_regions.test_hippocampus_ca1 \
  experiments.text_regions.test_graph_publication \
  experiments.text_regions.test_lace_separation \
  experiments.text_regions.test_paired_recall \
  experiments.text_regions.test_runtime \
  experiments.text_regions.test_content_checkpoint \
  experiments.text_regions.test_task_state_commit

python scripts/check_memory_context_resume.py \
  --parent artifacts/memory_context/parent.pt \
  --out artifacts/memory_context/resume_checks.json
```
