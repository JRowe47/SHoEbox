# Explicit shared hippocampus

SHoEbox now has an optional shared associative memory above its regional graph.
Regional experts still encode, interpret and compute with content. The shared
service holds compact associations and versioned handles to owner-local capsules.
The feature defaults off. The original retention checkpoint remains the working
parent until the paired-recall promotion gates pass.

## Execution and ownership

Each region has a dedicated input-driven cue trace, one multiplexed interface
and a bounded capsule bank. The compact interface trunk is shared. The existing
three regions request through C, P and Q, respectively; all seven nodes can own
capsules. Registration and owner handles replace pairwise global edges.

At each native internal round the interface updates its cue from currently
available ingress, reads the previous committed memory snapshot and delivers a
gated residual at the selected requester's expert input. Existing regional stages
then execute normally. A learned admission score chooses one eligible producer
per region; resulting offers are committed for the next round. Queries neither
re-ingest bytes nor advance source clocks. Counterfactual queries fork immutable
history and isolate all later writes, receipts and cue traces.

The initial request never includes retrieved candidates. Owner readers receive
capsule, local context, read cue and a multiplicative capsule/cue interaction.
Returned roles remain distinct through a shared composition projection. A shallow
CA1 head compares requested, expected and returned descriptors; accept/null and
bounded conditional rereading controls precede regional expert computation.

The DG code keeps eight of 256 learned cells. A bounded binary directory chooses
up to two shards before sparse posting lookup and fine scoring. CA3 retains the
original query score through at most two neighbor-supported updates. Two bindings
and up to eight owner readbacks fit within one cumulative request budget. Refining
an already returned capsule does not add a native graph round or transport read.

## Persistent state and learning

A handle includes stream/reset namespace, region/node, slot, generation, content
version and semantic version. Recycled slots cannot impersonate expired content.
Snapshots own immutable tensor pages; updates copy only touched record containers.
Capsules are ready before a binding references them. Invalid descriptors and
payloads are masked together. Stale returns become null; bounded maintenance
repairs links and refreshes DG postings from retained address descriptors.

Admission, reinforcement/revision/separation, write/erase, query matching,
selected-link weights, comparison, conditional reading and reinstatement remain
trainable. Existing regional parameters remain trainable too. DG and write/link
choices use explicit straight-through estimators. A rejected admission preserves
forward content exactly while providing a live-window write-gate tangent through
an existing owner capsule. Hard routing and discrete replacement receive credit
through the gathered/selected paths, not an exhaustive differentiable search.

Persistent content does not decay when a transient retrieval gate closes.
Retention/replacement acts only on selected records. Reinstated-only activity
cannot self-admit; regional offers require current external grounding and retain
derived provenance. Usage feedback follows actual validated contributor handles
and is applied once. A separate zero-initialized gain introduces this usage into
existing next-round publication demand without changing initial parent behavior.

Exact live autograd follows source outputs into capsules, queries, selected links
and readers. Detach is the declared optimizer boundary. The implementation adds
no historical content-gradient reconstruction, replay buffer, answer dictionary,
parser, frozen backbone or auxiliary answer-supervised binding objective.
Potentially threshold-dependent binding, revision and completion controls remain
independently ablatable rather than being removed on a small pilot failure.

## Compatibility and accounting

`GraphConfig.hippocampus` accepts `HippocampusConfig`; its default is disabled.
Disabled construction allocates no memory banks or new trainable modules and
consumes no extra RNG. Enabled reader initialization uses a private RNG scope;
zero initial output and publication-feedback projections preserve parent logits.
Old graph/checkpoint dictionaries migrate with absent hippocampal state. Optional
state has its own schema and supports fork, detach, row reset, device transfer,
unique tensor-storage accounting and plain-dictionary serialization.

Optimizer migration maps the retained parent's AdamW moments by stable parameter
name, including aliases only once. New moments appear only when a new parameter
receives a gradient. Activation checkpointing uses non-reentrant recomputation
for associative state to preserve `None` gradients on unused pages and owners;
the legacy path keeps its original implementation.

The paged backend is a CPU reference built from lazily allocated, immutable
one-record tensor pages and bounded posting lists. Python metadata and tensor
storage are measured separately. The actor fabric provides bounded queues,
idempotent submission, explicit overload/expiry, fair service, causal snapshots
and serializable pending messages. It is a single-process transport simulator;
physical asynchronous multi-device execution is the later H5 milestone.

## Reproduction

Use the retained parent at `artifacts/hippocampus/parent.pt` and the exact bundled
Tiny Shakespeare corpus. The matched run uses 64 updates at LR 0.003,
paired/paired/single/correction, gaps 0/4/8/16/32, language weight 0.25, syntax
weight 0.1 and global clipping 1.0. All inherited source cursors and RNG carry
forward. Validation uses seed 683/start 3000 and eight templates per cohort;
testing uses seed 1201/start 80000. Final evaluation uses 32 single templates,
16 paired/correction templates, eight documents per ordinary stream and 1,024
natural bytes. Source labels and answers never enter inference.

```bash
python -m unittest experiments.text_regions.test_hippocampus
python scripts/run_hippocampus.py --arm A --out artifacts/hippocampus/A
python scripts/run_hippocampus.py --arm B --out artifacts/hippocampus/B
python scripts/run_hippocampus.py --arm C --out artifacts/hippocampus/C
python scripts/finish_hippocampus.py --workers 3
python scripts/benchmark_hippocampus.py --out artifacts/hippocampus/scaling.json
```

`--resume` continues a saved run from its original parent with the same protocol.
`--initial-reference artifacts/hippocampus/A` reuses A's initial validation for B/C
only when parent/cohort identities match and both added output gains are exactly
zero. This reuse is recorded in the initial result. Each arm is evaluated normally
after training. `--stop-after` makes an explicit resumable checkpoint; it does not
change the declared 64-update budget.

A is the unchanged paired-training baseline. B enables dedicated cues and local
conditional anchors without the global index. C enables the shared index. All
arms have identical data and update budgets. Inference interventions block
readbacks, shuffle valid selected binding links, pool roles, or disable refinement
or feedback. The fixed update-64 endpoint is primary; validation-selected
checkpoints are reported separately.
