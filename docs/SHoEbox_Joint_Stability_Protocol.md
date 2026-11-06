# SHoEbox: stable joint text and memory learning

This phase continues all three fixed-budget joint-text endpoints from revision
`57c3e0156ce3c686d3979aea8218267e6b8e81ef`. Each parent produces two matched arms:
ordinary task-gradient summation and a cap on each task's contribution to each
optimizer group. The graph, 510,958 parameters, shared decoder, fresh procedural
memory objective, and byte-stream interface remain the same.

## Motivation and intervention

One development gradient probe on each parent found text/core norms of about
18,866, 3.91, and 11,559, versus memory/core norms of 10.83, 2.63, and 7.05.
The two large spikes largely arise before the readout. A single global clip
therefore suppresses unrelated readout and producer updates along with the spike.
These probes do not select confirmatory checkpoints; their modified weights are
discarded. Their fresh-episode seeds are also the declared training seeds.

Both arms capture the existing text and segmented memory backward passes
separately. The control adds them. The intervention caps each gradient at norm
one within each existing optimizer group (core, readout, producer), then adds
them. Both arms apply the same final global norm-one cap and one AdamW step.
Clipping never amplifies a task gradient, projects its direction, adds a model
parameter, or requires another model execution. Norms and dot products use FP64
accumulation. Per-task norms, cosine similarities, group scales, and subsystem
norms are logged every update.

## Locked paired study

Three pairs use parent seeds 20261310/11/12 and new memory seeds 20261400/01/02.
Each arm inherits exactly the same weights, Adam moments, text cursor, carried
graph state, and fixed alignment targets within its pair. A five-update learning
rate warmup is applied equally. Rates remain core 0.0003, decoder 0.002, producer
0.0001. Each arm executes 60 updates: 7,680 new natural-text bytes and 960 fresh
memory episodes. Both arms see identical ordered text and procedural batches.
There is no replay store.

Validation at updates 0, 20, 40, and 60 uses 512 episodes for each of all nine
recall/correction/withheld-correction conditions and 1,024 text target bytes.
This expands the earlier 64-episode, five-condition guard to 27 field scores.
An eligible checkpoint must score at least 90% on every field. Among eligible
checkpoints, lowest text NLL wins; exact ties retain the earlier checkpoint.
If no checkpoint is eligible, selection prioritizes the minimum field accuracy
and reports the failed guard. Initial, selected, and final weights are retained.

## Endpoints and controls

The primary text test scores 4,096 bytes from eight shards beginning halfway
through the reserved Tiny Shakespeare test split. These target positions do not
overlap the previous phase's primary 4,096-byte test. Synthetic tests use new seed
namespaces with 512 fresh episodes per condition. No test score selects weights.
The final checkpoint is a prespecified diagnostic and never replaces selection.

Targets are test text NLL below 4 nats/byte; all 27 primary memory fields at least
90%; at least five percentage points of composition gain from K1 to K2; exact
retained-owner state during text; and exact resumed generation. The unigram
reference is fitted on the training split and scored on identical test targets.

Selected checkpoints additionally run K1, K4, no exchange, blocked memory,
reset memory, shuffled memory, source absence/erasure, and long-gap controls.
Text controls use K1 and reset-per-chunk. Interleaved text preserves memory,
and a 128-byte generation session is serialized and resumed at byte 41.
Report both arms for every seed and the fixed-budget final diagnostics. Passing
the continuation endpoint is distinct from outperforming the matched control.

## Reproduction and integrity

`scripts/run_joint_stability.py` writes a hash-locked protocol before training.
Every five updates it saves the optimizer, stream cursor, graph state, RNG,
alignment targets, history, and counters. Repeating the command resumes them.
Tests cover clipping mathematics, nonfinite rejection, absent gradients,
withheld-condition selection, and exact resumed learning. Audits reconstruct
batch hashes, token positions, execution counts, validation selection, paired
data equality, recorded predictions, checkpoint provenance, and live sessions.
