# SHoEbox: joint natural-text and retained-memory protocol

This phase starts from the three published multi-fact checkpoints at parent
revision `f1ae920243715270aa0cac92b5ab5f814fef85d1`. It trains the same eleven-node,
510,958-parameter graph on Tiny Shakespeare and fresh retained-fact episodes.
The numerical protocol is locked in `artifacts/joint_text/protocol_lock.json`
before confirmatory training. The development pilot is separate.

## What is shared

Text and memory queries use the same seven-node composer, sparse expert banks,
connections and affine 256-byte decoder. Four additional owners retain shared
and private facts. No new parameters or task-specific output heads are added.
The composer, decoder, producer networks and producer value projections train.
The existing producer-edge context and gate parameters remain fixed.

Text bytes enter the A, B and C views. Each view has its own streaming HAT
state and all three encodings are counted. F, L, G and R are inactive during
text; their outgoing edges are masked. With no explicit state reset, text
preserves the stored facts and cannot read an unaddressed fact. Structured queries retain their explicit field
address and latest-valid shared-replica arbitration. Natural-language fact
parsing and natural-language query addressing are not implemented in this phase.

## Training and selection

Each of 160 updates accumulates a language objective and the existing
multi-fact objective before one AdamW step. Natural text uses eight ordered
streams and 16-byte chunks, carrying detached graph state between updates.
Memory learning uses 16 fresh procedural episodes, cycling recall and four
correction tasks across gaps 0, 4, 8 and 16. No replay store is used.

The memory loss includes write/hold supervision, retained/candidate latent
alignment and three final composed-answer losses. Its four alignment targets
are captured from each parent before training and remain fixed. Each memory
chunk backpropagates before detaching. Text and synthetic episodes have
independent runtime state; their weights and optimizer are shared.

AdamW uses betas (0.9, 0.95), weight decay 0.01 and global gradient clipping
at 1. Core, decoder and producer learning rates are 0.0003, 0.002 and 0.0001,
with linear warmup over five updates. All objective terms and execution
counts are logged separately. The multi-fact objective weight is 1.

Validation runs every 20 updates, including update zero. A checkpoint is
eligible when all 15 memory task/field scores are at least 90%; the eligible
checkpoint with the lowest text validation loss wins. Exact ties keep the
earlier checkpoint. Ineligible checkpoints cannot replace an eligible one.
Validation uses 64 episodes per memory task and 1,024 natural-text bytes.
No test result selects parameters or checkpoints.

## Data and acceptance

Tiny Shakespeare is hash-verified against
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
The existing contiguous 90%/5%/5% byte split is retained. Each run consumes
20,480 natural-text training bytes; it does not train on the entire corpus.
Held-out language evaluation uses 4,096 bytes from eight ordered test shards.
A smoothed unigram reference is fitted on the training split only and scored
on exactly the same target bytes.

| Target | Required measurement |
| --- | --- |
| Natural text | Test NLL below 4 nats/byte and at least 50% below the inherited parent |
| Retained capabilities | Each recall, correction and withheld-correction field at least 90%, with 512 fresh episodes per condition and seed |
| Recurrent composition | Two-round shared-right accuracy exceeds one-round accuracy by at least five percentage points |
| Ownership and generation | Exact memory preservation during text, and exact byte/RNG/state continuation after serialization |

The inherited parent was trained for four-value structured tasks; its initial
text loss is not a competitive language-model baseline. The unigram comparison
and generated samples show how much language structure this small run learns.

## Interventions and reproducibility

Repeat the preceding multi-fact evaluation matrix: held-out factor and
relation combinations, source absence and erasure, withheld corrections,
gap 64, K=1/2/4, delayed exchange disabled, blocked/reset/shuffled memory,
unconditional publication, repeated/live queries and forced stale replicas.
Natural-text evaluations compare K=1/2/4, no delayed exchange and state reset
at every 16-byte chunk. The latter two preserve the two-round execution count.

Insert 32 natural-text bytes between fact acquisition and querying in 512
episodes per seed. Verify every memory-owner tensor exactly. Also save a
128-byte autoregressive generation session with facts resident in memory,
then query all three fields. These interface checks do not teach natural
language to address the retained fields.

Checkpoints include the complete optimizer, language cursor and carried
state, RNG, fixed alignment targets, history and source hashes. Audits
reconstruct all training batches and counters, checkpoint selection,
validation and test predictions, and split/resumed generation. Source and
artifact ZIPs are packaged separately with hashes and a published commit.

## Separate final-checkpoint diagnostic

During training, validation showed a tradeoff between language loss and the
memory guard. Before confirmatory test results were available, a separate
diagnostic was declared in `final_diagnostic_protocol.json`: evaluate the
fixed-budget final weights on 4,096 text bytes and the five recall/correction
tasks, each with 512 episodes. This diagnostic does not alter checkpoint
selection or any primary target. Both the selected and final results are
reported, and their checkpoints are preserved.
