# SHoEbox joint producer and connection learning

Locked November 5, 2026, after a separate 40-update development pilot on parent 20261121 with data seed 20261200. No confirmatory test results are used to choose the protocol.

## Target

Improve the weakest retained producer/connection pair through local joint learning, preserving the frozen seven-node recurrent composer. Require at least 95% on each of recall, correction, and withheld-correction queries at gap 16 in all three selected joint models. On the previously weakest parent (20261121), require the joint model's minimum accuracy across those tasks to exceed both the unchanged parent and the matched edge-only continuation. Require a K=2 versus K=1 gain of at least five percentage points, exact preservation of memory across repeated queries, and unchanged composer weights in every seed.

The 90% target from the preceding integration phase remains visible as a reference. Gap 64, gate precision/recall, K=4, and correction after live queries are reported separately. The broader M7 shared/private and natural-text target remains outside this focused phase.

## Parents and matched arms

| New data seed | Published parent | Arms |
| --- | --- | --- |
| 20261130 | retained_composition/20261120/best.pt | Joint and edge-only |
| 20261131 | retained_composition/20261121/best.pt | Joint and edge-only |
| 20261132 | retained_composition/20261122/best.pt | Joint and edge-only |

Each pair starts from the same complete eight-node model, sees the same procedurally generated observations and relation tables, executes the same experts, and uses the same losses and checkpoint rule. The joint arm trains F's contextualizer, ingress projection, private and independently owned shared expert banks, publication gate, and the F-to-A value matrix. The edge-only arm trains that value matrix alone. All other parameters and buffers are fixed, including the common 256-byte readout. No additional model or training-only head is installed.

## Training and gradient boundaries

Use 80 updates, batch size 32, alternating recall/correction, cycling gaps 0,4,8,16 with both kinds represented at every gap. Data seeds are new and all three published parents are retained. AdamW uses beta=(0.9,0.95), weight decay 0.01, producer learning rate 0.0003, edge learning rate 0.001, and global trainable gradient clipping at 1. No schedule or hyperparameter sweep is used in confirmation.

The loss has four terms plus existing routing regularizers:

1. Mean squared error of the edge-projected retained payload against the frozen A ingress representation of the causally observed current fact.
2. The same alignment for current write candidates at observation positions.
3. Balanced binary cross-entropy for the learned write/hold decision.
4. Final 256-byte cross-entropy through the frozen composer on g(f(x)).

Both alignment terms are normalized by the mean squared magnitude of the four frozen reference embeddings. Labels are used only by these losses. The publication gate uses its ordinary hard threshold at zero; no straight-through estimator or forced training write is used. Local losses are normalized over the full batch and backpropagated before every 16-event detach boundary. The last chunk stays differentiable through the query; the optimizer advances once per batch. This gives local feedback to acquisitions before older activations are discarded, without extending the backpropagation horizon or replaying observations.

## Selection and held-out evaluation

Validate the unchanged parent and updates 20,40,60,80 on 128 fresh validation episodes per task at gap 16. Select lexicographically by minimum task accuracy, mean accuracy, and lower mean negative log likelihood, keeping the earlier checkpoint on exact ties. Checkpoint zero is eligible. Do not inspect test data until all six training runs finish.

Evaluate 512 fresh test fact episodes per condition and seed. Both arms and the unchanged parent use identical test cases. Relation-table pairs come from the existing held-out composition partition; query inputs contain B/C tables and query markers, with x supplied only through F's retained latent. There is no intermediate fact-byte decoding. Source F remains inactive during the query.

Report recall/correction/withheld at gaps 16 and 64; K=1,2,4; disabled delayed exchange; blocked F-to-A edge; reset memory; shuffled whole publications (payload and semantic header together); unconditional publication with the same weights; four consecutive queries; and corrections following three queries. The latter retains the preceding benchmark's restatement of the old fact before the optional correction. Report the selected and final endpoint, the original parent on the matched test set, and write precision/recall with exact hold counts. The original-target and substituted-memory-target accuracies are both reported for shuffling.

## Verification and publication

Save resumable model/optimizer/RNG checkpoints, protocols, batch hashes, table keys, histories, predictions and inference exports. Audit all 480 update batches, validation selection, parameter ownership, execution/exposure counts, all held-out prediction groups, and exact exported logits/state. Regressions must cover gradient reachability, history detachment, label isolation, unchanged inference, parameter ownership and bit-exact optimizer continuation.

Training wall time includes concurrent CPU contention and excludes validation and audit. Equal expert execution does not imply equal backward cost; producer training performs additional differentiation. Publish the completed source tree, full comparison, ordinary Markdown/PDF reports, and resumable artifact package.
