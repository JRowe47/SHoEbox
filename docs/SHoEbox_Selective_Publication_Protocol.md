# SHoEbox: selective publication protocol

Locked November 5, 2026, after a 40-update development pilot and before the six confirmatory runs.

## Target

Establish learned acquisition, retention and correction in one existing sparse producer before extending the mechanism to all seven communicating nodes. The producer is node A extracted from the selected gradient-horizon checkpoint. It keeps its byte HAT, latent contextualizer, three recurrent SHoE blocks, shared expert banks and ingress projection. A fresh common 256-byte head decodes the publication. A 33-parameter linear gate reads the producer's computed latent and decides whether to replace its publication.

The engineering acceptance target is at least 90% recall, correction and old-value retention when an update is withheld, on 512 fresh episodes per condition per seed at gap 16. Write precision and recall target 95%. Longer gaps are reported separately. Failure of an accuracy target is reported as a failure, not redefined after test evaluation.

The seven-node integrated checkpoint remains unchanged. Default-off compatibility must reproduce its original composition predictions and state tensors exactly. This phase does not claim joint memory/composition mastery; M7 remains open.

## Paired comparison

- Seeds: 20261110, 20261111, 20261112.
- Two arms per seed: learned hard publication decision; unconditional publication.
- Both arms have the same parameters, initialization, examples, candidate decoding loss, retention loss and balanced binary write loss. The control computes and trains the same gate but overrides its publication decision to always write.
- 240 updates per arm, 32 fresh episodes per update, alternating recall and correction. Gaps 0, 4, 8, 16 each occur with both task kinds every eight updates.
- Fresh AdamW moments, learning rate 0.003, beta=(0.9,0.95), matrix decay 0.01, gradient norm clipped to 1.
- Runtime state resets at episode boundaries. Gradients detach every 16 events. Node computation continues on held publications, so expert budgets are exactly matched.
- Candidate cross-entropy at observed writes, retained-value cross-entropy at every causally eligible event, and balanced write/hold BCE are summed with existing router regularizers. Gate decisions use a hard zero-logit threshold. There is no straight-through estimator.
- All write and value labels are training-only. Forward receives observed bytes, ordinary availability and reset masks. No task identifier, target value, write position, or parser output is supplied to the model.

## Data and selection

Procedural streams use independent train/validation/test seeds and identifiers. Write timing varies independently by row. Distractors include bare A-D values and invalid '=x' strings. Corrections replace the old observed value; withheld controls erase the correction observations and score the old value. Source absence removes all observations. Reset controls clear graph state immediately before the final query.

Validate every 60 updates on 128 fresh fixed episodes for each of recall, correction and withheld correction at gap 16. Select by minimum accuracy, then mean accuracy, then lower mean NLL. Evaluate both the selected checkpoint and final endpoint; test data never selects checkpoints.

For each selected checkpoint, evaluate 512 test episodes per condition: recall/correction/withheld at gap 16, the same three at gap 64, reset, no writes and source absence. Also override learned publication to always write for the three gap-16 tasks. Save every prediction and target, decision confusion counts, exact hold counts and execution budgets.

## Verification and reproduction

Tests cover immutable holds, active/inactive rows, publication validity, partial reset, serialization, causality, chunk equivalence, label isolation and exact optimizer continuation. Audit all training batch hashes and paired budgets, regenerate all test predictions and validation selection, and check exported model parity.

```bash
python scripts/run_selective_publication.py
python scripts/audit_selective_publication.py --evaluations
python scripts/check_all.py
```

The driver records each completed task and supports recovery. Each trainer supports `--resume` with identical source and protocol. The 40-update pilot is retained separately and does not contribute to confirmatory results.
