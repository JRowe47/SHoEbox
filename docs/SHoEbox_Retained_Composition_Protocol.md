# SHoEbox: retained facts and live composition

Locked November 5, 2026 after one 80-update development pilot, before confirmatory training and test evaluation.

## Target and architecture

Connect the learned publication producer to the existing seven-node recurrent composition model. An eighth node, F, owns an independent byte HAT and expert sharing bank copied from a previously trained producer. Its retained latent reaches A through one ordinary forward edge. B and C receive fresh permutation tables; the common P/Q byte head predicts g(f(x)), where x exists only in F's retained publication at inference. No decoded intermediate byte, fact label, parser output or task ID enters the query.

The new source ownership option defaults off and requires an exclusive explicit input view. Existing model parameters, byte paths and behavior must remain exactly compatible. Query workspace reset clears A/B/C/U/P/V/Q and their ingress state while preserving F's context, recurrent state, payload, header and provenance. A new workspace epoch distinguishes new publication versions. Queries do not execute F or overwrite its memory.

This is component integration in one eight-node graph, not completion of M7. Existing producer and composer weights are frozen. Only the new edge's 32-by-32 value matrix is trained; its gate is a constant 0.5 and semantic context projection is zero. Regional experts still perform the relation lookups and two-round composition. No output decoder is added.

## Locked runs

- Producer checkpoints: uncalibrated selected publication models from seeds 20261110, 20261111, 20261112; all three are included without choosing on test results.
- Training seeds: 20261120, 20261121, 20261122, paired in that order.
- Composer: unchanged selected full-gradient-horizon checkpoint, SHA256 bc1291e5c68f9cb2bbce338b428eae4caf78b080b9879220de9cd85b9a417850.
- 160 updates, 32 fresh episodes each, alternating recall/correction. Gaps 0,4,8,16 each occur with both tasks in eight updates. No cached example replay.
- AdamW learning rate 0.01, beta=(0.9,0.999), matrix decay 0.01, gradient norm clip 1. Fresh optimizer and deterministic source hashes are saved.
- Loss: common final-byte cross-entropy plus normalized mean-square alignment between the edge output and the original A ingress representation of the actually observed fact byte. The latter is training-only local interface supervision from procedural observations. No external model or corpus supplies targets.
- Source memory executes with 16-event chunks and detached carried state. Composer prefix and both query rounds remain differentiable through the fitted edge.
- Validate every 40 updates on 128 episodes per task at gap 16. Select by minimum task accuracy, mean accuracy, then lower mean NLL; test data never selects the checkpoint.

## Held-out evaluation

Engineering target: at least 90% final composition accuracy separately for recall, correction and withheld correction, 512 fresh fact episodes per condition and seed at gap 16. Require at least five percentage points K=2 gain over K=1 and exact F state/publication retention across repeated live queries. Report all seeds and failures.

Fact identities and random streams are split-specific. Relation-table pairs use the established disjoint train/validation/test split. Table selection is independent of the retained fact value. The target is the composition of the actually observed value; withheld corrections score the old value.

Report the three tasks at gaps 16 and 64. For correction at gap 16 also report K=1, K=4, two rounds without delayed exchange, blocked F-to-A edge, reset memory, shuffled whole memory publication, and forced-always publication using the same weights. Shuffled memory is scored both against the original answer and its counterfactual answer. Controls at the same K retain identical node/expert execution budgets. Four consecutive fresh table queries test an unchanged memory. A separate sequence asks three queries before observing a correction, then scores corrected and withheld branches.

## Verification and delivery

Verify exact transplanted producer behavior, parameter isolation, state/provenance preservation, causal label isolation, exclusive source ownership, chunk equivalence, query controls and exact optimizer resume. Reconstruct every training batch and table choice, checkpoint selection, frozen weights and execution totals. Independently regenerate held-out predictions and inference-export outputs. Check unchanged original composition predictions and run all regression suites.

Save complete resumable checkpoints, inference exports, raw predictions, source hashes, audit results and the development pilot separately. Publish source and results, and attach source/artifact ZIPs plus Markdown/PDF reports.
