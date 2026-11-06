# Task-driven memory pilot

2026-11-06. New optional mode; historical experiments remain intact.

## Objective

Connect ordinary byte input, owned publications, contextual reads, sparse expert computation, and later task credit in one continuous state. The model receives bytes and independent-stream resets only. It receives no entity, field, memory address, write label, or fixed latent target. Task masks and answers belong only to the objective and evaluator.

The procedural stream supplies observations, context-dependent questions, irrelevant facts and corrections. Answers are scored through the 256-byte decoder but are not inserted into the input stream. Related events share state. Fresh validation and test streams start with empty state.

## Memory and credit

The fixed-size memory consists of persistent publications owned by recurrent SHoE producer instances. A contextual query compares learned payload/header projections with current byte-context activity; a null alternative permits no read. The selected signal enters the composer before its sparse experts and contributes to its semantic routing. Reads use the pre-event snapshot; writes commit after prediction. Reading a publication does not mutate its owner.

Task gradients train the entire model within each active chunk. Straight-through write gates have discrete write/hold behavior and soft local derivatives. Across chunk boundaries a bounded forward-sensitivity trace carries the partial derivative of each surviving publication with respect to its small write-policy head. Later loss gradients at detached publication leaves contract with those traces. Current-chunk contributions are handled by ordinary autograd and are not injected again. Old traces are masked by reset, superseded according to the actual state update, and age-decayed.

This estimator follows the explicit publication-update path with candidate payloads and policy features held fixed in the delayed derivative. It is not full recurrent-gradient reconstruction. An exact small differentiable-horizon fixture measures this conditional derivative. Same-prestate intact versus blocked-read loss measures causal usefulness separately; unrelated-example loss differences never provide credit.

## Comparisons and selection

Paired initialization and byte streams are used for integrated credit, delayed credit disabled, and no-memory training. The no-memory arm runs the same sparse experts and computes the same read/write policies, then zeros memory contribution. Evaluation also blocks reads and substitutes shuffled or stale publications without reducing expert execution count.

Select checkpoints by mean validation answer negative log likelihood, with ordinary-byte loss reported separately. No universal exact-recall threshold is imposed. Report final-budget results as well as the selected checkpoint. Evaluate exact value accuracy, approximate category accuracy, correction response, long-delay performance, null-query interference, external bytes, expert executions, read/write rates, state and trace bytes, and wall time.

A pilot must demonstrate useful held-out recall before broader three-seed comparisons or the 24.9M scale-up. If only infrastructure or some behavioral criteria pass, report those separately and leave the unmet phase exits open.

## Reproducibility

The runner saves configuration, paired data digest, per-evaluation history, model and optimizer, carried state, credit traces, stream cursor and RNG state. It supports deterministic continuation. The delivery includes metrics, representative causal traces, a resumable checkpoint, source, and a concise Markdown/PDF report.

## Executed budgets and follow-up

The initial comparison uses initialization seed 17 and stream seed 123. Each arm receives 600 updates, batch 16, chunk length 16 (153,600 external bytes), AdamW at 0.002, answer CE plus 0.1 ordinary-byte CE and 0.001 routing auxiliary loss. Ordinary targets are masked at answer positions. Validation evaluates 8 fresh documents every 100 updates; selected and final checkpoints each evaluate 16 fresh test documents and 16 longer-delay documents.

After the seed-17 pilot demonstrated a held-out memory contribution, the same three arms were repeated with initialization/stream pairs 18/124 and 19/125. Each pair shares data, initialization and expert budget; checkpoints minimize validation answer NLL. The ordinary-byte stream here is procedural task text, not a natural-language corpus.

The initial integrated arm committed 97.4% of owner-byte opportunities at update 100. A separate seed-17 follow-up adds 0.02 times mean write probability to the objective, with delayed credit enabled and disabled. This is an explicitly reported cost for frequent writes, not a write-label target. Lower commit frequency alone does not establish useful retention; query publication ages and per-owner utilization are recorded.

Query-time blocked, shuffled and earlier-publication interventions fork the same pre-query state. Full read-blocked rollouts additionally start from empty state. Stale publications come from the first query's pre-event snapshot while current ingress and recurrent node states are preserved. Inference timing with interventions includes their diagnostic forwards; the reported live expert count excludes diagnostic branches. Valid bytes and padded executed bytes are reported separately.
