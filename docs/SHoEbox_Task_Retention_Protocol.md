# Task-memory retention and whole-query access

2026-11-06 | Source base fd4d1ac

## Question

Does persistent publication storage improve predictions beyond the same recurrent experts supplying their current outputs? Can learned writes keep useful older publications available while current-query computation continues?

The preceding three-seed result established a useful publication path. A new audit forks before the entire query: terminal-token read statistics cannot establish when retrieval happened. Its held-publication control separates pre-query committed content from fresh publications made while processing the query. A trained fresh-output control retains all producer/composer recurrence and expert executions but supplies current candidates directly at the read interface.

## Mechanisms

The optional memory-aware write policy sees its candidate plus detached current publication, validity, log age, and payload/header cosine similarity. Its new projection starts at zero; common model parameters and initialization RNG remain identical. No task label, entity, field, query position or memory-slot address enters the policy.

An optional competitive writer chooses one owner or no owner per external byte. Its fixed null score is zero. Conditional delayed sensitivities include the softmax competition between every pair of policy heads. The trace remains bounded and serializable; exact small-horizon fixtures compare its gradient with the matching explicit autograd graph. Historical conditioning features remain detached.

Generic initialization commits once to each previously invalid owner, one owner per byte, at independent-stream startup. These forced writes receive no gate credit. This prevents a permanently invalid owner from being mistaken for successful selective holding. Explicit hold controls override initialization. Both conditioned main arms use the same initialization rule; legacy behavior remains available unchanged.

## Stream and comparison

The retention preset has four recurring entities, one attribute, four possible value bytes, two related cycles, four memory questions per cycle, context-only questions, and minimum delays 2/48 bytes. Delay gaps contain changed-value observations of other entities. An observation irrelevant to the current question can become relevant to a later one. Questions and observations are raw bytes; answers remain outside the input stream. Only independent-document boundaries reset state.

The first paired pilot uses initialization/stream seeds 29/229. Five arms each receive 800 updates, batch 16, 16-byte chunks (204,800 external bytes), AdamW 0.002, answer CE + 0.1 ordinary-byte CE + 0.001 routing auxiliary loss. No write cost is imposed.

1. Legacy independent writer, unchanged policy features and initialization.
2. Conditioned independent writer with empty-owner initialization.
3. Conditioned competitive writer with initialization and delayed credit.
4. The competitive model with delayed credit disabled.
5. The competitive model trained with current candidate reads instead of persistent publications.

Arms 2-5 share all parameters and initial weights. All five execute the same sparse expert budget. The legacy arm has 864 fewer policy-conditioning parameters. Validation uses eight fresh documents every 100 updates; checkpoints minimize mean answer NLL. Selected and final checkpoints score 16 fresh test and 16 longer-delay documents. Report exact-value questions separately from semantic and immediate-context questions.

## Evidence and next decision

Audit full-query intact, blocked, held-publication, held+blocked, content-shuffled, and current-candidate forwards from identical pre-query states, without mutating the live stream. Record read weights, versions and publication ages at every query byte. Report source-fact delay separately from publication age; fewer writes, occupied owners and older states alone do not establish useful retention.

Primary evidence for persistent-publication value is lower held-out exact-task loss than the trained fresh-output control, supported by shuffling pre-query publication content while allowing normal query writes. Holding every publication also disables the fast expert communication path, so held controls remain diagnostic. An age-filtered shuffle targets pre-query cells older than 16 bytes; report eligible cells, actually changed content, and effects separately. Report whole-query read effects, correction response, old-read mass, owner utilization, forced versus learned commits, state/trace bytes, and matched byte/expert budgets. Repeat a successful pilot on two more paired seeds before claiming a stable advance. If a criterion remains unmet, preserve the result and identify it directly rather than scaling to 24.9M.

## Paired continuation follow-up

At 700/800 scratch updates, validation answer losses remained near 1.0 across all arms. The harder stream places many first uses outside the 16-byte gradient window; delayed credit trains write heads but does not train historical payload encoders. Complete the fixed scratch budget, then separately compare conditioned-independent, conditioned-competitive and trained-fresh continuations initialized from the same published seed-17 integrated best checkpoint. Use the same 29/229 seeds, stream, optimizer settings, 800-update budget and selection rule as the scratch pilot. Reset AdamW, all runtime states, eligibility traces and stream cursors; all model parameters remain trainable. The optional conditioning projection starts at zero. Record the parent SHA256 and its completed training budget; do not merge continuation and scratch results. This follow-up tests whether learned representations help acquisition on the harder stream.
