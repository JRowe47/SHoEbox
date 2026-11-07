# Delayed credit for owned payload encodings

November 6, 2026. Source base: `d47ddc6`.

## Question

Does credit from later publication reads improve the representation written by
an earlier event? The existing detached trace trains write-policy coefficients.
This experiment adds a separate derivative for a small payload adapter while
holding the reader, producer, curriculum, and computation budget fixed.

This implements the engineering proposal in the supplied recent-paper review.
The short-delay mixture, multiple read slots, Raven-style internal-state holds,
and time-aware addressing remain separate experiments.

## Mechanism

For each owner, adapt the producer output `x` using four coefficients `a` and
four fixed orthonormal directions `U`:

```
z = tanh(x @ U.T)
candidate = x + (a * z) @ U
```

Zero coefficients give an exact identity. Directions are deterministic DCT rows
and do not consume the model's initialization RNG. Write-policy features use
the original producer output; the adapter changes payload content only. All
existing producer, router, composer, and read parameters remain trainable.

An owned publication keeps the `z` of its actual last commit. A hold preserves
it exactly, replacement discards its predecessor, and row reset erases it.
There is no observation history or replay store. The historical payload
Jacobian factors into `diag(z) @ U`, so the trace stores four scalars per owner
and an owner-version identity instead of a payload-by-parameter matrix.

After a detached window's backward pass, the memory leaf gradient `g` yields
`sum_batch((g_payload @ U.T) * z)` for the adapter coefficients. Add this once
before ordinary global clipping and AdamW. Ordinary autograd already supplies
the current window's derivative and is not repeated. Bootstrap and forced
writes still carry payload sensitivity; they do not acquire write-policy credit.

This is the conditional derivative through committed payload adaptation with
historical producer features and write decisions held fixed. It does not restore
the complete historical producer/recurrent graph or differentiate through past
optimizer updates. Semantic-header derivatives retain their previous behavior.

With four owners, width 80 and rank four, the adapter adds 16 trainable
parameters and 1,280 bytes of fixed directions. At batch eight the persistent
payload trace adds 768 bytes, including its version identities. It does not grow
with stream length. Legacy configuration defaults to rank zero, keeps the
original eligibility layout, and loads existing checkpoints.

## Paired study

Initialization/data pairs are 41/241 and 43/243. Each has three matched arms:

| Arm | Read source | Delayed write credit | Delayed payload credit |
| --- | --- | --- | --- |
| Gate only | Stored publications | Yes | No |
| Payload | Stored publications | Yes | Yes |
| Fresh | Current producer candidates | No | No |

Every arm contains the same identity-initialized rank-four adapter and trains it
through the live graph. Every arm computes and carries both eligibility traces.
Only injection of the historical payload derivative changes between the first
two arms. The trained-fresh arm uses the existing fresh-read control.

Each arm runs 600 updates, batch eight, 64 bytes per update: 307,200 external
bytes and 18,432,000 expert executions. The model has 549,075 parameters. Use
200 acquisition, 200 bridge, and 200 target updates, with all original data
generation, answer density, optimizer settings and stage-reset rules retained.
AdamW uses LR 0.002, betas 0.9/0.95, weight decay 0.01, global gradient cap one.
Loss is answer CE + 0.1 ordinary-byte CE + 0.001 routing auxiliary loss.

Validate on 16 fixed-target documents every 100 updates and at stage ends.
Select the lowest all-answer validation NLL. Evaluate selected and final
checkpoints on 32 held-out test and 32 tripled-delay documents. Report exact,
semantic, context, and corrected-question performance separately. Run the
existing whole-query content interventions on both stored-publication arms,
including normal-write shuffles of content older than 16 and 64 bytes. The
64-byte cutoff specifically exceeds the complete live gradient window. Preserve
changed-cell counts and the actually changed-query subsets.

Log injected payload-gradient norms, the number of updates receiving nonzero
delayed payload credit, and actual AdamW coefficient displacement. The last
quantity includes both live and delayed gradients; it is not their causal
decomposition. The independent gradient oracle separately checks applied AdamW
updates against full-history autograd and against disabled delayed credit.

## Verification and decision

Require identity/initialization equivalence, conditional gradient agreement
with full-history autograd, actual held-publication gradient agreement in the
regional model, replacement/reset/version checks, and bit-identical resume of
weights, optimizer, state and both traces across stage transitions and internal
detach boundaries. Run the preserved experiment suites.

Compare payload versus gate-only first, with trained fresh as the retention
control. Keep exact test NLL, longer-delay NLL, corrections, and old-content
sensitivity visible for each seed and checkpoint. An implementation passes
when its derivative/lifecycle checks pass; a retention improvement requires
consistent behavioral gains and useful older content. Do not change the
default architecture on gradient connectivity alone.

## Reproduction

```
python scripts/run_payload_credit.py --out artifacts/payload_credit --workers 3
python -m unittest experiments.text_regions.test_task_payload
python scripts/check_all.py
```

The runner resumes existing arms and saves selected, final, acquisition, and
bridge checkpoints with optimizer/state/stream/RNG information. It also saves
raw histories, final evaluations, query audits, and per-arm completion records.
Use `--seeds`, `--arms` and `--workers` to divide the same fixed study budget.
