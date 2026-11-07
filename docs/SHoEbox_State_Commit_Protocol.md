# Selective internal-state commits

November 6, 2026. Source base: `150c8a0`.

## Question and mechanism

Does protecting the producer's internal memory, as well as its published
payload, improve retention under interference? This adapts Raven's principle
of leaving unselected memory contents unchanged ([paper](https://arxiv.org/abs/2607.25357),
[author implementation](https://github.com/goombalab/raven)). It is a SHoEbox
experiment, not a reproduction of the Raven architecture.

The optional `state_commit=publication` mode uses the existing owner-or-hold
decision to commit each owner's contextualizer and three recurrent block
states. Unselected contents are bit-for-bit unchanged. All producer candidates
are still computed. Publication updates still occur after the current prediction;
the state decision changes future events only. Shared byte ingress and composer
state continue advancing, so the model retains current context while owners hold.
Event counts, execution clocks, and publication ages continue advancing.

The initial experiment couples state and publication admission. Read selection
does not grant permission to update state. Forced publication holds also hold
producer contents in this mode. Bootstrap commits initialize one empty owner at
a time. Resets clear all contents and ownership for the requested document rows.

Hard commits receive the existing straight-through write-probability surrogate
inside the live graph. Selected candidate contents receive gradients; held
contents retain the identity path. The existing delayed publication-credit trace
still conditions on historical candidates and features. It does not track the
new historical recurrent-state path across detached windows. No new state trace,
replay store, trainable coefficient, or candidate-compute shortcut is introduced.
Legacy configs default to `always`; weight and state schemas are unchanged.

## Matched comparison

| Arm | Producer contents | Read source | Delayed publication credit |
| --- | --- | --- | --- |
| Always | Advance every event | Stored publications | Yes |
| Held | Commit on publication | Stored publications | Yes |
| Always fresh | Advance every event | Current candidates | No |
| Held fresh | Commit on publication | Current candidates | No |

Fresh replaces the read values only. Memory-aware admission continues to consult
stored publications in every arm. In Held fresh, those admission decisions can
affect future producer contents, so this is a read-interface control, not removal
of every publication-mediated path. Publication shuffles also test that path.

---

Use initialization/data pairs 41/241 and 43/243. All arms use the rank-four
identity-initialized payload adapter, trained through the live graph only;
delayed payload credit is disabled. Each has 549,075 parameters and receives
600 updates: 200 acquisition, 200 bridge, 200 target. Batch eight, 64 bytes per
update, LR 0.002, AdamW betas 0.9/0.95, weight decay 0.01 and gradient cap one.
Loss is answer CE + 0.1 ordinary-byte CE + 0.001 routing auxiliary loss.
Each arm consumes 307,200 bytes and executes 18,432,000 experts.

The published gate-only and fresh arms from the payload-credit study supply
the Always and Always fresh controls. Restore their complete artifacts, verify
hashes against the archive manifest, and preserve provenance. Their configs,
initialization, data, objectives, and budgets match this study. Do not interpret
wall-time differences across the two sessions as a speed comparison. The runner
can reproduce all eight arms from initialization without the earlier archive.

Validate on 16 fixed-target documents every 100 updates and at stage ends.
Select minimum all-answer validation NLL; also report the final 600-update
checkpoint. Test both on 32 held-out documents and 32 tripled-delay documents.
Keep exact, semantic, context, and corrected-exact answers separate.

Run whole-query content interventions on selected and final checkpoints.
Publication shuffles preserve all recurrence. Producer shuffles preserve ingress,
composer, publications, and clocks, replacing only owner recurrent contents.
Also shuffle contents strictly older than 64 bytes. In held mode, producer
commit ages equal publication commit ages; in always mode, producer age is one
after a processed event, irrespective of publication age. Report actually changed
owner cells and queries separately from eligibility, and compare predictions on
the changed subset. Normal query-time writes remain enabled for these contrasts.
Commit age measures time since the last state update, not the age of every fact
that a frequently updated recurrent state may still encode.

## Verification and decision

Require unchanged initialization/RNG, legacy config loading, all-commit parity,
exact holds, owner-row isolation, reset and bootstrap behavior, matching expert
budgets, an independent surrogate-gradient oracle, and bit-identical checkpoint
continuation through stage changes and internal detach boundaries. Verify query
forks match the uninterrupted stream before interpreting interventions.

Compare Held versus Always, then Held fresh versus Always fresh. A retention
claim needs repeatable held-out and long-delay gains plus useful older contents;
fewer state commits alone is an implementation result. Keep the feature optional
until the behavioral comparison supports a default change.

```
python scripts/run_state_commit.py --out artifacts/state_commit --workers 4
python -m unittest experiments.text_regions.test_task_state_commit
python scripts/check_all.py
```
