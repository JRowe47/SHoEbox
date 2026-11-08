# Cue timing and expert-input reinstatement

This pass adds two independent controls to the explicit hippocampus. The default
configuration preserves the original implementation and old checkpoints.

| Control | Original | New experiment |
| --- | --- | --- |
| `cue_clock` | `round`: advance the regional GRU at each native round | `event`: advance once per incoming byte and reuse the live cue across rounds |
| `reinstatement` | `ingress`: add the readback before the regional HAT adapter | `expert`: add it after that adapter, before expert routing and computation |

The owner-local reader still interprets the capsule. Selection, role composition,
CA1 acceptance, admission, revision, conditional reads, native recurrent state,
publication and causal commit stay learned and in their existing owners. The new
path adds no parameters. Its zero compose projection preserves the parent warm
start. Query identities and answers never enter the inference API as metadata.
The controls remain ablatable because useful binding may depend on capacity and
training beyond this small model.

The motivating audit measures actual query separation and readback influence on
a separate validation bank (seed 881, document start 4000). It uses observational
hooks and a blocked-read intervention; no diagnostic decoder is installed.

## Fixed comparison

Start from the retained parent, carrying AdamW moments, language state/cursor,
RNG and training document position. Train 64 updates, with the existing
paired/paired/single/correction schedule, LR 0.003, global clipping 1.0, language
weight 0.25, syntax weight 0.1, and gaps 0/4/8/16/32. Compare against the original
64-update shared-hippocampus endpoint and the retained parent. Source ledgers
are checked, including the full sequence of modes, gaps, counts and documents.

Validation remains seed 683/start 3000, eight templates, at updates 16/32/48/64.
Promotion requires both-correct accuracy at least 50% at gaps 0 and 32, single
accuracy within two points of the parent, correction within three points, and
natural-text NLL no more than 0.05 higher on matching validation cohorts.

The additional final bank is locked to test seed 1511/start 120000, overriding
the original evaluator's default bank. It uses 16 paired/correction templates,
32 single templates, pair gaps 0/32/64, single/correction gap 64 and 1,024 natural
bytes. This bank does not select weights. Three fixed-weight controls block
readbacks, restore the round cue clock, or restore ingress reinstatement.

## Reproduction and recovery

Install the repository dependencies and restore the exact corpus and checkpoint
archive into this source tree. The archived original A initial validation is
reused only for the proven zero-residual initialization.

```bash
python -m unittest experiments.text_regions.test_hippocampus experiments.text_regions.test_hippocampus_binding experiments.text_regions.test_content_checkpoint experiments.text_regions.test_runtime_fusion
python scripts/run_hippocampus.py --arm C --cue-clock event --reinstatement expert --out artifacts/hippocampus_binding/event_expert --steps 64 --stop-after 16 --checkpoint-training --checkpoint-min-positions 384 --validation-workers 2 --initial-reference artifacts/hippocampus/A
python scripts/run_hippocampus.py --arm C --cue-clock event --reinstatement expert --out artifacts/hippocampus_binding/event_expert --steps 64 --resume --checkpoint-training --checkpoint-min-positions 768 --validation-workers 3
python scripts/finish_hippocampus_binding.py --workers 3
python scripts/diagnose_hippocampus_binding.py --checkpoint artifacts/hippocampus/C/last.pt --out artifacts/hippocampus_binding/before_audit.json
python scripts/diagnose_hippocampus_binding.py --checkpoint artifacts/hippocampus_binding/event_expert/last.pt --out artifacts/hippocampus_binding/after_audit.json
python scripts/summarize_hippocampus_binding.py
python scripts/report_hippocampus_binding.py --out output
```

Fresh training requires an empty output directory. Move the archived new arm
aside or use a different `--out` for a new reproduction. `--resume` restores the
last completed checkpoint with identical protocol/configuration. Both the fixed
endpoint and validation-selected checkpoint are retained; update 16 is also
preserved as the explicit execution-policy boundary. The archival manifest
records source identities, execution phases and exact final cohorts.

Activation recomputation changes storage, not the credit window. Both new and
old runs retain live gradients through history and detach at optimizer updates.
The different recorded recomputation schedules can change floating-point
accumulation order. Configuration migration normalizes omitted default fields,
while still rejecting actual configuration/protocol mismatches.

[Completed report](hippocampus_binding_report.md) and
[machine-readable results](../experiments/text_regions/results/hippocampus_binding/summary.json)
record the final outcome and promotion decision.

## Bounded CA1 reread acceptance

The completed first phase exposed a second control issue. Legacy CA1 mixed the
original and refined returns, then multiplied both by the original accept
probability. Since accept, null and refine compete in one softmax, choosing
refine suppresses the very reread it requested.

The optional `ca1_policy='bounded'` separates the two accepted branches:
`accept * original + refine * terminal_accept * reread`. After the local reread,
CA1 compares the returned descriptor again. Its terminal decision has only
accept/null actions; refinement is unavailable once its budget is exhausted.
Disabling refinement also removes that unavailable action from the initial
decision. Null can still reject the result. The contributor receipt uses the
actual combined acceptance. Transport fetches, native rounds, parameter count
and global learning objectives are unchanged. `legacy` preserves earlier
checkpoint semantics and inference controls.

Two matched 16-update continuations start from the completed event/expert
checkpoint, including all weights, AdamW moments, RNG, language state/cursor and
next training document. The existing mode/gap cycle restarts for both branches;
no documents are replayed. Initial state and every subsequent source-ledger row
are checked for equality. These are additional updates, separate from the first
64-update experiment. The final endpoint is primary; initial and final validation
use the same locked eight-template cohorts as above.

The continuation test bank is seed 1699/start 160000: eight paired/correction
and 16 single templates, paired gaps 0/32/64, single/correction gap 64 and 1,024
natural bytes. Same-weight interventions block readbacks or restore the legacy
CA1 policy in the corrected endpoint.

```bash
python -m unittest experiments.text_regions.test_hippocampus_ca1
python scripts/run_hippocampus_ca1.py --policy legacy --out artifacts/hippocampus_binding/ca1/legacy
python scripts/run_hippocampus_ca1.py --policy bounded --out artifacts/hippocampus_binding/ca1/bounded
python scripts/finish_hippocampus_ca1.py
python scripts/diagnose_hippocampus_binding.py --checkpoint artifacts/hippocampus_binding/ca1/legacy/last.pt --out artifacts/hippocampus_binding/ca1/legacy_audit.json --gaps 32
python scripts/diagnose_hippocampus_binding.py --checkpoint artifacts/hippocampus_binding/ca1/bounded/last.pt --out artifacts/hippocampus_binding/ca1/bounded_audit.json --gaps 32
python scripts/summarize_hippocampus_ca1.py
python scripts/report_hippocampus_binding.py --out output
```

The continuation runner accepts `--resume` for its own matching source/protocol.
Exact-resume source-hash guards remain enforced: use the original revision to
resume an earlier run under its original code, or start an explicitly recorded
continuation with the updated controller. Loading old checkpoints for inference
remains supported. A direct check against public revision `59244b3` reproduces
legacy logits exactly for 16 trained-checkpoint histories and 32 query branches.

The exact 64-update cue/reinstatement source is preserved at public commit
`2e5bcce37984da7e840060a95405599bb39fd986` (tree
`ee654bf2e00fa59fc7aa686558d7601ecf010ec4`), identical to the local training
revision `94ba370`. This source predates the CA1 change and supports the original
update-16 resume boundary. The subsequent release contains the bounded CA1
controller, matched continuations and completed evidence.
