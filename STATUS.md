# SHoEbox handoff

## Completed: regional recall and mixed learning

Base `37a94dc`. `RecallWorkspace` now lives in `recall.py`, with a compatibility
import in `task_recall.py`. Ordinary graph receivers can use independent recall
slots over their configured incoming ports. Original forward/delayed timing,
masks, substitutions and state ownership are preserved. New output feedback
advances independently of held public outputs. Zero slots preserves legacy
weights, initialization and behavior.

## Completed evidence

- All 530 checks across five suites pass; the live-output example also passes.
  Ten new checks cover regional behavior and mixed-stream continuation.
- Seven-slot ordinary graph: seven nodes, two rounds, 311,711 parameters,
  32 updates, 2,048 bytes, 344,064 expert executions. Validation NLL falls from
  5.5321 to 4.3244. Separate 128-byte test: intact 4.2228, blocked reads 4.4871.
  All seven receivers learn role maps and feedback. Exact stream continuation
  reproduces logits and state. Added feedback state: 896 bytes per stream.
- Paired mixed continuations from the preceding distinct recall arm's final
  checkpoint: 320 updates each, 564,519 parameters, 81,920 natural-text bytes
  and 81,920 memory bytes, 9,830,400 expert executions. All weights train.
- Both sources use state holding and delayed payload credit, with independent
  runtime state, eligibility and cursors. Both produce delayed gate/payload
  gradients. Resume verifies corpus, config, schedule and stream fingerprints.
- Both arms select update 320. Distinct/pooled test natural-byte NLL:
  2.5220/2.5703; exact-answer NLL: 1.1422/1.1571; long-delay: 1.1612/1.1792.
- Distinct full-rollout interventions: pooling roles changes language/all-answer
  NLL by +0.0051/-0.0006; disabling feedback by +0.1347/-0.0003; blocking reads
  by +0.2563/+0.2639. These cover full rollouts, including later writes/state.
- The ordinary graph and owned-memory training are separate configurations
  sharing the recall module. Ordinary graph write gates still use their prior
  training path; they have not acquired the task model's delayed credit.

[Report](docs/SHoEbox_Regional_Recall_Report.md) |
[PDF](docs/SHoEbox_Regional_Recall_Report.pdf) |
[metrics](experiments/text_regions/results/regional_recall/summary.json).
The artifact archive includes source, exact corpus, common parent, selected/final
mixed checkpoints, graph checkpoint, optimizer/runtime/credit state, raw traces,
checks and hashes. The suite runner returned zero; its redirected stdout snapshot
is partial, and the final source's 530 tests were independently enumerated.

## Next development target

Bring the task-trained publication and delayed-credit interface into the ordinary
regional graph, then increase mixed-data exposure through the unified path. Keep
state holds, payload credit, distinct roles and feedback independently controllable.
The new mixed run supports useful recall and language feedback; it does not isolate
older-publication retention. Use the existing query-fork audits for that question.

Retain well-motivated mechanisms with trainable controls. Small probe outcomes
are not feature-removal gates. Comments mark role specialization and feedback as
potentially training-threshold-dependent features for later ablation. No replay
store is added; slots are current views of existing owned publications.

## Reproduction

```bash
python scripts/run_regional_recall.py --parent /path/to/parent.pt \
  --data data/tiny_shakespeare.txt --out artifacts/regional_recall
python scripts/audit_regional_recall.py
python scripts/summarize_regional_recall.py
python scripts/check_all.py
```

The paired runner resumes existing checkpoints. Graph training is a separate
32-update invocation documented in the experiment README. Pauses between
scheduled evaluations preserve the checkpoint-selection schedule.
