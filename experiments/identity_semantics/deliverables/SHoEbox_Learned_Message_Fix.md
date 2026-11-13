# SHoEbox: learned-message repair

13 November 2026 | Base: 9566f4f | 1,031,936 parameters | Two fresh seeds

## Result

Implemented learned constituent transfer, independently bounded feedback for activation and routing, and credit to the actual adapter or neural producer. Stable IDs, structural descriptors, role binding and relay forecasts remain intact. The new runtime is selected with `fresh_model(kernel="messages")`; `relays` remains available as the unchanged comparator.

Mean held-out loss increased from **1.940301 to 1.944734 nats/byte** (+0.004434). Same-core median training throughput changed from **3275 to 2215 bytes/s** (-32.4%).

| Seed | Relay NLL | Message NLL | Change |
| --- | --- | --- | --- |
| 1 | 1.951268 | 1.949225 | -0.002044 |
| 2 | 1.929333 | 1.940244 | +0.010911 |
| Mean | 1.940301 | 1.944734 | +0.004434 |

Each model trained on one million next-byte targets from the same Tiny Shakespeare stream. Evaluation used 78,771 reserved test targets with fresh stream state and causal adaptive associative memory. Six 100,000-target pilots on seed 3 selected controls using a separate 32,768-target validation segment. Both full-run variants used peak learning rate 0.08, the same cosine schedule, local rate LR/3, and router rate 0.002. No test data selected controls.

## Implemented behavior

- Parents consume a normalized, bounded role-transformed bundle of `child.code - child.structure` in their content input and routing cue. Exported codes retain their local adapter provenance; received payloads are not relabeled as parent-produced adapter vectors.
- Declared delayed feedback adds its own bounded branch to activation and routing. The legacy context path remains. With all three new channel gains zero and semantic LR multiplier one, training exactly reproduces the relay runtime.
- Semantic credit uses saved source occurrences, roles, features and derivatives. It updates the actual semantic adapter, including through held publications and relays. Feedback uses the saved neural producer, chosen/runner keys, sparse mask and smooth gate derivative. The hard selection itself is not differentiated.
- The two feedback terms for the same consumed occurrence share one reuse count. Source freeze and task-rate controls apply to transferred semantic credit.

Selected gains: **payload 1.0; feedback activation 0.1; feedback routing 0.1; semantic LR multiplier 1.0**. The independently normalized branches are bounded by `tanh`; all channel gains are constrained to [0, 1]. No neural parameters were added.

## Reproduction

From `experiments/identity_semantics`, restore the pinned corpus with `python src/restore_artifacts.py --download-data`. Run `python src/run_messages_study.py` with `pilots`, then `full`; run `python src/check_messages_results.py` with `interventions`, then `timing`. Unit checks: `python -m unittest discover -s tests -v`. Native checks: `python tests/run_messages_sanitizer.py`.

<!-- pagebreak -->

## Regional contribution

Entries are two-seed mean changes in test NLL when the named path is removed. Positive values mean removal hurts prediction; negative values mean removal helps. These interventions keep the remaining dynamics live and are not additive shares of knowledge.

| Region | Old expert removal | New expert removal | New neural removal |
| --- | --- | --- | --- |
| E4 | -0.00001861 | +0.00000528 | +0.00008329 |
| V4 | -0.00000201 | -0.00000836 | +0.00000426 |
| G5 | +0.00000783 | +0.00043769 | +0.00064972 |
| F4 | +0.02303623 | +0.02257785 | +0.02007158 |
| L3 | +0.03868643 | +0.03647032 | +0.02979158 |

`expert_transform` removes the learned matrix/bias while preserving the input skip. `neural_output` removes the full published neural response, including its skip and recall components.

| Region | Semantic transfer removal | Feedback branch removal | Whole-source removal |
| --- | --- | --- | --- |
| E4 | +0.00057858 | +0.00000426 | +0.00140366 |
| V4 | +0.00003690 | +0.00000000 | +0.00032036 |
| G5 | +0.00000000 | +0.00033687 | +0.00060339 |
| F4 | +0.00680406 | +0.00001829 | +0.16825823 |
| L3 | +0.01260091 | +0.00000036 | +0.30668940 |

`semantic_payload` removes only outgoing adapter residuals at their consumers. `feedback_channel` removes only the newly added bounded feedback branch, keeping the legacy context path. G5 has no immediate constituent consumers, so its semantic-transfer intervention is intentionally inactive. Whole-source removal also changes content/event transport.

All **100 of 100** interventions that promise content-event preservation passed exact identity/provenance trace checks (100 verified). Routing changes caused by feedback or payload removal are intentional. 120 full-test interventions and 4 exact production/diagnostic parity comparisons completed. Semantic-residual and relay-context measurements are included in the machine-readable results.

G5's learned-expert removal penalty is now +0.00043769, versus +0.00000783 previously. E4 and V4 learned-expert effects remain near zero. E4's transferred semantic payload now contributes +0.00057858 nats/byte when removed. These measurements show greater G5 influence and usable semantic transport, while the mean test loss still favors the relay comparator. Keep `relays` for current training; `messages` is available for continued channel/credit experiments.

<!-- pagebreak -->

## Delivered task credit

Mean delivered expert-credit norm per update, averaged across seeds. These are gradient magnitudes before the parameter update, not parameter-change norms. Semantic-adapter deliveries are recorded separately.

| Region | Old expert credit norm | New expert credit norm | Ratio |
| --- | --- | --- | --- |
| E4 | 0.00031929 | 0.00047917 | 1.50x |
| V4 | 0.00036922 | 0.00054012 | 1.46x |
| G5 | 0.00364620 | 0.00538416 | 1.48x |
| F4 | 0.05827921 | 0.05753348 | 0.99x |
| L3 | 0.06910594 | 0.06822359 | 0.99x |

Both repaired runs completed with zero missing legacy credit parents, zero expired retained-input edges, zero missing feedback producers, and zero expired or duplicate forecast producers. The saved forecast accounting balances exactly.

## Verification

All 199 repository unit checks passed, including 132 checks in the repaired composition package. All 17 new tests also passed at 64 experts. Native ASan/UBSan passed. The final production binary exactly matched the training binary for 4,096 evaluation targets and 4,096 continued training targets, including logits, parameters and full state.

The new tests cover: fixed-ID payload sensitivity on E4 -> G5/EC-X, V4 -> G5 and L3/F4 -> Y; finite differences for semantic payloads and G5 feedback to F3/E4/F4/V4; the saved smooth response gate; expert switching under the normal sparse mask; held and relayed occurrence credit; source freezes and zero rates; exact zero-control parity; diagnostic isolation; pending-forecast teaching; checkpoint continuation; and explicit old-checkpoint conversion. Native ASan/UBSan exercises training, evaluation, draining, reset and state import/export.

The throughput comparison alternated six trials per runtime on CPU core 0, with 1,024 warmup targets followed by 8,192 measured training targets. Compilation, loading, snapshot copies, warmup and checkpoint I/O are excluded. Concurrent full-study run times are not used to calculate the throughput comparison.

## Use and continuation

```python
import sys
sys.path.insert(0, "experiments/identity_semantics/runtime")
from runtime import fresh_model, Model
model = fresh_model(seed=1, kernel="messages")
# Resume a new-format checkpoint exactly:
# model = Model.load("messages.npz")
```

Message checkpoints use version 32 / composition ABI 6 and persist the controls and eligibility. Old relay checkpoints retain version 31 / ABI 5. Convert explicitly at a fresh stream boundary with `python src/convert_to_messages.py old_relays.npz new_messages.npz`. Conversion preserves weights, semantic adapters, dictionaries, fixed transforms and preprocessing means; it resets stream state, forecasts, eligibility and associative memory. It does not overwrite the source checkpoint.

## Evidence

Source and replay commands are documented in the package README. Exact compact measurements are in `results/message_*.json`; source hashes are in `SOURCE_MANIFEST.json`. Test checkpoints, corpus bytes, compiled binaries and intermediate runs are excluded.

