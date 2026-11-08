# SHoEbox hippocampus: retrieval control repair

November 7, 2026 (Mountain Time)

The corrected endpoint does not pass the declared validation gates; keep the original retained parent as the working model. This pass completes a 64-update cue/reinstatement experiment and two matched 16-update controller continuations. All three controls remain independently selectable, with legacy defaults preserved.

## The CA1 correction

The audit exposed a suppression loop: accept, null and refine compete in one softmax, but the legacy controller multiplies the refined return by the original accept probability. When refine wins, it suppresses the reread it requested.

The corrected bounded policy delivers `a*v + r*a_after*v_refined`. The reread gets a fresh terminal accept/null decision, and its contributor receipt uses actual combined acceptance. Legacy delivery is `a*((1-r)*v + r*v_refined)`. Null can still reject either branch. A disabled or exhausted refinement budget cannot consume probability through an unavailable action.

The fix adds no parameters, transport fetches or native graph rounds. It rereads the selected owner capsules already in the bounded query and reuses the learned comparator.

## Matched 16-update continuation

Both arms start from identical update-64 weights, AdamW state, RNG, carried language state, stream cursor and next document. Only CA1 policy differs. The fresh test bank uses seed 1699/start 160000 and 96 conflicting histories per delay; both questions about a history must be correct.

| Controller | Both +0 | Both +32 | Both +64 |
| --- | --- | --- | --- |
| Legacy CA1 | 0/96 | 1/96 | 0/96 |
| Bounded CA1 | 0/96 | 0/96 | 0/96 |

| Controller | Single +64 | Correction +64 | Text NLL |
| --- | --- | --- | --- |
| Legacy CA1 | 98.4% | 79.7% | 2.955399 |
| Bounded CA1 | 98.4% | 97.7% | 2.976824 |

Retention uses 64 single-fact and 128 correction answers. Text NLL uses 1,024 held-out natural bytes; lower is better. These tests do not select checkpoints.

The corrected arm still favors the newest fact: immediate newer-value accuracy is 100.0%, older-value accuracy is 25.0%, and changing the question switches 0/96 conflicting-pair outputs. Query-conditioned selection remains the next development target.

| Same final weights | Both +0 | Both +32 | Both +64 |
| --- | --- | --- | --- |
| Intact bounded | 0/96 | 0/96 | 0/96 |
| Readbacks blocked | 0/96 | 0/96 | 0/96 |
| Legacy policy restored | 0/96 | 0/96 | 0/96 |

## Earlier placement experiment

The first change advances the regional cue GRU once per incoming byte instead of once per internal round. The second places recalled evidence after the regional HAT adapter, directly before expert routing and computation. Capsule interpretation remains with the owner. Global objectives and all-parameter training are unchanged.

This 64-update comparison starts from the original strong parent. Its separate held-out bank uses seed 1511/start 120000 and 192 conflicting histories per delay. It must not be pooled with the later 96-history bank.

| Model | Both +0 | Both +32 | Both +64 |
| --- | --- | --- | --- |
| Retained parent | 0/192 | 0/192 | 1/192 |
| Previous hippocampus | 5/192 | 0/192 | 1/192 |
| Event cue + expert input | 0/192 | 4/192 | 3/192 |

| Model | Single +64 | Correction +64 | Text NLL |
| --- | --- | --- | --- |
| Retained parent | 100.0% | 98.8% | 3.132340 |
| Previous hippocampus | 97.7% | 88.3% | 3.009036 |
| Event cue + expert input | 95.3% | 90.6% | 3.028510 |

The intact endpoint solves 0/4/3 pairs at +0/+32/+64. Blocking readbacks gives 0/2/3; restoring the round clock gives 0/4/3; restoring ingress delivery gives 0/2/3. Each count is out of 192.

After this first phase, query-cue cosine remains 0.999952 (previously 0.999985). The controller allocates 83.8% to refine but only 1.8% to accept. This observation motivated the separate CA1 repair.

## Controller signal audit

| At +32 bytes | Legacy continuation | Bounded continuation |
| --- | --- | --- |
| Effective acceptance | 1.55% | 9.89% |
| Returned residual, RMS | 0.003354 | 0.027199 |
| Query cue cosine | 0.999791 | 0.999947 |
| Answers changed by blocking | 1/64 | 0/64 |

The diagnostic bank is seed 881/start 4000, two templates and 64 answers, separate from final tests. Effective acceptance includes accepted rereads; raw terminal and three-way softmax probabilities are not directly comparable.

## Reproducibility and delivery

| Locked validation gate | Legacy continuation | Bounded continuation |
| --- | --- | --- |
| Paired recall | Fail | Fail |
| Single-fact retention | Pass | Pass |
| Corrections | Fail | Pass |
| Language NLL | Pass | Pass |

Promotion still requires at least 50% both-correct pairs at delays 0 and 32, no more than a two-point single-fact drop, no more than a three-point correction drop, and no more than +0.05 language NLL versus the original strong parent. Validation is seed 683/start 3000 with eight templates. Fixed final endpoints are primary; validation-selected checkpoints are also preserved.

Validation selects update 16 for legacy CA1 and update 0 for bounded CA1. Update 0 is the unchanged starting weights under the new policy; the result tables above use update 16 in both arms.

The hippocampus models retain 391,174 unique parameters. The source schedules match within each comparison, including every mode, delay, document index and source count. The continuation starts at the next unused document and restarts the same mode/gap cycle in both arms. There is no replay, task parser or auxiliary answer target.

| Budget per arm | Answer targets | Task bytes | Language bytes |
| --- | --- | --- | --- |
| 64-update path arm | 1,536 | 46,976 | 8,192 |
| Each 16-update CA1 arm | 384 | 11,904 | 2,048 |

Sparse activation recomputation preserves the complete live credit window; state detaches only at optimizer updates. The new 64-update run uses 16-byte segments above 384 batch-byte positions for updates 1-16 and above 768 thereafter. The original comparison has its recorded adaptive recovery schedule. Both CA1 arms use the same 512-position threshold. Different recomputation schedules can change floating-point accumulation order.

42 targeted tests pass: 21 existing hippocampus, five cue/reinstatement, five CA1 and 11 runtime/checkpoint checks. They cover warm-start parity, inactive rows, expert influence, accepted and rejected rereads, causal credit, checkpoint migration and recomputed logits/gradients. Legacy mode also reproduces public revision 59244b3 logits exactly on 16 trained-checkpoint histories and 32 question branches.

Recorded training-update time is 33.7 minutes for the path arm, 8.3 for legacy CA1 and 8.2 for bounded CA1. Validation, diagnostics and final evaluation are additional.

The source and checkpoint archives include the original parent, previous hippocampus, new path endpoint and both CA1 continuations; selected weights, full predictions, optimizer/RNG/stream state, exact corpus and SHA-256 checksums are preserved. Reproduction commands are in `docs/hippocampus_binding.md`; the current handoff is `STATUS.md`. Exact-resume source guards remain enforced: use the recorded revision to resume an old protocol, or start an explicit continuation under the new controller.
