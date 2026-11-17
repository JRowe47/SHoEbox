# A6: admission improved; identity and generalization remain

A6 fails the unchanged development gate. E8 stays public; A1–A6 audits stay
closed. The comparison below uses newly fitted matched A5/A6 on the A6 folds,
not the older A5 campaign's different worlds. Full autoregressive episode
exactness is 87.890625% competing, 94.140625% corrected overwrite, 95.312500%
clean and 84.375000% diverse256. Isolated speed is 67.583762% native.

## Long-diverse admission and signature survival

The registered frozen first-32-world ×4-order trace contains 256 required
target occurrences per seed. Both answers are teacher-fed for diagnosis.

| Measure | Matched A5 | A6 |
|---|---:|---:|
| Diverse256 original-signature survival | 93.75% | 100% |
| Target-removal events, all three seeds | 48 | 0 |
| Incoming fully diverse pairs recognized, per seed | 1,268 / 7,968 | 248 / 7,968 |
| Incoming maximum-over-four-phase false recognition | 15.913655% | 3.112450% |
| Individual phase false positives, per seed | 1,408 / 31,872 | 248 / 31,872 |
| True record-phase false negatives, per seed | 0 / 256 | 0 / 256 |

Every A5 removal occurs while all slots are recognized; none is caused by a
true-target presence rejection. A6 eliminates those observed removals with the
same 16 slots and admission policy. External reconstruction verifies 10,392
admission decisions per seed/arm. Shorter 16/64-diverse traces retain all target
signatures in both arms. These are separate measurements from readout success.

## Surviving targets can still lose at readout

| Diverse256 teacher-fed measure | Matched A5 | A6 |
|---|---:|---:|
| Accuracy conditional on signature survival | 85.0000% | 96.8750% |
| Surviving target recognized | 100% | 100% |
| Surviving target shortlisted | 91.6667% | 100% |
| Recognized filler phases per answer | 11.15625 | 1.93750 |
| Filler mass on wrong answers with surviving target | about 97.78% | about 49.99% |

A6 reduces false filler recognition and selection pressure, but remaining
filler can still compete with a surviving, recognized, shortlisted target.
All selected pairs/weights match an independent exhaustive selector.
Admission failure, presence rejection, shortlist exclusion and final decoding
failure are recorded separately in the raw traces.

The diagnostic subset is easier than the entire development endpoint:

| A6 autoregressive whole-episode exactness | First 32 worlds | Other 96 worlds | All 128 |
|---|---:|---:|---:|
| Clean | 97.65625% | 94.53125% | 95.31250% |
| Competing | 95.31250% | 85.41667% | 87.890625% |
| Corrected overwrite | 96.87500% | 93.22917% | 94.140625% |
| Diverse256 | 93.75000% | 81.25000% | 84.37500% |

Thus the 96.875% teacher-fed per-answer result does not contradict 84.375%
full-population autoregressive episode exactness. Neither 100% signature
survival nor the readout decomposition is established outside the traced worlds.
The data-only attribution was registered before execution in
`DIAGNOSTIC_ATTRIBUTION_ADDENDUM.md`; it uses saved outcomes without new inference.

## Residual identity/version geometry

Competing nearest-query key identity is identical at 96.875% for matched A5/A6.
Targets are present/recognized in every traced answer, but only 97.65625%
are shortlisted and answered correctly. Approximately 4.2% of selected mass
belongs to another annotated key. Raising presence cost does not repair the
key geometry.

For recognized competing records, reconstruction squared error averages
3.28851452, compared with 0.15897195 for queries. Wrong nearest identities have
fixed-code Hamming distances 4–6 (mean five). The full 256-code book has no
collisions; these errors arise in learned reconstructions, not duplicate codes.

On corrected overwrite, all required targets survive and are shortlisted.
Teacher-fed exactness is 98.4375%; changed-key answers are 100%, with zero
old-value predictions in this subset. Nearest-query selection alone does not
reliably identify the latest occurrence: same-key old/new squared distances
span approximately 0.324–10.223, overlapping different-key geometry. Chronology
helps this subset, but its zero old-value count cannot establish complete
version resolution while the full overwrite gate fails.

## One next registered revision

Exact traversal of one 2M continuation, shared across seeds and arm schedules,
finds 28,856 positive record-key events. Of these, 26,255 (90.9863%) have padding
in the raw eight-byte window; there are 5,997 distinct key/phase/window tuples.
The 9,963 fully diverse negative-window events cover 3,252 unique raw windows.
The coverage table and per-key/phase counts are retained in `coverage.json`.

`A7_REGISTRATION.md` fixes one record-context curriculum change: replace only
the existing 0–7-byte continuation spacers after the two original records with
deterministic per-occurrence bytes from the existing diverse alphabet. Keep
lengths, labels, masks, budget, all A6 model arithmetic and every gate. The
controls retain PAD spacers. A7 is not implemented or fitted. No model state,
cache slots, codebook or matching coefficients are changed.

The speed gate remains independently blocking. Exact arithmetic reuse is
permitted only with pre-fit bit-identical per-row evidence and disabled FMA;
no speed recovery is assumed. The current public-E8 timing control also falls
below 80% despite identical prediction checksums, and is preserved without
retesting. All original qualified E8 measurements and sources remain intact.

## Evidence and scope

The archive retains raw `conflicts.json`, `retention.json`, `survivors.json`,
`geometry.json`, `matching_geometry.json`, `identity_attribution.json`,
`coverage.json`, complete outcomes, source/result locks and process receipts.
Compact summaries are in `evidence/`. No diagnostic changed a selected fit,
opened an audit, evaluated an A7 example or tuned a coefficient. The report's
accounting distinguishes fitting, discarded correctness trajectories,
inference, local loss/gradient events, comparisons and whole-process CPU.
