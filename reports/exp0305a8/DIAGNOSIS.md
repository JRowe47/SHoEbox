# A8 diagnosis: balanced exposure retains the presence/identity tradeoff

A8 fails development. Qualified public E8 remains unchanged, and A1–A8 audits
remain closed. The registered E8/A6/A8 campaign completed exactly 20.25M fitting
bytes. Full outcomes, all 18 model/cursor pairs and diagnostic inputs were locked
before diagnosis. No extra fit, sweep or generic/capacity qualification followed.

## Full autoregressive outcomes

Whole-episode exactness on 128 fresh worlds ×4 record/query orders, averaged over
seeds 1/7/19. These outcomes determine the decision. Fresh A6 controls are kept
separate from A6's original campaign and E8's original qualification.

| Endpoint | Matched A6 | A8 |
|---|---:|---:|
| Clean binding | 92.968750% | 87.890625% |
| Competing bindings | 87.109375% | 80.859375% |
| Corrected three-value overwrite | 92.187500% | 78.515625% |
| Diverse256 | 81.250000% | 83.593750% |
| Isolated throughput, relative to native | 67.443791% | 68.587044% |

The fixed balanced curriculum modestly improves diverse256 but lowers clean,
competing and overwrite outcomes. The 95% primary, 90% delay and 80% speed
mean gates remain, together with every seed/position/offset/lesion/binary gate.
Neither matched A6 nor A8 passes. The full per-seed decisions are in
`verification.json`; the original A1–A7 outcomes and costs are untouched.

## Surviving signatures and admission

The registered diagnosis uses the first 32 development worlds ×4 orders,
both teacher-fed answers, all seeds. These are subset per-answer localization
measurements, not whole-population autoregressive episode scores.

| Diverse256 diagnostic endpoint | Matched A6 | A8 |
|---|---:|---:|
| Target signature survival | 100% | 100% |
| Target removals, all seeds | 0 | 0 |
| Incoming fully diverse pairs falsely recognized | 3.212851% | 1.857430% |
| True-record phase false negatives | 0% | 0% |
| Answer accuracy given surviving signature | 87.500000% | 89.843750% |
| Surviving target shortlisted | 99.218750% | 99.218750% |
| Recognized filler phases per answer | 2.031250 | 1.187500 |
| Filler mass on wrong survivor answers | 28.663252% | 38.276743% |

All 31,176 admission decisions per arm are reconstructed independently. A8
reduces false recognition and the number of recognized filler phases. Selected
filler mass on the remaining wrong answers rises; fewer false candidates does
not imply that their conditional influence falls. Signature capacity is not
the observed cause of these errors: every required signature survives here.

## Record identity remains the bottleneck

| Competing/overwrite subset endpoint | Matched A6 | A8 |
|---|---:|---:|
| Competing nearest-query key identity | 92.968750% | 88.802083% |
| Competing selected mass on other keys | 10.292267% | 16.044485% |
| Changed-key old-value predictions | 3.125000% | 7.812500% |
| Competing required target shortlisted | 95.572917% | 92.187500% |

Every required competing/overwrite signature is present and recognized.
For A8, the competing trace's mean record-key reconstruction squared error is
5.170884, while query reconstruction error is 0.220569. Wrong nearest keys
span Hamming distances 3–8; the full 256-code book has no collisions. This
supports focusing on record-key learning rather than adding memory slots or
another chronology threshold. It does not identify a code collision.

Keep the subset/population distinction explicit:

| A8 autoregressive episode exactness | First 32 worlds | Remaining 96 | All 128 |
|---|---:|---:|---:|
| Clean binding | 80.468750% | 90.364583% | 87.890625% |
| Competing bindings | 76.562500% | 82.291667% | 80.859375% |
| Corrected overwrite | 73.437500% | 80.208333% | 78.515625% |
| Diverse256 | 81.250000% | 84.375000% | 83.593750% |

## The registered balance occurred exactly

Data-only traversal finds 28,810 positive record-key events per continuation
in both A6 and A8. A8 reduces PAD-containing positive windows from 26,213 to
18,784 and increases distinct key/phase/windows from 6,022 to 15,058. Both
regimes cover all 64 phases without extra RNG calls or terminal balancing.

| A8 block | Encountered episodes | Consumed input bytes | Positive key events | PAD-containing positives | Distinct key/phase/windows |
|---|---:|---:|---:|---:|---:|
| PAD | 5,568 | 1,003,872 | 14,482 | 13,190 | 4,059 |
| Diverse | 5,520 | 996,128 | 14,328 | 5,594 | 11,865 |

Distinct-window counts overlap between blocks; their union is 15,058. Exactly
38,797 original PAD input bytes are replaced per A8 continuation, 116,391
across seeds. The incomplete final episode and all its consumed bytes remain
as encountered. All lengths, phases, labels, masks and both RNG states match
controls. The reduced clean/identity scores are not a missing PAD-block bug.

## One focused next registration

`A9_REGISTRATION.md` retains A8's input curriculum and allows positive
record-key hybrid updates only in PAD blocks. Presence continues learning
from PAD and diverse blocks; query-key, value, applicability and native rules
stay unchanged. This directly tests whether the useful presence exposure can
be retained without updating record-key projections on diverse-block examples.

The mask is explicit training-only credit, derived from the existing episode
index. It never enters inference, parses a record, adds model state, changes
labels, repeats an example or compensates omitted gradients. A9 is registered
only, not implemented or fitted. Matched E8/A8/A9 requires fresh worlds and
independent warmups. The throughput gate remains independently binding;
bit-identical reuse requires pre-fit scalar-reference receipts and disabled FMA.

## Preserved verification history

One initial pre-fit Python assertion required crossing episode 128, but its
9,000-byte trajectory stopped at episode 115. Model/cursor equality had passed.
The correctness trajectory alone was extended to 14,000 bytes and passed before
research fitting. Both sources, invocation logs, direct event counts and CPU
receipts remain. They consume 47,662 +57,662 discarded correctness bytes;
the 18 final cursor comparisons separately consume 36,000 discarded bytes.

An early source-copy command had a Python parse error before executing any
model or file edit. Its setup note is retained; setup/editor CPU is not metered.
The inherited survivor tracer's narrow checkpoint glob covers A6 only. An
additional complete 18-checkpoint diagnostic lock was written before any trace
and verified afterward; the pre-fit source closure was kept unchanged. All
fitting/evaluation commands and the final delivery verifier pass. No research
fit was repeated and no measurement was discarded.
