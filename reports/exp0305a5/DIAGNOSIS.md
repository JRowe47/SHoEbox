# A5 frozen diagnosis and next focused revision

All numbers below are frozen development diagnostics, not replacement autoregressive gates. No audit, alternate policy or additional fit was evaluated.

## Identity, version and mixture

| Matched arm | Competing nearest identity | Competing other-key mass | Changed-key old-value predictions |
|---|---:|---:|---:|
| A4 | 97.6562% | 4.9035% | 4.6875% |
| A5 | 97.6562% | 3.5349% | 6.2500% |

A5 same-key old/new squared distance averages 2.838467; its observed range overlaps different-key distances. Nearest overwrite identity is 99.21875%, but nearest version is only 82.03125%. Chronology remains necessary, and residual old-value choices remain.

Query reconstruction squared error averages 0.166137, versus 3.592463 for recognized record keys. Wrong nearest identities differ by 5–8 fixed-code bits; these errors are not code collisions or one-bit-neighbor failures. No replacement codebook was evaluated.

## Long-diverse admission and survivors

| Arm | Original signature survives | Correct given target survives | Surviving target shortlisted | Unannotated recognized phases per answer | Filler mass on wrong surviving-target answers |
|---|---:|---:|---:|---:|---:|
| A4 | 95.3125% | 81.9672% | 86.0656% | 11.5625 | 93.7949% |
| A5 | 95.3125% | 80.6011% | 86.0656% | 11.5625 | 95.0343% |

All surviving targets are recognized. All 36 A5 removals happen during filler under all-slots-recognized pressure; none is an oldest-unrecognized removal. Fully diverse incoming windows are falsely recognized in 1336/7968 cases per seed (16.7671%). Missing signatures account for only part of the long-diverse failure. False recognized filler also displaces surviving targets during selection.

## A6 choice

The next single revision increases only the negative-example cost for the four existing record-presence classifiers from 1 to the existing positive cost 4. It targets both admission pressure and false readout candidates, with no new state, weights, threshold or inference work. The query classifier, A5 key objective, paired values, A4 chronology and all other updates stay fixed. It does not claim to repair the separate residual record-key/chronology errors. Every original gate, including competing bindings and >=80% native speed, remains mandatory. Read `A6_REGISTRATION.md` in full; A6 is not fitted.

## Scope and integrity

Main geometry/occurrence/admission traces use the first 32 development worlds x4 order siblings, seeds 1/7/19. Survivor attribution was explicitly registered after the first trace and then run on the same subset with A4/A5, no intervention. Checkpoint/source hashes and separate input/padding/process-cost receipts are retained. Identity attribution performs no inference.
