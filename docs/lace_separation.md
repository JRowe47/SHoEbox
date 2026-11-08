# Query-conditioned lace separation

This is a new-protocol continuation of the fixed bounded-CA1 endpoint from public
revision `f762518a30c7cc01cc308a4a362a3be73d62b52e`. The retained working model stays
`artifacts/hippocampus/parent.pt` until the existing promotion gates pass.

## Forward path

`GraphConfig.lace_separation` defaults to uninstalled/disabled. Installing a head
uses a private generator and adds 448 parameters per recurrent node, 3,136 total.
Every parameter remains trainable; a bypassed head simply has no gradient.

Each node concatenates the exact local context passed to its lace controller
(`x[:,0]`, after its recurrent adapter and any expert-input reinstatement) with
that node region's dedicated `state.hippocampus.cues[:,region]`. The latter is
the live input-driven trace after the current external byte, reused across both
native rounds. It is not an owner capsule, a publication candidate, an answer ID,
a question ID, or an evaluator annotation. Upstream cues receive ordinary task
learning. Extra credit holds these cached inputs fixed.

`h=U*z; bias=0.5*tanh(V*h/0.5)` has rank four, private deterministic 0.02-normal U,
and exactly zero V. The bias is added to each fresh selection score before top-k
in both reference and optimized controllers. It does not accumulate through
refinement. Existing shared utility biases remain separate. Byte ingress, HAT
symbol formation, expert counts and stored capsules are unchanged.

## Local credit

Capture is enabled only for paired training, at the last question byte, for both
native rounds. Keys include node and round; tensor axes preserve layer and lace
identity. Active row indices preserve pair alignment. Only the final refinement's
selected scores get hooks. The load-balancing branch consumes the full external
scores separately and does not contribute directly to those selected-score hooks.

After the ordinary backward, detached gradients are normalized by a per-decision
pair RMS, epsilon `torch.finfo(float32).eps`, and clamped to [-2,2]. Helpful unique
routes get alpha=0.10 additional negative credit. Shared routes with opposing
normalized gradients (product < -0.01) get beta=0.05 positive credit only on the
harmful side. Same-target pairs, missing/inactive counterparts, zero/nonfinite
scales and pairs with both correct-over-contender margins >=1 receive no extra
credit. No second model forward or backward is used by the credit rule.

Analytic U/V additions use cached forward values and the tanh derivative. Additions
sum by parameter identity, then their combined norm is capped at 25% of the
ordinary head-gradient norm before the unchanged global clipping and AdamW step.
Only unscaled FP32 is supported by this extra-credit path. Recording skips no-grad
checkpoint forwards and occurs once during recomputation. Records clear after
credit; nothing is retained across updates or inference events.

## Locked experiment

All four arms install identical heads and inherit the same weights, AdamW state,
RNG, carried language state, source cursor and next document from
`artifacts/hippocampus_binding/ca1/bounded/last.pt` (update 16).

| Arm | Forward bias | Alpha | Beta |
| --- | --- | ---: | ---: |
| A | Bypassed | 0 | 0 |
| B | Task-trained | 0 | 0 |
| C | Task-trained | 0.10 | 0 |
| D | Task-trained | 0.10 | 0.05 |

Every arm runs 32 updates: paired/paired/single/correction, gaps 0/4/8/16/32,
LR .003, AdamW (.9,.95), weight decay .01, language .25, syntax .1, global clip 1.
The inherited data seed is 683. The next document is 800 and the endpoint is 856.
Every update is saved; immutable snapshots are retained at 8/16/24/32.
All arms use 16-byte sparse recomputation segments above 512 batch-byte positions.
The checkpoint carries the entire optimizer, RNG, language stream and state.
Exact resumes reject changes to source hashes, protocol, corpus or configuration.

The primary endpoint is fixed update 32. Initial and final validation are retained;
no intermediate weight selection is performed. Final tests cannot run until all
four endpoints and matched ledgers pass the completion audit.

## Cohorts

`artifacts/lace_separation/cohorts.json` contains rendered inputs, targets and
SHA-256 case hashes, enumerated before training. Diagnostic seed/start is
2309/180000, validation 2411/200000, final 2539/240000. Validation uses eight
paired templates (128 histories, 96 distinct-answer histories) per delay; final
uses sixteen (256 histories, 192 distinct-answer histories). Full 4x4 value sets
include equal-answer controls, and every base entity ordering has a reversal.

The compact native grammar has a finite set of immediate strings. Fresh seed and
index combinations repeat old byte sequences. Each new paired evaluation history
therefore starts with 24 natural-text bytes from its held-out split, before both
writes. Added retention delays remain 0/32/64. The retained parent is evaluated
on these exact same inputs. Case hashes are checked against reconstructed prior
paired banks and training document identities and against all new cohorts.
This controls exact input reuse, not the reuse of semantic fact combinations.

A separate, predeclared unseen-wording bank changes `A x?` to `Value of A x?`.
Its results are reported separately from native primary results. Neither wording
nor any labels are installed as an inference parser.

Established retention/language cohorts stay fixed: validation seed 683/start 3000,
eight single templates (32 answers) and eight correction templates (128 answers)
at +0/+32, plus 256 natural-byte targets. Final seed 1699/start 160000 uses sixteen
single templates (64 answers) and eight correction templates (128 answers) at +64,
plus 1,024 natural-byte targets. Predictions are preserved.

Promotion requires >=50% both-correct native validation pairs at +0 and +32,
<=2 percentage points single-fact loss, <=3 correction points lost, and <=+0.05
language NLL versus the retained strong parent on matching cohorts.

D's final same-weight tests bypass the head, swap paired question cues only at
the head input while retaining their valid shared history, or block hippocampal
readbacks. C versus D isolates training-time inhibition. Lower overlap alone is
not success; fixed top-k means functional redistribution does not save experts.

## Reproduction

Restore the source and checkpoint archives into one directory, install
`requirements.txt`, and run:

```bash
python -m unittest experiments.text_regions.test_lace_separation experiments.text_regions.test_hippocampus experiments.text_regions.test_hippocampus_binding experiments.text_regions.test_hippocampus_ca1 experiments.text_regions.test_content_checkpoint experiments.text_regions.test_runtime_fusion -q
python scripts/run_lace_separation.py --arm D --out artifacts/reproduction/D --initial artifacts/lace_separation/initial.json
python scripts/evaluate_lace_separation.py --checkpoint artifacts/reproduction/D/last.pt --cohort validation --part pairs --out artifacts/reproduction/D_validation_pairs.json
```

Use `--resume` on the identical arm/output/protocol after an interruption. New
runs require empty output directories. Recorded artifacts include the locked
cohorts, source hashes, preflight, per-update budgets/targets/credit measurements,
full predictions, retention/text metrics, timing and checksums.
