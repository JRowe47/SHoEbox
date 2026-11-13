# SHoEbox: reference acceptance contract

13 November 2026 | Sixth recommended fix | Base: 19f2fc1

The sixth fix makes acceptance enforce the implemented reference contract. One command now checks live credit, temporal composition, reusable recognition, paired memory, frozen downstream effects, generation reproducibility, checkpoint continuation, the native regressions and sanitizers. The change adds observations and tests; it does not alter learning, learned parameters or checkpoint v39 / composition ABI 10.

## Credit and failure detection

Missing neural parents, feedback producers, input dependencies, content producers and semantic handles fail acceptance in every region. Overwritten credit, omitted semantic work, capacity pruning, expired forecasts and duplicate schedules also fail. Local prediction and language/DFA deliveries are reported separately, together with eligible expert-update calls and content updates. The native causal tests check actual weight changes and derivatives. All reference features must be enabled; an inactive path cannot pass merely because its error counters are zero.

## Primary clocks and local sequences

Each publication records primary and all-source coverage spans separately, including sums, maxima and logarithmic histograms. Hierarchy checks follow actual repeated primary-source roles, including E4 to G5. Temporal length counts completed frames in L2-G5, excluding relays; binding three source roles is still one frame. Recognition of newly formed units is separated from inherited relay recognition. Partial EC/Y startup fusions without a designated primary observation are counted explicitly as unclocked; they do not inflate primary-span means.

## Matched contribution and generation checks

Each intervention uses a fresh frozen copy of the same fitted model and identical bytes. Channel controls preserve exact event identity, provenance and primary-clock hashes. Class removal may change completion. Both positive and negative prediction effects are retained. Cold/warm frozen memory is explicit, and zero held-out recall is not treated as a successful recall intervention. Two held-out prompts use the same seed and sampling settings in production and diagnostic kernels; outputs must match and fitted state must remain unchanged.

## Lifecycle and reproducibility

The new observations are excluded from model fingerprints and serialized state. Reset, composition import and snapshot load begin a new observation window; existing diagnostic counters keep their established scopes. The default acceptance command runs the full regression suite and ASan/UBSan and writes explicit failure status on error. The mechanisms-only option is labelled partial and cannot report complete acceptance. The existing reference smoke shares the same credit and hierarchy contracts.

## Verification and cost

224 package tests passed, including 14 new acceptance tests. All 14 also passed at 64 experts. ASan/UBSan passed, including native observation exports, lifecycle handling and a live semantic DAG above 16,384 records. Actual baseline-v39 continuation, current chunked continuation, diagnostic logits and generated outputs were exact. The execution session ended before the regression summary: 197 completed passing cases were retained, and the remaining 27 cases passed in a separate run. The combined test-ID ledger covers the complete suite.

Four alternating measurements restored the same actual fifth-fix v39 checkpoint before each 512-target continuation, on one pinned CPU. Median throughput was 139.70 -> 146.20 bytes/s (+4.65%). Compilation, initialization and loading were excluded. All measured continuations produced the same logits and final fingerprint. The final timing ran after the other checks finished; an earlier concurrent timing, including its long outlier, is retained separately.

## Hierarchy measurements

| Region | Publications | Primary mean | Coverage mean | Multi-frame | Recognized |
| --- | --- | --- | --- | --- | --- |
| L2 | 2576 | 1.59 | 1.59 | 952 | 912 |
| L3 | 493 | 8.31 | 8.31 | 172 | 243 |
| E4 | 220 | 18.59 | 81.95 | 0 | 3 |
| F4 | 129 | 31.75 | 34.99 | 0 | 4 |
| G5 | 110 | 37.11 | 225.39 | 0 | 0 |

Means are bytes per publication. Multi-frame counts local temporal completion; recognized counts newly formed publications. The short 4,096-target probe has no recognized G5 units and no local G5 multi-frame completions. Its longer primary span comes from lower-level constituents. The acceptance output keeps those facts distinct instead of equating broad coverage with learned upper sequences.

## Frozen downstream contributions

| Removed contribution | Delta NLL | Logit RMS | Applications |
| --- | --- | --- | --- |
| L3 content | -0.00006371 | 0.026398 | 358 |
| F4 content | +0.00041917 | 0.019871 | 97 |
| G5 content | +0.00000208 | 0.000868 | 91 |
| G5 feedback | +0.00011341 | 0.002043 | 941 |
| Pattern classes | +0.02315859 | 0.228722 | 5838 |
| Paired recall (familiar) | +0.00004118 | 0.000325 | 60 |

Held-out baseline NLL is 2.862002 nats/byte. Delta is removal minus baseline: positive means the intact contribution helped prediction. The held-out memory probe had 0 accepted recalls; the separate familiar probe exercises paired returns. These are measurements from a small synthetic mechanism run, not a claim that each active region improves prediction.

## Held-out prompt generation

Seed 17; temperature 0.8; 64 generated bytes; printable ASCII plus newline. Both outputs reproduce exactly in the diagnostic kernel.

Prompt: The traveler followed 

```text
uboe rede ir ztain he sivge iugw` tle the hr ie the the Tgbe hee
```

Prompt: Beyond the garden 

```text
te mhe tlPe c|s rn he rovre quew ntihe! th vThe rorer th h h hee
```

## Reproduce

```bash
python experiments/identity_semantics/src/run_reference_acceptance.py
SHOEBOX_EXPERTS=64 python -m unittest discover \
  -s experiments/identity_semantics/tests \
  -p test_reference_acceptance.py -v
python experiments/identity_semantics/src/compare_acceptance_runtime.py \
  --baseline-runtime /path/to/base/experiments/identity_semantics/runtime
```

Evidence: results/reference_acceptance.json, reference_acceptance_tests.txt, reference_acceptance_recovery_tests.txt, reference_acceptance_recovery.json, reference_acceptance_sanitizer.txt, acceptance_runtime_comparison.json, acceptance_runtime_comparison_concurrent.json, acceptance_smoke.json and reference_acceptance_checks.json. No trained models, datasets or compiled binaries are included.
