# SHoEbox: constituent experts

13 November 2026 | Third recommended fix | Base: 5d20d40

Recommendation 3 is implemented: each encoded constituent now selects a regional expert using only its own structural descriptor and learned child content. The selected expert transforms that content before export. It shares the existing W/bias/keys banks, adds no learned parameters, and is enabled in fresh reference models. The 12-expert configuration remains 1,031,952 parameters.

## Constituent-only forward pass

Role-normalized learned child residuals are passed through tanh and the bounded payload gain. The content input is structure plus that payload. A masked key comparison selects the expert and runner; their margin controls the response gate. The exported residual combines the regional structural adapter, direct child payload, expert bias and gated expert matrix response. This runs at each actual encoding, including bound-frame admission before recurrent execution.

## Identity and context separation

Content selection and transformation read no recurrent h/u, feedback, usage counts or running means. At fixed weights and constituent inputs, unrelated retained history cannot change the code. Exact IDs, structural descriptors, role provenance, completion clocks and publication decisions keep their existing rules. Byte atoms and transparent relays keep their existing behavior. Expert choice does not introduce reusable pattern-class identities.

## Saved causal credit

Each semantic record owns the content expert, runner, input, mask, response and gate derivatives used at encoding. Reverse credit reaches that selected bank and recursively reaches child content, including repeated roles. Adapter and payload-gate derivatives include the new path. Shared-bank content updates are accumulated until all derivatives in the enclosing semantic/local pass have been computed. Task/local scales, semantic learning rate and frozen-node transmission remain in force.

## Checkpoint and lifecycle

Version 37 / composition ABI 10 saves content policy and generation-checked content eligibility. The sidecar follows semantic ownership, collection and pool growth. Versions 35/36 import without dropping live events and retain their prior content policy; version 35 also retains its prior local-credit policy. Explicitly call configure_content_experts(True) to upgrade an older loaded model. Previously encoded constituents retain their original credit semantics. Missing or mismatched new eligibility is rejected.

## Verification

All 184 package regression tests passed (104 before the runner stopped, followed by 80 resumed checks). All 9 new content-expert tests also passed at 64 experts. ASan/UBSan passed training, evaluation, drain, reset, export/import and a live semantic DAG larger than 16,384 entries. The actual version-36 baseline reproduced continuation logits, parameters and saved state bit-for-bit after import.

Controlled interventions verify selected-expert effects on child code and parent input, unchanged structural identity, unrelated-expert isolation, and independence from retained history. Finite differences check own and two-hop weight/bias derivatives and routing-margin gradients, with fixed hard routing. Tests also cover shared-bank update ordering, repeated roles, frozen transmission, task/local scales, relays, 192 source replacements, malformed sidecars and exact checkpoint/chunk continuation.

This extends online saved-eligibility credit: the saved gate/input derivative is combined with the current pre-update expert matrix. The content path uses its own input; recurrent context continues through the existing context expert path.

## Matched smoke results

Seed 7; one pinned CPU; 2,048 warmup targets, then 2 rounds of 2,048 targets at LR 0.04 and local LR 0.04/3. Both policies enable fix 2 and differ only in content-expert encoding. Compilation, draining and evaluation are outside training timing. The repeated-sentence training smoke and separate synthetic held-out prose measure this mechanism and its cost.

| Measure | Previous policy | Content experts |
| --- | --- | --- |
| Median wall-clock bytes/s | 121.7 | 39.4 |
| Median bytes/CPU-second | 121.8 | 39.8 |
| Final round training NLL | 2.072541 | 2.071205 |
| Synthetic held-out NLL | 2.844947 | 2.847694 |

Median wall-clock throughput changed by -67.7%; CPU-time throughput changed by -67.3%. The new run encoded 27,919 constituents and applied 28,352,950 local and 11,733,299 task content-update calls. All missing content/local/neural/feedback producer counters, overwritten credit, expired semantic handles and expired forecasts remained zero. Both policies preserved frozen evaluation/generation weights and exact chunked continuation. Individual round timings are recorded in the raw JSON.

## Frozen contribution check

| Removed content transform | Delta NLL |
| --- | --- |
| L3 | -0.00005886 |
| F4 | -0.00000147 |
| G5 | +0.00002830 |

The content_expert control removes only the chosen region's new constituent transform. Positive delta NLL means removal worsened prediction. The existing expert_transform control still targets recurrent context. Production and no-intervention diagnostic logits matched exactly.

## Generation check

Printable ASCII plus newline; temperature 0.8; seed 17; prompt "The river "; 96 bytes per policy.

Previous policy:

```text
tgh he th he wr th he throe rs wr the the. uEd he the rl#e i!e srde h torrosde The se hhe theie
```

Content experts:

```text
tgh he th he wr th he throe rs wr the the. vOe ired the Thde heee thd nshee le The th hre uoruha
```

## Reproduce

```bash
python -m unittest discover -s experiments/identity_semantics/tests -v
SHOEBOX_EXPERTS=64 python -m unittest discover -s experiments/identity_semantics/tests -p test_content_experts.py -v
python experiments/identity_semantics/tests/run_completion_sanitizer.py
python experiments/identity_semantics/src/validate_content_experts.py
python experiments/identity_semantics/src/report_content_experts.py
```

Raw results: content_expert_validation.json, content_expert_checks.json and content_expert_legacy36.json. No trained models, corpus files or compiled binaries are included.
