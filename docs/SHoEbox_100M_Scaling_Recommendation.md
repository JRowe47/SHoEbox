# SHoEbox: capacity diagnosis and a 100M design

November 10, 2026

**Recommendation: move to wider regional computation and communication. Build a 256-wide, 18-region, approximately 100M-parameter SHoEbox, with one expert selected per regional event. First validate width scaling on a matched approximately 20M comparison.**

The existing models have reached a practical plateau under their tested continuation policies. The evidence supports a combination of limited per-event representation and learning dynamics. It does not establish that every stored parameter is fully used. The 5M model still substantially outperforms the 1M model, so scaling has already helped.

This recommendation is grounded in the complete 175M package, its runtime and checkpoints, the recorded loss history, the preceding intervention reports, and an additional read-only diagnostic of both selected checkpoints. No training or checkpoint modification was performed for this review. The 100M configuration below is a design proposal, not an implemented or trained model.

## 1. What the loss curves establish

### The useful comparison is on identical evaluation bytes

| Model | Selected source | Additional fresh targets | Common 4M test: source | Common 4M test: 175M | Change |
|---|---:|---:|---:|---:|---:|
| 1M | 165.5M | 9.5M | 1.841776 | 1.845213 | +0.003437 |
| 5M | 150.25M | 24.75M | 1.760330 | 1.761167 | +0.000837 |

Losses are nats per byte; lower is better. The recorded paired-document 95% intervals for the changes are +0.002538 to +0.004420 for 1M and +0.000571 to +0.001099 for 5M. Both endpoints also worsened on the separate previous 4M test. New documents alone did not restart sustained improvement.

The selected 5M model improves common-test NLL by **0.081446 nats/byte**, or **7.82% lower byte perplexity**, versus the selected 1M model. This is real benefit from additional experts, not identical behavior across sizes.

The large drop after 126M is partly optimization progress revealed by cooling. On the unchanged historical validation, 5M improves from 1.804318 at 126M to 1.760409 at 136.25M. Later interventions principally suppress drift and extract much smaller gains. The late flat region therefore represents a settled continuation regime, not a clean experiment measuring capacity alone.

Two plotting details matter. The 5M validation changes from 131,072 bytes to a different 2M-byte split at 136.25M; its endpoint cannot be compared directly with the blue endpoint. Also, the bold trajectory joins retained checkpoint lineages after rollbacks. Its x-axis excludes discarded branch work. There is no single fixed-policy, fixed-split 0-175M experiment from which to fit a reliable 100M scaling exponent.

### Why the plateau is partly a learning-policy ceiling

The 5M final continuation used main LR **4e-6**, local LR **4e-6/3**, A1 decoder multiplier **0.1**, and effective A1 prototype EMA multiplier **0.001**. That is a consolidation regime. During the additional 24.75M bytes, regional expert matrices changed by only about **0.0014%-0.132%** in relative norm. Fresh data was tested with very limited plasticity; this does not establish that the data mixture can never help at a different training stage.

My judgment is to stop spending long runs on these same settled checkpoints. Keep the 165.5M 1M checkpoint and 150.25M 5M checkpoint as references, and spend the next compute budget on a wider model trained with a coherent schedule.

## 2. What the current model actually scales

The recovered CPU runtime uses a residual affine expert followed by tanh:

`h = tanh(u + gate * W_expert * u + bias + optional_EC_recall)`

The 1M model has 12 experts per region; 5M has 64. **Both use width 64**, one selected expert per regional event, 64-dimensional inter-region messages, and a decoder reading the concatenation of A1 and centered Y: 128 features total. Thus the earlier size increase expanded the number of available transformations while leaving each selected transformation and each message the same size.

Transport uses distinct fixed signed permutations for source slots, summed into a regional bundle. The historical 1B/2B/4B recipe labels are weighting conventions in this runtime; its actual messages are continuous 64D vectors. Learned bottom-state gates exist in A1/B1. All regions also carry filtered recurrent state. This proposal scales that implemented model rather than assuming the older discrete-port architecture is still in use.

The upper routers inspect 16 of 64 coordinates. Bottom routing uses all 64. Temporal credit uses 16 selected coordinates, publication credit has a 64-feature budget, and credit propagation is bounded to three hops. Merely increasing expert count would leave these shared limits intact.

### Utilization and the additional frozen diagnostic

On the recorded common 2M validation, the selected 5M model uses 47.9 effective experts in A1, 52.1 in B1, and roughly 57-61.5 elsewhere, out of 64. Effective count is the exponential of routing entropy. This rules against widespread expert non-use; it does not prove useful specialization or exhausted capacity.

I also evaluated both best checkpoints on the same first 131,072 targets of that validation stream with diagnostics enabled. Source checkpoint fingerprints remained unchanged. The native observer samples each region every 16 executions.

| Region | 1M output effective rank | 5M output effective rank | 1M saturated outputs | 5M saturated outputs |
|---|---:|---:|---:|---:|
| A1 | 27.58 | 29.24 | 38.76% | 20.83% |
| B1 | 23.19 | 29.74 | 10.79% | 0.52% |
| E4 | 41.56 | 42.56 | 0.00% | 0.01% |
| V4 | 39.06 | 41.77 | 0.00% | 0.01% |
| G5 | 5.30 | 29.10 | 0.02% | 0.00% |
| EC-C | 15.84 | 18.42 | 46.55% | 29.72% |
| Y | 30.70 | 37.78 | 0.01% | 0.02% |

Effective rank is entropy-based covariance rank within a 64D output. Saturation means absolute tanh output greater than 0.95. These are representation diagnostics, not regional language losses. The larger model develops richer representations, particularly G5, but A1 and EC-C still operate with substantial saturation. Maintain controlled per-coordinate activation scale when widening, and inspect their input, recurrent, and recall contributions separately. Width is the strongest next capacity experiment; these measurements do not prove an exclusively width-limited ceiling.

The selected 5M checkpoint's existing frozen probes show +0.01192 nats/byte when changing Y is replaced by its fitted mean and +0.000802 when recall is blocked. The dynamic hierarchy and recall help, but their measured contributions remain modest. I would first improve the width of the hierarchy and its readout, keeping the existing memory-store capacity.

## 3. Recommended 100M configuration

Use a common width of **256** so every interface can carry the wider representation without repeated compression to 64. Keep the topology, source recipes, scheduling, event-triggered EC execution, top-1 selection, residual-tanh experts, binary byte head, and local-plus-lace training architecture for the first scaling run.

| Property | Current 5M | Proposed approximately 100M |
|---|---|---|
| Counted learned/stored parameter scalars | 4,990,720 | 99,727,616 |
| Regions | 18 | Same 18 |
| Regional width | 64 | 256 |
| Selected expert matrix | 64 x 64 | 256 x 256 |
| Experts selected per regional event | 1 | 1 |
| Total experts | 1,152 | 1,488, allocated by region |
| Continuous publication and state width | 64 | 256 |
| A1 + centered Y decoder input | 128 | 512 |
| Upper routing coordinate budget | 16/64 | 64/256 |
| Bottom routing coordinate budget | 64/64 | 256/256 initially |
| Temporal credit coordinate budget | 16 | 64 |
| Publication credit feature budget | 64 | 256 initially |
| Feature-membership mask | One 64-bit word | Four 64-bit words or a generic index list |
| Local predictors and EC recall projection | 64 x 64 | 256 x 256 |
| Learned A1/B1 state gates | 64D | 256D |
| Publication ring / bottom temporal ring | 16 / 64 records | Retain initially; log expiry and overwritten credit |
| Credit depth | 3 | Retain initially; measure useful credit by hop |
| Persistent recall entries | 8,192 per EC channel | Same counts, 256D keys and values |
| Primary learning algorithm | Normalized local updates plus bounded lace credit | Same, with width-calibrated rates and coordinated adaptation |

The feature budgets preserve the current fractional routing/temporal sparsity while widening the full payload. Full publication credit matches the current policy initially. Reduce that budget later only after measuring its quality/throughput tradeoff.

### Regional allocation

| Regions | Count | Experts each: 5M | Experts each: proposal | Proposed expert-matrix parameters |
|---|---:|---:|---:|---:|
| A1, B1 | 2 | 64 | 128 | 16,777,216 |
| L2, S2, R2, F2 | 4 | 64 | 96 | 25,165,824 |
| L3, S3, R3, F3 | 4 | 64 | 80 | 20,971,520 |
| E4, F4, V4, M4 | 4 | 64 | 64 | 16,777,216 |
| G5 | 1 | 64 | 48 | 3,145,728 |
| EC-C, EC-X | 2 | 64 | 64 | 8,388,608 |
| Y | 1 | 64 | 96 | 6,291,456 |
| **Total** | **18** | **1,152 total** | **1,488 total** | **97,517,568** |

This allocation is my engineering starting point, not a measured optimum. It gives the largest banks to frequent lower-level execution and increases Y capacity. Upper regions retain all 18 roles and much larger individual experts. G5 has fewer experts but twelve times its old expert-matrix capacity because width grows fourfold. EC and fourth-order regions each gain sixteen times their expert-matrix capacity. There is no need to add more regions to reach 100M.

### Exact accounting under the current parameter convention

| Component | Proposed parameters |
|---|---:|
| Expert matrices | 97,517,568 |
| Expert biases | 380,928 |
| Router keys | 380,928 |
| Local prediction / EC port matrices | 1,179,648 |
| Predictor biases | 4,608 |
| Binary byte decoder weights and allocated bias array | 131,328 |
| Learned A1/B1 state gates | 1,536 |
| Bottom prototypes and task offsets | 131,072 |
| **Total** | **99,727,616** |

The formula is `sum(E_r)*(D*D+2*D) + 18*(D*D+D) + 256*(2*D+1) + 6*D + 2*(E_A1+E_B1)*D`, with D=256. It reproduces the archived 4,990,720 count at D=64 and E_r=64. It excludes fixed embeddings, fixed permutations, recurrent buffers, memory entries, and diagnostics, matching the existing counting convention. The binary head retains the existing allocated 256-row convention.

FP32 parameter storage is about **380.43 MiB**. The two-channel key/value memory payload grows to about **32 MiB**, excluding metadata. Runtime and snapshot memory are additional. Selected expert matrix arithmetic grows **16x per execution**, while total parameters grow about **20x**. Top-1 routing preserves conditional execution, but this is a real increase in active compute. In the latest 5M run there were 7.37 regional executions per byte including EC wakes. Router scans, memory lookups, local predictors and credit updates must also be included in throughput measurements; a parameter ratio alone cannot predict bytes/second.

## 4. Required implementation work

1. **Make width a real runtime/configuration dimension.** Core shapes, learned-state shapes, fixed arrays, diagnostics, native offsets, serializers and snapshot copies currently contain literal 64s. Regenerate fixed transports and embeddings at the chosen width and version the checkpoint layout.
2. **Replace every 64-bit feature-mask assumption.** Native routing, feature selection, credit distribution, temporal records, popcount overlap and AVX kernels depend on one uint64. Changing D alone would leave incorrect shifts, masks or coordinate coverage. Test coordinates on both sides of every 64-bit boundary.
3. **Support per-region expert banks.** Replace `(r*E+e)` addressing with regional offsets. Resize counters, prototype state and cached/transposed router keys accordingly. The parameter count must include the actual packed banks, not padded maximum-capacity arrays counted as live experts.
4. **Widen the entire useful path.** Include source messages, recurrent state, Y, the decoder, EC recall transforms, memory keys/values, and credit records. A wider expert feeding a 64D publication would preserve a central bottleneck.
5. **Calibrate initialization and update scale.** Preserve controlled per-coordinate activations, fan-in-scaled expert initialization, and the separate A1/Y decoder normalization. Adjust decoder initialization for wider inputs. Check activation distributions, tanh saturation, actual parameter updates and prediction changes at widths 64, 128 and 256 before transferring rates.
6. **Retain the CPU execution model.** Keep selected-bank updates, cached routing data and bounded local credit. Avoid introducing full-model gradient buffers or optimizer moments as a side effect of widening. Verify the generalized D=64 implementation against the present runtime before changing model size.

## 5. Training strategy learned from these experiments

### Keep the successful mechanisms and fix their relative timescales

The degradation-isolation study used frozen counterfactuals and matched replay to localize the main problem to A1 routing, with a secondary A1-readout penalty. Main/local cooling did not automatically cool prototype EMA adaptation. Turning off loss-driven routing worsened the replay, while slowing competitive prototype movement helped. Expert learning itself still contained useful changes.

Use explicit, separate rates for expert/task learning, local prediction, competitive prototypes, loss-driven routing, fitted centering statistics, learned memory gates, and A1/Y readout branches. During consolidation, slow competitive drift with expert plasticity while preserving useful task-driven routing. Preserve the A1 prototype-cooldown link and independent A1 decoder control, but calibrate their onset for a new model. Applying the fully cooled 5M settings from initialization would handicap learning before representations form.

### Use warmup, sustained learning, and deliberate cooldown branches

Train the new model from scratch on the established fresh document mixture. Use an initial approximately 1M-byte warmup, then sustain the learning rate selected on the proxy. At milestones, branch a checkpoint for a 5-10M-byte cooldown comparison while retaining the ongoing learning branch. Compare settled validation windows, not isolated minima. This separates learning progress from high-rate oscillation and avoids permanently trapping the only continuation at a finishing rate. Warmup-stable-decay literature supports this schedule pattern [L2]; SHoEbox's own cooldown history provides the immediate motivation.

For the width-128 proxy, an initial main-LR screen of **0.002, 0.004 and 0.008**, with local predictor LR initially main/3, brackets the archived early-training rate of 0.008. These are candidates, not a selected 100M setting. Calibrate actual updates and prediction movement across widths before transfer. Start with ordinary prototype adaptation for formation, then introduce slower adaptation as routing settles; use the successful approximately 0.1 inherited EMA scale as a mature-phase reference, with the stronger A1 slowdown during consolidation. Log effective rates, not just nominal route arguments.

There is no supported reason to switch the whole model to AdamW or a different learning algorithm now. Retain the normalized local/lace system as the scaling baseline. Rewarm/freeze experiments reduced deterioration but did not beat their source at the final endpoints. E4/V4 teaching changes and fixed contribution suppression also failed to improve endpoint test loss. Keep regional teaching multipliers and output gains at unity initially; measure actual local/task update paths without forcing them to equal magnitude.

### Make the larger run informative

Use the same fixed, document-excluded 2M validation for every size and preserve a separate final test. Because existing tests have already informed several decisions, reserve a new final test before the scale comparison. Track a fixed seen-training probe separately from held-out loss; changing-stream training averages cannot diagnose overfitting by themselves.

Track effective expert count, selection churn on a fixed probe, output rank, tanh saturation, actual update norms, local/task contributions, expired/overwritten lace credit, and controlled Y/recall ablations. A region's clipped local prediction MSE is not its language NLL. Measure regional causal contribution with matched interventions that preserve normalization.

Budget **500M fresh byte targets** for the first full 100M learning curve, with review checkpoints at roughly 25M, 50M, 100M, 200M, 350M and 500M. Keep 1-2B fresh bytes available if cooled validation continues improving. This is an experimental budget, not a fitted compute optimum. Do not apply a dense-model tokens-per-parameter rule directly to sparse byte-level SHoEbox, and do not assume the larger model must plateau at the old 175M point.

## 6. The next experiment I would run

First generalize the kernel and validate width-64 parity. Then compare fresh models on the same 25M-byte stream:

| Arm | Width | Experts per region | Parameters | Question |
|---|---:|---:|---:|---|
| Current-size control | 64 | 64 | 4,990,720 | New-policy reference |
| Expert-count scale | 64 | 256 | 19,638,016 | Does a larger bank break the plateau? |
| Width scale | 128 | 64 | 19,565,824 | Does wider active computation learn more efficiently? |

Use two initialization seeds and the same warmup/cooldown structure, with rates calibrated to each width. Compare both loss at equal byte exposure and loss against measured training time. A 25M run is an early screen; extend close outcomes rather than declaring a final capacity winner from early ranking. This comparison closely matches total parameters while changing the scaling direction. It is more diagnostic than jumping directly to another expert-count increase.

If width wins, proceed to the specified 99.73M model. The 100M model's first success criterion is a clear improvement over the selected 5M **common-test 1.760330** after validation-based selection on a shared protocol, accompanied by continuing improvement on fresh data. A margin of at least 0.02 nats/byte on a new matched evaluation would be a useful practical win threshold, not a predicted loss. The exact future NLL cannot be inferred from two differently optimized model sizes.

## Evidence and references

Run evidence examined:

- `SHoEbox_0_to_175M_Loss_Data.json`: retained/superseded branches, split boundaries and raw loss values.
- `SHoEbox_Fresh_175M_Package.zip`: exact runtime, checkpoints, policies, common tests, regional changes and data provenance. Package SHA-256: `e5eb1ea82e1379c2340fa9531671e8dd0d87373e49c75db66b29d04353eeb4c1`.
- `SHoEbox_Cooling_Adaptation_Report.md`; `SHoEbox_Freeze_Rewarm_144p25M_Report.md`; `SHoEbox_Degradation_Isolation_139p75M_Report.md`; `SHoEbox_A1_Cooldown_144p25M_Report.md`.
- `SHoEbox_E4_V4_Teaching_161M_Report.md`; `SHoEbox_Regional_Modulation_161M_Report.md`.
- Additional read-only probe: both selected checkpoints, first 131,072 targets of the shared validation, reset stream state and preserved fitted statistics. Probe NLLs: 1M 1.851760838; 5M 1.765674077. These are subset diagnostics, not replacements for the full test scores. Model fingerprints were unchanged after evaluation.

Primary research informing the recommendation:

- **[L1] Clark et al. (2022), Unified Scaling Laws for Routed Language Models.** Routed capacity and computation are distinct scaling axes. This supports separating expert-count growth from active-width growth; it does not supply a numerical scaling law for SHoEbox. https://arxiv.org/abs/2202.01169
- **[L2] Wen et al. (2024/2025), Understanding Warmup-Stable-Decay Learning Rates.** A continuing learning branch plus cooldown evaluations provides flexible training horizons. Its application here is a proposed adaptation to SHoEbox's local learner. https://arxiv.org/abs/2410.05192
- **[L3] Yang et al. (2022), Tensor Programs V.** Hyperparameter transfer can be stabilized through appropriate parameterization. Use the principle of checking width-dependent updates; the current local/lace learner is not automatically covered by Transformer muP results. https://arxiv.org/abs/2203.03466
