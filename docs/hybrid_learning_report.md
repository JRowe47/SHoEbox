# SHoEbox: hybrid learning Phase A

November 8, 2026. Six-region hierarchy; matched continuation from b4507b4.

Phase A reaches 2.188464 nats/byte versus 2.185723 for the matched legacy objective (Phase A minus legacy +0.002741; 0.13% higher NLL). Accuracy is 37.488% versus 37.743%.

Retain legacy as the default and keep Phase A available as an explicit objective mode. The middle-level predictors learned useful future-input structure, but L3's advantage over shuffled futures is small and Phase A did not improve byte prediction in this comparison. Strengthen L3 grounding before introducing owner-local credit assignment. The requested implementation and matched pilot are complete; no Phase B-D dynamics were added.

## What changed

L2 predicts four future A1 messages; S2 predicts four ordered A1/B1 pairs; L3 predicts four ordered L2/S2 pairs. Future labels are detached; the current representation and global byte path retain upstream gradients. Complete spans must stay in one document, batch row and forward window. The existing recurrent stream trajectory is unchanged; document masking applies to objective pairs. Live published messages receive variance and redundancy penalties. Sampling constructs no target records and runs no auxiliary heads. Training adds 83,200 parameters; inference remains 2,652,512.

## Matched run

Training uses 93 previously unused FineWeb documents, 262,144 targets per arm, B8/T256, joint AdamW 0.001, betas (0.9, 0.95), matrix decay 0.01 and gradient clip 1. Validation is unchanged: 65,536 targets across 23 documents, with 128-byte evaluation windows. All prior training, validation and reserved documents are excluded by ID and full source-text hash. Calibration uses a separate nine-document 8,192-target corpus; only its first 4,096 positions are observed.

Both arms start from b4507b4's 1,048,576-target checkpoint and retain its shared weights and all 464 saved Adam states at age 768. Two disjoint B8/T256 calibration windows observe 4,096 training bytes without any weight update. New predictors start with fresh Adam state. At each endpoint, active shared states are age 896; the candidate's new heads are age 128 and retained legacy heads stay at 768. Each arm ends at 896 lifetime optimizer updates and 1,310,720 optimizer target exposures.

Fixed Huber coefficients: L2 0.04311737, S2 0.03430131, L3 0.01301088. Calibration targets 10% of the owning publisher's global-byte gradient norm, bounded to [0.005, 0.2]. Variance weight 0.01, covariance weight 0.001, standard deviation target 0.1; source normalizer EMA rate 0.05 and variance floor 0.01. These coefficients stay fixed throughout training.

## Full held-out result

Byte losses and swap penalties are nats/byte; lower ordinary NLL is better.

| Measurement | Legacy | Phase A |
|---|---|---|
| Next-byte NLL | 2.185723 | 2.188464 |
| Next-byte accuracy | 37.743% | 37.488% |
| Decoder swap penalty | +0.002250 | +0.000530 |
| Decoder + feedback penalty | +0.007853 | +0.003659 |
| Training bytes/s | 183.7 | 183.5 |
| Peak RSS (MiB) | 1348 | 1352 |
| Stored parameters | 3,050,848 | 3,134,048 |
| Inference parameters | 2,652,512 | 2,652,512 |

Measured update time: legacy 23.78 minutes; Phase A 23.81 minutes. The arms ran concurrently, each with one CPU thread and the existing fused CPU kernels. Throughput is aggregate external byte targets per second, not single-stream generation speed. Peak RSS is process-wide.

## Predictive grounding and message health

| Region | Future Huber | Persistence | Shuffled future | Gain vs shuffled | Valid / eligible |
|---|---|---|---|---|---|
| L2 | 0.43552 | 0.66076 | 0.47038 | 7.41% | 15,856 / 16,377 |
| S2 | 0.44274 | 0.64602 | 0.49704 | 10.93% | 15,856 / 16,377 |
| L3 | 0.46282 | 0.55483 | 0.46627 | 0.74% | 3,567 / 4,088 |

Against shuffled-future labels, the largest predictive advantage is S2 (10.93% lower Huber); the smallest is L3 (0.74%). Persistence repeats the preceding ordered source bundle. Shuffling rotates complete future bundles across valid examples while retaining their internal source/time order. All latent comparisons use the same checkpoint's fixed training statistics.

Changing future observations after an identical 32-byte prefix leaves predictor inputs exactly unchanged and changes the subsequent teacher bundles. Normalized target-response MSE: L2 1.17021, S2 1.21398, L3 1.82349. This verifies sensitivity to future evidence; the shuffled-future comparison measures how much of that evidence the trained predictors anticipate.

Mean within-window message standard deviations: A1 0.135, B1 0.123, L2 0.193, S2 0.375, L3 0.155. Training pair coverage:
L2 98.30%; S2 98.30%; L3 93.21%. Coverage excludes incomplete future bundles and cross-document spans;
no target is borrowed from a later optimizer step. Per-region gradients, full
message-health histories and per-source losses accompany the checkpoints.

## Verification

All 62 existing checks and seven focused Phase A checks pass. An independent real CLI run exactly matches uninterrupted weights, Adam state, recurrent state, normalizers and RNG after pause/resume. Changed objective, source or document manifest is rejected. Both final checkpoints exactly reproduce their full held-out results after reload; validation leaves normalizers unchanged.

## Fixed-prompt samples

Same prompts, seed 731, temperature 0.8, 64 generated bytes.

### Legacy

```text
The ound Sarding and the dony sonvelen you the of that mare Sholfom.
In the mayn a sprowness the dony sonvelen you the of that matear offem.
```

### Phase A

```text
The (3 I SThing ENA-Onomaty ous Roverney Nothw of Envered Walloads s
In the mayn a sprowness the dongate velet you the of the cratial of cou
```

## Reproduction and handoff

Commands and API contracts: `docs/hybrid_learning.md`. Implementation retains
legacy objectives as the default. Calibration, exact CLI verification, endpoint
audits and report generation have dedicated scripts in `scripts/`.

The checkpoint ZIP contains the original seed, calibrated seed, both final
checkpoints, exact prepared corpora and manifests, cached FineWeb responses,
fixed coefficients, histories and verification records. Model source is published
in [implementation commit 67ac68c](https://github.com/JRowe47/SHoEbox/commit/67ac68c65feae1a2328dd0d3d94a26e2089a2f37),
branch `feat/hierarchy-pilot`; an attached patch contains
this implementation and documentation relative to b4507b4.

| Provenance | SHA-256 |
|---|---|
| Shared starting checkpoint | `4130cc9cb6f5674f231dd54ce3a0bc45cbaf6e028417676a0fbb8e4558f82301` |
| Fresh training bytes | `c321eb9c0a5df5d8cb18b9a22d2821933fbe51f7e5ebf613d5b1ca03ccc2d90a` |
| Fixed validation bytes | `6c77d22de33ebaf9b5f00e5056ea87ffa4aad15bd02e96f65351fc34395d2084` |
| Legacy endpoint | `2a808aca60ee4a23f6235ee0f041052aba5ae0e4e3ef39df32d6c5179f08512a` |
| Phase A endpoint | `728b3cdd04e514f760b37404def89e2999e180207846d6cc09d382297d5e31c1` |
| Runtime source | `0f1c40cec9a9a682c4f08a2e0272d332e5cc0351bb41a8644026dee289a797f2` |
