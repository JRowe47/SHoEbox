# SHoEbox: fresh-data joint training

November 8, 2026. Six-region predictive hierarchy; 3,050,848 parameters.

Held-out next-byte NLL fell from 2.392272 to 2.236783 (6.50% lower; byte perplexity 14.40% lower). The combined upper-message swap penalty increased from 0.001448 to 0.004213 nats/byte.

Keep joint training with fresh data and the longer window as the preferred continuation. The endpoint monitoring loss is still improving. Upper-context influence has grown, while the direct upper decoder contribution remains negligible.

## Policy and completed training

The previous joint run was still improving and beat the matched freeze/thaw
schedule (2.392272 vs 2.448039 nats/byte). This phase therefore keeps all regions
learning at AdamW 0.001 and extends both the corpus and the gradient window.

| Setting | This phase |
|---|---|
| Start | 512 lifetime updates; 524,288 lifetime target exposures |
| Added training | 256 updates; 524,288 byte targets |
| End | 768 lifetime updates; 1,048,576 lifetime exposures |
| Corpus | 195 fresh FineWeb documents; one pass, no repeats this phase |
| Credit window | B8/T256, up from T128; 16 L3 updates per gradient window |
| Optimizer | Joint AdamW 0.001; betas (0.9, 0.95); matrix decay 0.01; clip 1 |
| Auxiliary objectives | Next-four-byte weight 0.2; next-sixteen-byte weight 0.1 |
| Held out | Same 65,536 targets / 23 documents; evaluation chunks stay at 128 |

Weights, Adam moments and update ages are preserved. Recurrent state and the
corpus cursor reset for unrelated streams. There is no freezing, route attenuation
or learning-rate anneal. Lifetime exposure counts retain each phase's actual
window size; the older T128 updates are not rescaled to T256. Across the complete
training history, 718,848 distinct target positions have been used.

## Held-out result

All losses are nats/byte. Prediction loss is lower-is-better; swap deltas are
intervention loss minus ordinary loss.

| Measurement | Before | After | Change |
|---|---:|---:|---:|
| Next-byte NLL | 2.392272 | 2.236783 | -0.155489 |
| Next-four-byte NLL | 3.043644 | 2.979965 | -0.063679 |
| Next-sixteen-byte NLL | 3.209371 | 3.182433 | -0.026939 |
| Decoder-summary swap delta | -0.000003 | -0.000055 | -0.000052 |
| Decoder + feedback swap delta | 0.001448 | 0.004213 | +0.002766 |

Next-byte accuracy: **32.45% -> 36.02%**.
Decoder-only swapping preserves producer trajectories. The combined swap also
mismatches top-down messages and recomputes lower trajectories from fresh states.
A positive penalty means those upper messages help the ordinary prediction.
Fresh data and a longer gradient window change together in this continuation.

## Runtime and verification

Training completes in **42.72 minutes** of measured update
time, averaging **204.57 aggregate bytes/s** on one
CPU thread with the AVX512 backend. Peak process RSS: **1257.0 MiB**.
These counts refer to external byte targets across all eight streams.

**62 checks pass.** An independent CLI transition/pause/resume run exactly matches
uninterrupted model, Adam, recurrent state and RNG. It verifies state reset,
preserved Adam update age and mixed-window exposure accounting. The final
checkpoint exactly reproduces the full held-out byte and future-span scores
after reload; parameters and recurrent state are finite. The phase's initial
8,192-target monitoring score also exactly matches the seed checkpoint.

## Data and reproduction

Source: [FineWeb sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb),
official viewer rows 1000-1299. Complete response bytes and checksums are included
in the checkpoint bundle. Preparation excludes every prior training, validation
and reserved document by ID and full source-text hash. Validation bytes and
document boundaries are identical to the previous phase.

Commands: `docs/fresh_joint_training.md`. Data preparation:
`scripts/prepare_hierarchy_phase.py`. Endpoint audit:
`scripts/audit_hierarchy_checkpoint.py`. Full phase metrics, per-region gradients,
optimizer state, exact prepared data and raw source responses accompany the
checkpoint. The seed checkpoint is also retained for reproduction.

- Runtime source SHA-256: `5e89dca3ac52f1d1c0b6e5e4dd7f74ccb7dbf0c1f1314736a3dde99a8fa375d8`
- Initial checkpoint SHA-256: `032aa8a4aa00e7e591fc0ebfa2cd00a7ac352e70b8b8ae8a22d9ac33744db149`
- Final checkpoint SHA-256: `4130cc9cb6f5674f231dd54ce3a0bc45cbaf6e028417676a0fbb8e4558f82301`
- Training SHA-256: `d184ed71dc4cd7b2547498f66389233ccd4c89d4231c9af251c7e98f975423ff`
- Validation SHA-256: `6c77d22de33ebaf9b5f00e5056ea87ffa4aad15bd02e96f65351fc34395d2084`
