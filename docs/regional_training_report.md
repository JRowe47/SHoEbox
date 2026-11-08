# SHoEbox regional training cycle

November 8, 2026. Six-region predictive hierarchy; 3,050,848 parameters.

The regional cycle did not improve held-out next-byte prediction over matched joint training. Cycle minus joint: **+0.055767 nats/byte**. Keep joint AdamW as the default. The cycle remains an opt-in experiment; do not promote this schedule on this result.

## Matched result

Both arms start from the same 256-update checkpoint, including Adam moments,
runtime state and corpus position. Each completes **256 additional updates /
262,144 additional byte targets**: 512 lifetime updates and
524,288 lifetime target exposures. Held-out evaluation covers **65,536
targets across 23 separate documents**. All prediction losses are nats/byte; lower is better.

| Measurement | Joint AdamW | Regional cycle |
|---|---:|---:|
| Next-byte NLL | 2.392272 | 2.448039 |
| Next-four-byte NLL | 3.043644 | 3.055406 |
| Next-sixteen-byte NLL | 3.209371 | 3.210510 |
| Decoder-summary swap delta | -0.000003 | 0.000679 |
| Decoder + feedback swap delta | 0.001448 | 0.005682 |

Swap deltas are intervention NLL minus ordinary NLL; positive means mismatched
upper messages worsen prediction. Decoder-only swapping leaves feedback and
producer trajectories intact. The combined intervention also swaps top-down
contexts into A1/B1/L2/S2 and recomputes their trajectories from fresh states.

The starting checkpoint scores 2.615519 nats/byte. Swapping all outward upper messages adds 0.001180 nats/byte; feedback-only swapping adds 0.000849.

## Implemented regimen

| Phase | Updates per cycle | Learning parameters |
|---|---:|---|
| Upper development | 32 | L2/S2/L3, future heads, Y and nonlocal decoder |
| Lower alignment | 32 | Byte embedding and A1/B1 |
| Joint learning | 64 | All; fast producers and local projection use 0.5x learning rate |

Both future objectives must improve by 2% on their training EMA before lower
alignment is enabled. The existing checkpoint already meets this condition.
Every fourth upper-phase chunk scales the decoder's direct local slot to 0.5;
producer inputs remain intact. The endpoint uses every route at full strength.
Executed phases: {'upper': 64, 'lower': 64, 'joint': 128}; attenuated chunks: 16.

Freeze/thaw preserves Adam moments and stops both momentum updates and weight
decay for frozen parameters. States continue evolving, and gradients pass
through frozen modules to their trainable inputs. No additional replay or
regional execution dropping is introduced. Joint training remains the default.

## Validation and reproduction

**59 checks pass**, including frozen parameter/moment invariance, credit in both
directions through frozen regions, producer-preserving decoder attenuation,
feedback intervention causality, exact optimizer migration and exact resume.
A separate six-update command-line pause/resume run reproduces model, optimizer,
state, RNG, corpus position and controller exactly across all three phases.
Both final checkpoints exactly reproduce their complete held-out byte and span
scores after reload; parameters and runtime states are finite.

Training uses B8/T128, AdamW 0.001 with betas (0.9, 0.95), 0.01 matrix weight decay,
gradient clipping at 1, and unchanged 0.2/0.1 future-loss weights. There are
194,560 unique training targets; the corpus repeats. Runs share a CPU host and
use one CPU thread each. Observed aggregate training throughput:
196.36 bytes/s joint and 205.14 bytes/s cycle.
These timings are not an isolated hardware benchmark.

This is one matched continuation, with two schedule cycles. Rotation, fast-group
learning rates and local-route attenuation change together. Per-component effects
are not separated. GradNorm and adaptive phase lengths are not implemented here.

Reproduction commands and parameter roles: `docs/regional_training.md`.
Machine-readable comparison: `docs/regional_training_results.json`.
Source digest: `e374cd21f035a55c93f2cddb23834c7ba4ff0aff2ec2c7b129ef47860a9f29d5`.
Initial checkpoint SHA-256: `dbcae7c9da6adb4e2dc0175468ad10b1b96e2b200bc51204736c4c0ef71503e2`.
