# MAD capability research campaign

Base runtime: `6384d8ca072075ec58010b4c3d5964e9e60fcd3e`.

The first stage calibrates all nine existing MAD task families with corrected C,
three model seeds (1, 7, 19), 128 development cases, 256 training episodes,
250,000 observed bytes per fit, and the registered base setting.
The native configuration is 500K requested fitted parameters, LR scale 1 and
native auxiliary weight 0.1. Actual capacities and dataset digests are recorded.
No new architecture or production defaults are introduced.

Decisions use development scores only. Choose a learnable unsaturated difficulty
before interpreting a paired mechanism result. Next stages revisit upper loops
and final-only/head-only/intermediate-state supervision, then inspect readout and
retention if those tests remain near chance. Held-out accuracy, seed variation,
matched bytes, measured CPU-time checkpoints and frozen interventions are the
decision evidence. Preserve all failures and uncalibrated cells.

This branch runs the existing verified harness remotely after the interactive
execution host became unavailable. The interrupted local launch has no recovered
results and is excluded from all reported measurements.

This is SHoEbox's byte adaptation of MAD, not an original MAD leaderboard score.
