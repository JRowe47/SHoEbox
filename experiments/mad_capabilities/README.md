# MAD capability research — completed campaign

The completed report, summary and primary PNG are in
[reports/mad_capabilities](../../reports/mad_capabilities).

The campaign completed 435 fits and 131.25M observed training bytes. It includes
nine-task calibration, calibrated 0/2/4-pass comparisons on six task families
with equal six-policy optimizer searches and three seeds, and 1M/5M reruns of
intermediate state supervision. Corrected C remains the production runtime.

Key results: shared-expert loops harm compression, provide a small recall
training benefit that mostly survives with inference loops disabled, and do
not solve procedural tasks. Intermediate state credit improves all three seeds
at both model sizes, but final task accuracy remains near 15%. Query-dependent
binding, readout conditioning and retention are the next targets.

The initial host-disconnected launch supplied no recovered measurements.
All reported fits are completed remote jobs with exact checkpoint reloads.

Commands:
    python -m experiments.mad_capabilities.screen --task compression --out results/compression
    python -m experiments.mad_capabilities.state_campaign --size 1000000 --budget 1000000 --out results/state_1000000.json
    python -m experiments.mad_followup.screen --task variable_tracking --out results/variable_tracking
    python -m unittest experiments.mad_followup.test_baselines -v

This is SHoEbox's byte adaptation and registered state extension of MAD, not an
original MAD leaderboard score. Full task contracts remain in experiments/mad.
