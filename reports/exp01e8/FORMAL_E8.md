# EXP01-E8 final qualification registration

Registered after the fixed E8 development pilot and fresh generic screen, before
final fitting or either unopened audit. Starting main: c29693f8a2bd2b1c11ed895dda26b455d3e7e847.

The seed-7 pilot retains 99.4141% native binding after 1M language targets;
offset mean 98.0469%, worst 95.3125%, both orders 99.2188%, independent
0/64/128/256 delays 100%. Fresh three-seed generic development is
2.236391303 versus 2.227510914 nats/byte (+0.008880390 <=0.01), copy
71.8750% versus 72.0052%, reconstruction 77.9297% versus 77.4089%.
These trigger the conditional comparison in the write-once E8 amendment.

## Fixed comparisons and budget

Use E2-family grouped train/development/audit and unchanged order seeds 701/702,
spacing RNG 1702, 0–7 record spacers, all 64 balanced phases, 250K charged
native warmup plus 2M continuation for each arm and each seed 1/7/19.
At nominal 500K compare native every-byte, true frozen D3 answer-mask,
generic-loss gate (E7 kind 3), E8 locally supervised gate, and E8 no-local.
The existing E1 adapter's paired_latest is numerically historical D3; it is
used for that control, never the calibrated adapter's internal alias.
All five arms then receive the original pinned retention corpus's 1M next-byte
targets without local labels, followed by every binding gate. Total 48.75M bytes.

Repeat native/frozen D3/E8 at nominal 1M and 5M, seeds 1/7/19 and 2.25M
bytes per arm/seed: 20.25M each, 40.5M combined. Capacity does not alter
initialization, local targets, rates, state, vocabulary, evaluation or thresholds.
Additional final qualification: 89.25M charged bytes. Initial E8 screen:
3.25M pilot plus 13.5M fresh generic fits = 16.75M. Repeated native/D3 controls
are charged anew; the historical E7 pilot is a previously charged comparison.

Lock every final checkpoint, cursor, source manifest and development outcome
before opening the E2-family binding and E7 generic audits. Score all registered
arms, never choose a checkpoint from audit results. The 500K final development
gate and isolated throughput precede audit access. If either fails, keep audits
unopened. Capacity comparisons can be reported as development diagnostics.
No old opened audit informs this recipe or subsequent revision.

## Unchanged gates and measurements

- Full-256-byte autoregressive native two-answer exactness: mean >=95%,
  every seed >=90%; identical thresholds for first and second answers.
- Both cloned-prefix query orders: mean >=90%, every seed >=85%.
- All 64 offsets: mean >=95%, worst offset averaged across seeds >=90%.
- Independent +0/+64/+128/+256 delays: each mean >=90%, each seed >=85%.
- No-query drop >=20 points; reverse strictly hurts. Preserve no-head,
  no-state, uniform, reverse, unpair and gate-off/on measurements.
- Audit 1,024 primary presentations; 256 for lesions/counterfactuals;
  same first 64 at each offset, next 16 crossed with all offsets at each delay.
  Shared worlds across seeds are not independent samples.
- Fresh generic audit: mean language increase <=0.01 nats/byte, copy and
  separately budgeted final-L3 reconstruction losses <=1 percentage point.
- Isolated full-distribution inference >=80% native: pinned CPU, three
  interleaved 65,536-byte repeats, compilation/cloning excluded.

Report generic-text gate activation before/after binding and retention,
language NLL with natural and forced-off gate, fitted parameter/state counts,
every byte including padding, all objective/local update events, training CPU,
evaluation CPU where instrumented, and correctness-only discarded updates.
No fixture-local label is supplied on generic language/copy/reconstruction.

The frozen diagnostic helper's nested-clone intervention bug was corrected
after initial fits; it was never imported by their execution paths. Both helper
hashes and original fitted source are retained. This changes no learned weight,
training source, model policy, data order or score used for qualification.

Production remains v19 unless every gate passes. Preserve E1–E7 and all failed
fits; preregister and execute the next focused revision if qualification fails.
Direct verified main publication, including negative results, is authorized.
