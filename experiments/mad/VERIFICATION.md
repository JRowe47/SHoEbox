# MAD protocol release qualification

The protocol implementation and the current development runtime are ready for
`main`. Corrected C remains the selected runtime. Route-local rates, selective
loops and the structured-state research adapter remain available with their
existing explicit controls.

## Checks performed

| Check | Result |
|---|---:|
| New Python protocol tests | 15 passed |
| New native protocol bridge checks | 23 passed under ASan/UBSan |
| Existing multiscale Python tests | 44 passed |
| Existing structured-state Python tests | 6 passed |
| Existing structured-state native checks | 19 passed under ASan/UBSan |
| Existing loop/native/lace/credit/routing/rate harnesses | All passed, including ASan/UBSan |
| Nine-task smoke run, two loop settings | 18 fits completed |
| Two-seed, four-hyperparameter selection exercise, two loop settings | 16 fits completed |
| Serialized prediction reloads | Exact for all 34 fits |
| Actual L3 execution | Present in all 34 fits |
| Test evaluation of unselected hyperparameters | None |

The smoke fits consume 4,096 observed input bytes each, totaling 73,728 bytes.
The selection exercise adds 65,536 bytes. These short runs qualify execution,
masking, comparisons and selection behavior; they are not a capability study.
The smoke calibration is provisional or unresolved, and no mechanism is promoted.

The all-task smoke includes frozen 0/1/2/4-pass sweeps, memory-read removal,
active L2/L3 expert permutations, independent longer-delay tests, and
compression zero-bottleneck replacement. Each pair has identical dataset hashes,
training exposure, total fitted parameters and recorded persistent-state
capacity. The source model remains unchanged by frozen probes.

The native bridge tests additionally verify exact public-runtime parity with
all labels enabled and identical training when only ignored labels change.
Compression decoder gradients pass finite differences; live checkpoint tests
preserve pending native learning and the fitted readout. Existing native
harness counts and source fingerprint are recorded in `verification.json`.
LeakSanitizer was explicitly disabled; AddressSanitizer and UndefinedBehaviorSanitizer
remained enabled.

## Evidence and reproduction

- `smoke_results.json`: complete nine-task curves, predictions and frozen audits.
- `selection_results.json`: optimizer selection and paired-seed cost comparison.
- `verification.txt`: test output and existing runtime harness summaries.
- `verification.json`: compact check counts and implementation source hash.
- `README.md`: protocol, task contracts, decision rules and execution commands.

```bash
python -m multiscale.tests.run_all --no-leak-check
python -m unittest experiments.structured_state.test_adapter -v
python -m experiments.structured_state.check_native --sanitize
python -m unittest experiments.mad.test_protocol -v
python -m experiments.mad.check_native --sanitize

python -m experiments.mad.run --preset smoke --execute --audit-probes \
  --out /tmp/mad-smoke.json

python -m experiments.mad.run --preset smoke --execute \
  --tasks memorization --steps 0 2 --seeds 1 7 \
  --lr-scales .5 1 --auxiliary 0 .1 --budget 4096 \
  --out /tmp/mad-selection.json
```

Measured runs record the development parent commit plus the SHA-256 of the
actual edited implementation. Their `dirty` flag is true because qualification
preceded the release commit. The protocol addition does not edit production
runtime source files or reinterpret existing checkpoints.
