# SHoEbox Action 6: sparse bank epochs and worker overlap

14 November 2026

Implementation: [ef8d9d2](https://github.com/JRowe47/SHoEbox/commit/ef8d9d22a9a0bfc015dbabd4571799bbc8f40767); baseline: `dc3b9b0`.

Action 6 is implemented. At main/local LR 0.16 / 0.05333, the four-byte worker profile reaches median 1,697.3 B/s versus 1,512.4 B/s for Action 4. The median paired throughput gain is 12.2%, and time to NLL 3.33 improves by 11.2% (3/3 seeds faster). Median final NLL is 3.08759 versus 3.08863.

Advance four-byte sparse epochs with one-epoch lag and one learning worker as the preferred measured CPU training profile.

## Matched natural-text qualification

| Profile | LR | B/s | Held-out NLL | Target seconds | Bank MiB |
| --- | --- | --- | --- | --- | --- |
| Action 4 event | 0.16 | 1,512.4 | 3.08863 | 9.50 (3/3) | 0.0 |
| 4-byte + worker | 0.16 | 1,697.3 | 3.08759 | 8.44 (3/3) | 10.3 |
| Action 4 event | 0.32 | 1,488.9 | 3.07513 | 9.63 (3/3) | 0.0 |
| 4-byte + worker | 0.32 | 1,702.2 | 3.05125 | 8.11 (3/3) | 10.3 |

Host: AMD EPYC 9V74 80-Core Processor; g++ (Debian 14.2.0-19) 14.2.0; production -O3 -march=native -std=c++17 -pthread. Caller CPU 0; learning worker CPU 1. Nominal 1M: 18 regions, width 64, 12 experts per region; 1,014,800 fitted parameters with compact rank-16 forecasting and Y stride 8. Learned frames are retained. Natural-text qualification uses 17,408 training targets, 1,024 separate validation targets and matched seeds 7, 11, 19. Data is the pinned repository README/STATUS fixture and held-out cortical_fineweb documentation, not a FineWeb corpus run. Only native training calls are timed, including logits export and worker quiescence. Compilation, construction, evaluation, inspection and file IO are excluded. All training bytes, including startup, count in elapsed training time; there is no excluded warmup for these quality comparisons. Target time is the first observed crossing at 2,048-byte validation intervals. Timings, tests and profiling run separately.

The short screen tests 30 combinations: event, historical pipeline and all eight epoch/lag pairs at main LR .08/.16/.32, with local LR main/3. A 17,408-byte seed-7 rate comparison and three matched seeds at .16 and .32 extend that screen. The primary comparison retains the prior selected .16 rate. Repeated-text checks use .08, 8,192 targets, three seeds and target NLL 2.62; document checks use four 512-target documents and irregular 257-byte chunks.

## Storage and worker selection

| Profile | B/s | Reserved bank MiB | Held-out NLL |
| --- | --- | --- | --- |
| Action 4 event | 1,524.7 | 0.0 | 3.40758 |
| 4-byte serial | 1,716.9 | 10.0 | 3.40778 |
| 4-byte + worker | 1,788.4 | 10.0 | 3.40778 |
| 16-byte zero lag | 1,637.9 | 21.4 | 3.40779 |
| 16-byte serial | 1,620.2 | 21.3 | 3.40758 |
| 16-byte + worker | 1,732.0 | 21.3 | 3.40758 |
| 4-byte dense | 1,458.5 | 16.0 | 3.40778 |

At LR .08 over three rotated 8,192-byte trials, sparse four-byte serial jobs use 10.0 MiB of reserved bank arenas versus 16.0 MiB for dense copies. Their held-out NLL is identical. Sparse serial is 17.7% faster. Adding one learning worker improves the median throughput by 4.2% over the same serial schedule. Sixteen-byte jobs retain roughly twice the arena storage and do not beat the four-byte worker configuration in this screen. Bank MiB excludes the main model, record archives and allocator overhead; full process RSS is in the JSON results.

## 5M storage cross-check

| Profile | LR | B/s | Held-out NLL | Target seconds | Bank MiB |
| --- | --- | --- | --- | --- | --- |
| 4-byte serial | 0.08 | 1,426.1 | 3.59642 | - | 15.9 |
| 4-byte dense | 0.08 | 682.7 | 3.59642 | - | 52.5 |

The 5M storage cross-check uses 64 experts per region and three rotated 4,096-target runs at LR .08. Sparse and dense jobs produce identical held-out loss; the comparison isolates version copying and storage at the larger bank count.

## Repeated-text check

| Profile | LR | B/s | Held-out NLL | Target seconds | Bank MiB |
| --- | --- | --- | --- | --- | --- |
| Action 4 event | 0.08 | 2,062.5 | 2.38954 | 0.86 (3/3) | 0.0 |
| 4-byte + worker | 0.08 | 2,163.5 | 2.38957 | 0.85 (3/3) | 12.8 |

## Implementation

**Sparse versions and ownership.** The committed upper content banks are immutable during each epoch. Unchanged banks use references; only changed W tiles and bias/router vectors are copied. Two reused arenas retain shared input values and clipped low-rank factors. Router-only jobs do not copy W. The dense-copy oracle uses identical arithmetic.

**Epochs and worker lag.** Lengths 1/4/8/16 and lag 0/1 epochs are separate configuration fields. Lag 0 commits at the end of the epoch. Lag 1 reads the prior visible bank version, commits the preceding job after the current epoch, and applies the new factors to that committed base. Each job records read/base versions, commit sequence, objective mask and step interval. One caller owns commits; the worker only owns immutable job data.

**Learning boundaries.** Root union, per-request clipping and router normalization retain their original scopes. Roots are consumed into factors before API boundaries and are not merged across bytes. Event replay remains per step, followed by terminal synthetic credit. MinKron context, adapters, forecasting, bottom experts and decoder retain online schedules; slow Y retains its separate stride.

**Supported policies.** One or two MinKron context heads, local event or full ancestry credit, compact forecasting, fast/slow Y and either frame policy are supported. This supplies sparse ancestry epochs for MinKron context. The older dense-context reverse epoch remains unchanged and cannot be combined with this schedule. The bank clock counts training bytes plus forced document-drain passes, separately from primary byte clocks.

**Checkpoint and reset.** Version 46 and wire extension 3 preserve completed pending results and unfinished factor arenas. Save/load, snapshots, worker changes and arbitrary chunks preserve continuation. Frozen evaluation parks pending jobs and uses committed banks. A reset bug is repaired: final replay factors are now committed before their producer records are released. Historical pipeline checkpoint formats remain loadable.

```python
from shoebox_reference import fresh_model
m = fresh_model(
    profile='epochs', pipeline_workers=2,
    bank_epochs=dict(epoch_bytes=4, lag_epochs=1, storage='sparse'),
    prediction=dict(compact=True, rank=16, slow_stride=8))
# Ordinary run/save calls preserve incomplete epochs.
# m.finish_pipeline() is an explicit commit boundary.
```

## Validation

Final production: 37 tests passed, including all eight schedules, worker/serial/unfused/dense parity, partial restarts, an 8,193-byte wrap/collection continuation, task coverage, frozen banks, malformed imports, reset replay and causal frozen evaluation. Four targeted 5M checks and two explicit step-clock checks passed. The broader 124-test sweep passed 122 in production; its two tests that require diagnostic counters passed in the separate audit build. ASan/UBSan passed every epoch/lag pair with serial and worker execution. LeakSanitizer is unavailable under this host's ptrace environment; address and undefined-behavior checks ran with leak detection disabled.

## External production sampling

| Profile | Thread | CPU seconds | Wait samples | Copy/alloc samples | Other/unresolved |
| --- | --- | --- | --- | --- | --- |
| Action 4 event | caller | 6.35 | 0.0% | 0.3% | 28.1% |
| 4-byte serial | caller | 5.76 | 0.0% | 0.5% | 33.9% |
| 4-byte + worker | caller | 5.28 | 0.8% | 1.0% | 32.6% |
| 4-byte + worker | learning worker | 0.68 | 0.0% | 0.0% | 90.8% |
| 16-byte + worker | caller | 4.88 | 0.0% | 0.8% | 32.4% |
| 16-byte + worker | learning worker | 0.75 | 0.0% | 0.0% | 98.7% |

External production instruction sampling uses a separate diagnostic run with 1,024 warmup and 8,192 measured targets at main/local LR .04/.01333. Samples are wall-time instruction positions, not inclusive CPU attribution. Synchronization and copying are observed outside training. The four-byte worker uses 0.68 CPU seconds versus 5.28 for the caller; reverse/credit work accounts for 37.0% of caller samples. This favors one learning worker; a larger disjoint-bank worker team is not selected. Most worker wall samples are unresolved libc locations, so the table does not measure a precise worker wait fraction. Inlined work is attributed to its containing symbol. No inline timers or historical monitoring counters were added.

## Remaining work

The measured 1,697 B/s profile remains 8.84x short of 15,000 B/s on this host and export mode. Next: Action 7, dedicated forward-only runtime; Action 8, resident GPU and independent-stream batching; Action 9, longer 1M/5M/100M qualification, joint tuning and final time-to-quality selection. This action publishes compact code and evidence without datasets, trained models or temporary snapshots.

Reproduction: `src/measure_bank_epochs.py`, `src/profile_bank_epochs.py`, `tests/test_bank_epochs.py` and `tests/run_bank_epoch_sanitizer.py`. Full policy: `BANK_EPOCH_POLICY.md`. Compact raw evidence: `results/action6_*`.
