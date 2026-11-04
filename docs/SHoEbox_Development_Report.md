# SHoEbox initial development report

Date: November 4, 2026

Repository: https://github.com/JRowe47/SHoEbox

Baseline: `JRowe47/Sparse-Hierarchy-of-Experts` at `b4bb7cfea18c3833819f6ed08973fd014f15f107`.

## Delivered implementation

The public repository preserves SHoE's history and experiments and adds `experiments/text_regions`. The default model has **2,561,212 unique parameters**, seven nodes in three state-owning regions, and all 256 input/output bytes. Its 1,536 physical expert MLPs contain exactly **1,228,800 expert-matrix parameters**. Each full round executes **168 expert calls per external byte**, or 336 for K=2.

R0 owns A, B and C. R1 owns U and P; U receives A+B and P receives U+A. R2 owns V and Q; V receives B+C and Q receives V+C. P and Q feed the shared byte decoder. B publishes to both branches while retaining one state owner. External graph depth is configurable; each internal SHoE retains three blocks.

The runtime ingests each byte once, commits forward stages in order, and reads lateral/descending messages from the preceding round's snapshot. Every further round reruns the complete node pipelines. Tensor payloads and semantic maps remain differentiable across node/region boundaries and rounds. Expert weights are shared; contextualizer and minKron states are independent.

Publications carry validity, per-row versions, epochs, occurrence/round clocks, namespaced memberships and bounded dependency IDs. Full graph state supports selective reset, detach, serialization, and storage accounting. Source addresses retain only the current occurrence and cannot pin an entire input window.

## Validation

**87 tests passed: 60 preserved checks and 27 new checks.** These cover exact legacy logits/state/gradient parity, shared-weight identity and independent state, snapshot replacement, all-state detach, inactive nodes, selective resets, chunk equivalence, future-suffix causality, target isolation, independent stage order, K-independent source counts, per-round rerouting, upstream/earlier-round gradients, unavailable views, equal-execution edge interventions, configurable topology, and checkpoint continuation through the next optimizer step.

A controlled transport test shows that changing A cannot affect Q in the first round but can in the second. A full-width 128-byte, two-round regression verifies finite gradients. A storage regression checks that long input windows do not remain referenced by carried provenance.

## Learning checks

Both overfit runs use the repeating byte fixture `ABCD\n`, seed 20261104, batch 4, eight-byte TBPTT segments, K=1, and the inherited AdamW settings, warmup/taper and clip=1. Each uses a contiguous 90/5/5 split. The validation probe is one fresh 32-target batch; the last 5% is unused. These runs test optimization and generation on the declared tiny fixture.

| Configuration | Parameters | Steps | Exposed byte targets | Initial validation nats/byte | Final validation nats/byte |
|---|---:|---:|---:|---:|---:|
| Reduced-capacity seven-node graph | 246,142 | 100 | 3,200 | 5.553991 | 0.153478 |
| Default seven-node graph | 2,561,212 | 80 | 2,560 | 5.481635 | 0.238733 |

The recorded generated samples reproduce the fixture pattern. Training checkpoints include weights, optimizer, all carried state, publications, stream positions, dataset identity, RNGs, and the original schedule horizon.

The default K=2 model also completed 2 consecutive optimizer steps on verified Tiny Shakespeare bytes with carried state, batch 4 and 128-byte segments: 1,024 byte targets and 344,064 expert executions. This is a streaming shakedown; the longer language experiment is pending.

## CPU engineering profile

Profiles use default model capacities, batch 4, 128 bytes per row, two CPU threads, seed 20261104, and one timed iteration in separate processes. Data is random bytes for execution/gradient checks. Each iteration exposes 512 byte targets. The training measurement includes forward, backward, clipping and optimizer step. Different compute budgets are stated explicitly; this table does not compare learned quality.

| Model | Unique parameters | Expert calls/byte | Inference seconds | Training-step seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| Original HAT-minKron | 1,624,588 | 48 | 0.147 | 0.663 | 577.6 |
| SHoEbox K=1 | 2,561,212 | 168 | 5.705 | 23.191 | 2119.9 |
| SHoEbox K=2 | 2,561,212 | 336 | 12.040 | 41.630 | 3853.4 |

| Model | Carried state bytes (batch 4) | Optimizer-state bytes | Routing forward seconds | Expert forward seconds |
|---|---:|---:|---:|---:|
| Original HAT-minKron | 32,832 | 12,997,344 | 0.041 | 0.035 |
| SHoEbox K=1 | 184,856 | 19,698,344 | 1.544 | 1.050 |
| SHoEbox K=2 | 185,912 | 20,492,392 | 3.243 | 2.321 |

Component times come from a separate instrumented inference pass: routing means router/lace forward calls; experts means split-bank forward calls. Persistent state excludes weights, optimizer storage and training activations. The event-major reference is slower than the packed baseline; the runtime prioritizes explicit causal boundaries before optimization.

An initial K=2 run with delayed gates starting at 0.5 produced nonfinite gradients at 128 bytes. Delayed gates now start at **0.05** while forward gates start at 0.5; both remain trainable. The same full-size profile passes, and the regression is in the test suite. No cross-region detach, replay, gradient replacement, or minKron recurrence change was introduced.

## Milestones

| Milestone | Status | Evidence / remaining work |
|---|---|---|
| M0 | Complete | Pinned baseline/history; 60 baseline checks; common profiles; seeded episode fixtures. |
| M1 | Complete | Latent output and optional decoder; reused SymbolLevel; exact baseline logits/state/gradient parity. |
| M2 | Complete | Staged tensor ports, dependency IDs, namespaces, functional state, reset/detach and full resume. |
| M3 | Complete | Seven-node one-pass byte LM; dedicated paths, edge masks/substitutions, overfit and generation. |
| M4 | Engine implemented | Ordered stream cursors, boundaries, carried state and TBPTT pass. Learned delayed recall/correction acceptance remains. |
| M5 | Engine implemented | K=1/2/4 schedules, previous-round feedback, rerouting, traces, causal transport and early-round gradients pass. Learned composition remains. |
| M6 | Next | Shared/private representation objectives, eligibility packets and explicit post-optimizer inter-node utility updates. |
| M7 | Pending | 512-episode acceptance tests, natural-text experiment, matched-compute controls and three-seed comparisons. |

## Next development increment

1. Wire generated delayed-recall/correction episodes into the ordered trainer, including explicit multi-view availability and supervised answer masks.
2. Train carried-state and reset-state controls with relevant facts beyond the 128-byte segment, then score 512 fresh episodes per seed.
3. Add shared/private representation objectives and post-optimizer utility commits independently, logging every term.
4. Compare K=1,2,4 query computation from the same saved prefix and weights; add equal-execution bridge substitutions and a comparable-execution feedforward control.

No new local view objective or inter-node routing-utility buffer is silently enabled. The current graph trains on language loss plus existing balance losses.

## Reproduce

```bash
git clone https://github.com/JRowe47/SHoEbox.git
cd SHoEbox
python -m pip install -r requirements.txt
python scripts/check_all.py
python -m experiments.text_regions.train --context 8 --batch 4 --steps 80 --out artifacts/default_overfit
python -m experiments.text_regions.evaluate --rounds 2 --context 128 --batch 4 --out artifacts/k2_profile.json
```

The experiment README documents tensor/view inputs, masks, substitutions, generation, state reset and exact checkpoint resume. The original specification is included unchanged under `docs/SHoE_Text_Development_Specification.md`. Raw profiles, learning histories, samples, verification results and example traces are in `experiments/text_regions/results`.
