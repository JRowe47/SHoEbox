# SHoEbox development handoff

15 November 2026 (America/Denver). Runtime baseline: minimal corrected C, checkpoint v19, source `317fbcd023b9d9b3c2b359ba8f99f3dadbf46783`.

**Next action: EXP01-B — learn factorized query/key selection and value readout online over existing evidence.** EXP00 and EXP01's initial diagnostic screen are complete. The screen localizes two-key binding failure to selection/readout learning: minimal L3 supports 99.93% strict first-query recovery with a separately supervised offline probe, while native two-answer exactness is 49.45%. Generic online heads do not repair it. Minimal v19 remains the production default.

## Read first

1. [Current minimal capability matrix](docs/capabilities/SHoEbox_Minimal_Capability_Matrix.md): all 66 capabilities and eight engineering enablers.
2. [Minimal baseline specification and qualification](reports/minimal_core/SHoEbox_Minimal_Core_Baseline.md).
3. [EXP01 findings, literature and costs](reports/exp01/SHoEbox_EXP01_Query_Readout.md) and [implemented adapter/commands](experiments/query_readout/README.md).
4. [Completed EXP00 results](reports/exp00/README.md) and [task/control implementation](experiments/mad_next/README.md).
5. [Research plan](docs/capabilities/SHoEbox_Capability_Research.md) and [machine-readable backlog](docs/capabilities/SHoEbox_Experiment_Backlog.json).

## Runtime contract

| Item | Minimal baseline |
| --- | --- |
| Public API | `shoe_multiscale.Model` / `fresh_model` |
| Explicit selection | `core="minimal"` |
| Width / clocks | 64; 1/4/16/64 bytes |
| 1/4-byte scales | Corrected-C sparse experts; exact search; coverage-only warmup |
| 16/64-byte scales | Ordered trainable summaries; diagonal recurrence; memory reads |
| Upper output modulation | Learned constant gain/bias vectors |
| Memory | MinKron context, addressed memory and retained-unit retrieval |
| Learning | Separate fast/slow/query rates; reconstruction and bounded child credit |
| Prediction | One forward pass |
| Persistence | v19; exact continuation including partial constituents and pending credit |
| Historical comparator | `Model(core="full")`: full C, v10 coverage-only warmup |
| Excluded extensions | Upper expert banks/routing/forecast heads; dynamic upper modulation; D predictors; refinement loops; route-rate cycles; experimental credit multipliers |

Capacity moves into the surviving expert banks in a 2:1 ratio. Simplification does not reduce the requested parameter budget.

| Nominal parameters | Actual fitted parameters | Experts by scale |
| --- | ---: | --- |
| 500K | 498,367 | 56 / 29 / 0 / 0 |
| 1M | 996,799 | 135 / 68 / 0 / 0 |
| 5M | 4,996,927 | 766 / 384 / 0 / 0 |

Use explicit architecture arguments and inspect the resulting model:

```python
from shoe_multiscale import Model

with Model(core="minimal", parameters=500_000, seed=7) as model:
    stats = model.stats()
    assert stats["core"] == "minimal"
    assert stats["routing_arm"] == "C"
    assert stats["level_kinds"] == ["expert", "expert", "summary", "summary"]
    assert stats["experts"][2:] == [0, 0]
```

The public default and research defaults differ. `experiments.mad.backend.BenchModel` defaults to full for historical reproducibility; pass `core="minimal"` explicitly. The existing EXP00 runner reproduces its historical experiment and is not an EXP01/minimal qualification command. Check or extend its adapter selection explicitly before reuse.

Do not substitute `Model(routing_arm="C")` for `Model(core="full")`: explicit historical C can select the earlier v9 policy. Experimental constructor options can select full when core is omitted; explicit minimal rejects incompatible extensions. Loaded checkpoints preserve their saved architecture and are not converted to minimal.

The explicit `bind/lookup/payload` API remains callable, but automatic association dispatch is excluded. Do not use that external exact store as a hidden solution to learned query binding.

## What the minimal baseline actually demonstrates

Three paired seeds, nominal 500K models. FineWeb: 1M next-byte targets per fit. Synthetic tasks: 250K charged input bytes per fit.

| Measurement | Full C | Minimal |
| --- | ---: | ---: |
| FineWeb test NLL, nats/byte | 2.358039 | 2.252588 |
| Selective-copy token accuracy | 73.70% | 73.83% |
| Selective-copy exact whole answers | 47.40% | 47.66% |
| Bounded-state reconstruction token accuracy | 70.77% | 80.14% |
| Bounded-state reconstruction exact whole answers | 22.14% | 40.36% |

Removing addressed/retained reads lowers minimal copy accuracy to 11.85%. Zeroing final L3 state lowers reconstruction to 7.42%. The reconstruction decoder reads only final L3 state and is included in the fitted budget. This result qualifies bounded-state reconstruction, not semantic summarization or native-language answering.

The original runtime qualification included 48 runtime and 35 research Python tests, optimized/native sanitizer checks, 713 added minimal checks, exact minimal continuation and byte-identical continuation of seven historical policies. EXP01 reran the 48 runtime tests, the current 40 MAD/EXP00/query tests, and the complete native suite; its 1,992 added native checks also pass optimized and ASan/UBSan builds. See the EXP01 report for exact scope and source/log hashes.

The existing short timing comparison does not justify a strong throughput claim. Preserve language, copy and reconstruction as regression fixtures; measure new work with isolated timing.

## Completed EXP00: historical full C

All 48 formal fits and the held-out audit are complete. None of eight development settings crossed the comparative calibration gate.

| Task | Full-C exact episodes | Strongest shortcut | Learned control |
| --- | ---: | ---: | ---: |
| Balanced two-key binding | 50.07% | 50.00% | 100.00% |
| Initialization | 0.00% | 4.30% | 100.00% |
| Set | 0.00% | 0.02% | 100.00% |
| Copy | 0.00% | 0.02% | 100.00% |
| Swap | 0.00% | 5.47% | 100.00% |
| One-hop graph | 4.65% | 28.65% | 100.00% |

Controls use parsed records and independently registered budgets; their scores establish learnability through those adapters. These are full-C measurements, not minimal failures. Retain grouped splits, independent interpreters, counterfactuals and shortcut baselines.

Formal EXP00 cost: 897,000,000 encoded bytes. Development pilots: 301,552,768 bytes. The older 94-fit/31M-byte first-tranche table is a conditional planning estimate, not the actual EXP00 cost or an instruction to repeat the campaign.

## Completed EXP01 initial screen

| Measurement | Minimal | Full C |
| --- | ---: | ---: |
| Native two-answer exactness | 49.45% | 49.51% |
| Single supplied-value copying | 99.35% | 98.70% |
| Factorized strict first-query probe from L3 | 99.93% | 99.22% |

The native endpoint uses full-256-byte autoregressive output. The probe is an
offline, separately supervised diagnostic with predicted query/key/value heads;
it does not qualify native binding. All six native/additive/bilinear arms remain
near the 50% shortcut. No candidate is promoted and no production runtime file
changes. The new native C++ adapter is online, one-pass, and adds no persistent
sequence state or optimizer moments.

There are 36 comparison branches across two explicit cores and three seeds.
Formal charged training is 21M bytes, plus 1.7M pilot and 1.5M copy-control
bytes: **24.2M actual online training bytes**. Offline probe fits are recorded
separately. Compact evidence and citations are in `reports/exp01/`.

## EXP01-B execution contract and remaining qualification

The original diagnosis contract below remains the qualification roadmap. The
first screen covered native rollout, frozen/joint online readouts, separate
representation sources, counterfactual query orders, and training-only probe
normalization. Full write-to-read delay sweeps, all 64 answer offsets, optimizer
confirmation, and language regressions remain pending. Do not label those
completed from the first screen.

1. **Verify architecture and fixture selection.** Use minimal v19 as the production baseline and explicit full C/warmup as the historical comparator. Record actual fitted parameters, state inventory, seeds and source hashes. Start with a single key/value binding and no distractor writes; then use the balanced two-key fixture with distinct values and independent record/query order.
2. **Locate the failing stage.** Measure value availability immediately after writing, after retention, at the query-conditioned read and at native decoding. Probe addressed reads, retained reads and final L3 state separately and in combination. Freeze the same observation state while changing only the query.
3. **Separate representation from readout.** Compare identically fitted raw, centered and standardized probes. Test a query-conditioned low-rank readout against a matched-capacity generic head. First freeze the core; then train end to end. Deduct new readout parameters from expert-bank capacity.
4. **Use native answers as the endpoint.** Report autoregressive whole-episode exactness, full-256-byte versus restricted outputs, query-swap behavior and the strongest shortcut. Keep teacher-forced token accuracy separate. Explicit query-index oracles and symbolic stores remain diagnostic-only.
5. **Control timing and selection.** Check all 64 answer offsets while separating alignment from added delay. Keep equivalent initial-state/query/operation variants in one split. Select on development data, lock the recipe, then audit. Register the diagnostic training budget before fitting; record pilots separately.
6. **Choose the smallest repair.** If a frozen probe retrieves the value but native generation cannot, repair query/readout integration. If the value is unavailable, proceed to EXP02/03 for state capacity, retention or interference. Do not launch a broad optimizer/slot/loop sweep before this distinction is resolved.

The comparative calibration gate remains shortcut-normalized exactness in the declared band, with learned controls and the required seed/sample counts. A diagnostic may explain a failure without clearing that gate; do not present it as a capability promotion.

The existing `experiments/query_readout/` adapter and registered train/audit runner
reuse `experiments/mad_next/` fixtures. Reproduction is documented there. Build
the next candidate with separate query/key selection and value decoding, using
[key-value memory separation](https://aclanthology.org/D16-1147/), plus direct
local credit from already observed constituents. Compare that credit with
answer-only training at matched capacity. Keep parsed record IDs and the
offline probe out of native inference. Require native exactness ≥95%, no seed
below 90%, query counterfactuals, one-pass cost, and the remaining regression
checks before default promotion. Register a new audit fold after these results
have informed the new design.

## Following work

| Order | Packages | Implementation target |
| --- | --- | --- |
| 1 | EXP01-B | Online factorized query/key selection and value readout; then native qualification |
| 2 | EXP02/03 | Independent state capacity; delay, interference, eviction and retrieval |
| 3 | EXP04/05 | Initialization, set/copy/swap; selective erase/write and state correction |
| 4 | EXP06/07/22 | Useful 16/64-byte summaries; suitable objectives; query-time/boundary access |
| 5 | EXP08/09/20 | Multi-hop retrieval, composition, new rules and transfer |
| 6 | EXP10–12 | Revisable workspace, useful extra inference, stopping and verification |
| 7 | EXP15–19 | Native language, instructions, grounding, reasoning, code, tools and uncertainty |
| Support | EXP13/14/21 | Capacity allocation, continual/episodic memory, joint scaling |
| Optional | EXP23 | Modality/environment adapters |

For EXP06, assess upper summary contributions first. Minimal has no upper experts to permute; upper expert restoration needs a separately named candidate and a measured benefit over summaries. For EXP20, minimal expert reuse concerns the 1/4-byte banks. New reasoning or route-cycle candidates must declare a compatible architecture.

## Verification and deliverables for the next implementation

Use the existing relevant runtime/native, causality, checkpoint and MAD checks for changed paths. Validate new derivatives and persistence where math/state changes. Keep historical policies exact. Existing commands include:

```bash
python -m multiscale.tests.run_all
python -m unittest experiments.mad.test_protocol experiments.mad_next.test_exp00 -v
python -m unittest experiments.query_readout.test_adapter -v
python -m experiments.query_readout.check_native
python -m experiments.mad.run --core minimal --preset smoke --execute --out /tmp/minimal-mad-smoke.json
```

The smoke command is a small harness check, not EXP01 qualification. Check the repository's documented sanitizer options for the current host.

Record source/protocol hashes, explicit core/policies, inventories, all training costs, selected settings, per-seed exact/native scores, frozen interventions and isolated timings. Update the capability matrix and backlog only for the tested scope. Publish compact source, results and primary PNGs; keep datasets, compiled binaries and repeated weight snapshots out of git.
