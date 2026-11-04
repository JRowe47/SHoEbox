# SHoEbox: balanced source availability

2026-11-04 | https://github.com/JRowe47/SHoEbox

Balanced source exposure delivers nearly perfect B-only shared recovery, while reducing held-out all-view private-answer accuracy under both publication policies. The schedule remains optional and the default is unchanged.

## Three-seed comparison

| Accuracy | Original policy: dropout | Original policy: balanced | Gated policy: dropout | Gated policy: balanced |
|---|---:|---:|---:|---:|
| Held-out overall | 78.2% | 61.5% | 68.2% | 63.4% |
| Held-out shared | 83.5% | 84.4% | 88.7% | 91.1% |
| Held-out private, pooled | 75.6% | 50.1% | 57.9% | 49.5% |
| Held-out left private | 76.8% | 49.6% | 53.8% | 51.6% |
| Held-out right private | 74.3% | 50.6% | 62.0% | 47.5% |
| B-only shared | 35.1% | 99.8% | 85.1% | 96.5% |
| A-only left private | 72.7% | 75.3% | 65.7% | 76.8% |
| C-only right private | 72.7% | 71.7% | 63.1% | 79.2% |

Values are unweighted means across fine-tuning seeds 20261106, 20261107 and 20261108. Each policy has three matched schedule pairs. All six new models retain the original baseline loss, with both gradient detachment and observed-only masking disabled. View objectives are on; utility updates are off.

| Policy / seed | Overall change, pp | Private change, pp | B-only shared change, pp |
|---|---:|---:|---:|
| original / 20261106 | -20.3 | -30.7 | +74.8 |
| original / 20261107 | -11.8 | -13.6 | +50.0 |
| original / 20261108 | -17.9 | -32.2 | +69.3 |
| gated / 20261106 | +0.8 | -1.6 | -1.0 |
| gated / 20261107 | -12.5 | -19.6 | +35.2 |
| gated / 20261108 | -2.7 | -3.9 | +0.2 |

Changes are balanced minus dropout. Each evaluation uses 512 fresh episodes per query, 1,536 per control, and an unrestricted 256-byte output vocabulary.

The exposure schedule removes most of the original publication policy's B-only deficit: mean recovery reaches 99.8%. It sacrifices all-view performance, with private accuracy falling to about 50% under both policies. More frequent missing-source practice alone does not satisfy both capability requirements at this budget.

---

## What changed

| Query | Allowed retained sources | Episodes per subset |
|---|---|---|
| Shared | A, B, C, AB, AC, BC, ABC | 180 or 184 |
| Left private | A, AB, AC, ABC | 320 |
| Right private | C, AC, BC, ABC | 320 |

The schedule rotates complete source descriptions and always preserves a source containing the answer. Each group of four answer values shares one subset; another rotation across the 32-episode tuple cycle prevents a fact tuple from being tied to one subset. Every subset is exactly answer-balanced. Each shared tuple appears under every supported subset 5-6 times; each private tuple does so 10 times. These counts cover 1,280 episodes per query.

Identifiers, facts, templates, byte inputs and query positions are identical to the paired dropout streams. Only availability changes. Excluded sources stay absent for the whole episode; graph nodes continue their normal expert executions. Selection uses the requested query category and episode ordinal, never its answer value. Loss targets and audit counters stay outside model inputs and carried inference state.

## Exposure and auxiliary supervision

| Mean per training run | Original dropout | Balanced |
|---|---:|---:|
| Source byte occurrences | 238,438.3 | 171,807.3 |
| Shared-query B-only episodes | 54.3 | 182.7 |
| Pooled private targets: A | 6,948.7 | 5,304.0 |
| Pooled private targets: C | 6,941.3 | 5,306.7 |
| Joint A/C shared samples | 6,318.7 | 3,296.0 |
| Baseline eligible targets | 17,371.7 | 13,260.0 |

The two publication policies have identical exposure counts within each seed and schedule. Counts deliberately differ across schedules. Private counts refer to eligible pooled labels per actual training chunk; shared counts refer to jointly available A/C samples. Loss weights and normalization rules remain fixed. The comparator reconstructs these counts from source events, fact positions and resets; all 240 new segments are audited. Legacy control counts are reconstructed and explicitly marked as such.

## Matched compute and evaluation

Every run trains for **240 optimizer steps, 96,000 aligned external events and 8,064,000 expert executions**, with batch 32, segment 16 and K=1. All arms share a correction-trained graph initializer and matched auxiliary tensors within seed. The model has 265,694 allocated parameters, including a 246,142-parameter graph with seven nodes, three regions and 384 physical experts. These are three fine-tuning seeds from one graph initializer.

Test/all holds out 16 fact combinations. Independent controls exhaust all 64 combinations with balanced answers: all views, A only, B only, C only, private-value withholding and reset-before-query. Each control uses 38,400 events and 3,225,600 expert executions. Source-readout fits use separate training descriptions. Host-contended wall times are not throughput comparisons.

---

## Next development step

Freeze the original and balanced graphs and fit one common byte decoder at the final query across all query types and supported source combinations. Compare the raw 64-D P/Q concatenation with the current 32-D mixed/normalized latent using identical training cases. Query labels must neither select heads nor enter features. Separately fit fixed factor readouts at the description boundary and final query to measure changes in fact decodability. Keep expert weights unchanged and K=1.

## Reproduce and inspect

Extract SHoEbox_source_schedule_checkpoints.zip at the repository root. It restores the common initializer, six balanced models, six matched controls and twelve source readouts under their artifact paths. Install requirements.txt, then run:

```bash
python -m experiments.text_regions.view_train --steps 240 \
  --batch 32 --segment 16 --gap 8 --dropout 0 \
  --source-schedule balanced --seed 20261106 --threads 2 \
  --base-checkpoint artifacts/probes/correction_warm150/checkpoint.pt \
  --view-objective --evidence-gated-ports \
  --out artifacts/reproduction/balanced_gated_seed1

python -m experiments.text_regions.view_score \
  artifacts/reproduction/balanced_gated_seed1/checkpoint.pt \
  --publication-policy evidence --count-per-query 512 \
  --batch 64 --threads 2 --predictions \
  --out artifacts/reproduction/balanced_gated_seed1/evaluation

python -m experiments.text_regions.source_schedule_compare \
  --manifest experiments/text_regions/results/source_schedule/manifest.json \
  --out artifacts/reproduction/source_schedule_comparison.json

python scripts/check_all.py
```

Repeat with seeds 20261107 and 20261108. For original publication, omit --evidence-gated-ports and score with --publication-policy original. The matched control uses --source-schedule dropout --dropout 0.25. Use a separate output directory for each run.

**All 245 automated tests passed.** GitHub CI also passed for the executable-code commit. Tests verify answer/subset/tuple coverage, unchanged rendered content, legacy dropout fingerprints, exact balanced and legacy checkpoint continuation, actual loss eligibility, fixed expert budgets, and strict comparison rejection cases. The default training path also matches the prior implementation bit for bit through two batch-32 optimizer steps, including all 691 gradient entries, logits, losses, weights, optimizer, state and RNG. All three earlier published comparisons reproduce exactly.

Raw protocols, histories, predictions, source provenance, exposure audits and the test transcript are under experiments/text_regions/results/source_schedule. The portable manifest references earlier control results without duplication. Every training/evaluation invocation records exact runtime source hashes matching the tested code at ced8db47063c576680cb1409db607229e93adcc2. The source archive includes all referenced results and this report; tensors are in the separate checkpoint bundle.

Checkpoint archive SHA-256:

```text
81ba76516ed93b61fe973099305286cbaf4d9f4a1d429e1198160dff5e55b737
```

Development starts from SHoEbox revision f50cae0126cc1585518932a3f0c9f979b0d47ea4. The inherited SHoE revision is b4bb7cfea18c3833819f6ed08973fd014f15f107.
