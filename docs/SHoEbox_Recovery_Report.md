# SHoEbox publication and recovery

November 4, 2026 - Mountain Time

## What is available

The latest cohort-diversity report is preserved. The preceding shared-preservation implementation and its checkpoints are also preserved and packaged. The cohort-diversity implementation and its 432 fitted heads were not recovered.

| Material | Recovery status |
|---|---|
| SHoEbox source through shared preservation | Available, with a small audit-portability fix |
| Twelve frozen regional graphs | Available |
| Earlier readouts, emission features, residual controls | Available |
| 144 shared-preservation fits and four longer-fit checks | Available |
| Cohort-diversity report | Original PDF and previously reconstructed Markdown available |
| Cohort-diversity source, 432 fits, expanded caches and full comparison JSON | Not recovered |

The source ZIP contains the published code and reports. The recovered-artifacts ZIP adds 112 tensor files, cached features, validation locks, synthetic predictions, and the dependencies needed to reproduce the preservation study. Its file manifest records every entry's size and SHA-256. Neither ZIP claims to restore the later cohort implementation.

## Preserved cohort findings

The completed study expanded decoder fitting from 960 to 3,840 cases across twelve frozen graphs. It compared identity and GELU residual corrections, penalties 0, 1 and 10, and three initializations: 432 fits in the retained report. The original execution record reports 305 passing tests. Those counts describe the earlier completed run; they are not newly rerun cohort checks.

The selected identity correction improved both private and shared accuracy in three of four publication/scheduling conditions. Validation accepted nonzero identity corrections in 33 of 36 cases, versus 3 of 36 with the smaller cohort.

| Identity correction | Private: small to large | Shared: small to large |
|---|---:|---:|
| Original / dropout | 73.177% to 77.257% | 87.240% to 88.108% |
| Original / balanced | 55.990% to 57.986% | 80.469% to 82.812% |
| Gated / dropout | 60.938% to 59.809% | 91.146% to 90.104% |
| Gated / balanced | 54.948% to 56.641% | 87.500% to 88.976% |

GELU improved private accuracy in all four conditions but reduced shared accuracy in three. The original report, full result tables and methodology remain in docs/SHoEbox_Cohort_Diversity_Report.md and its PDF. The default model/readout is unchanged.

## Recovery verification

The retained baseline ZIP is byte-identical to the published shared-preservation package. All 740 entries in its original manifest were checked by size and SHA-256 before restoring the artifacts.

An audit portability defect surfaced during reproduction: rebuilding float64 normalization and losses produced differences around 1e-16, which exact numeric comparisons rejected. The fix permits 1e-12 absolute and relative tolerance for reconstructed normalization and three named loss fields only. File and tensor hashes, prediction bytes, correct counts, selected strengths and penalties, and eligibility decisions remain exact. Regression checks reject changed decisions, significant loss differences and nonfinite losses.

Fresh verification results are recorded in experiments/text_regions/results/recovery/verification.json and its logs. These checks concern the preserved study and the recovery changes, not the missing cohort artifacts.

## Reproduce the preserved study

Extract SHoEbox_recovered_artifacts.zip and run from its SHoEbox directory:

```bash
python -m pip install -r requirements.txt
python scripts/check_all.py
python experiments/hat_minkron/example_live.py
python -m experiments.text_regions.preservation_compare \
  --source artifacts/shared_preservation_v1 \
  --convergence artifacts/preservation_convergence_v1 \
  --out artifacts/reproduced_preservation_comparison.json
```

RESTORE_RECOVERY.md and PACKAGE_MANIFEST.json are included at the archive root. The source-only ZIP runs the automated tests; the artifact audit additionally requires the recovered-artifacts ZIP.

## Exact recovery boundary

The last fully retained experimental source revision is faed544b37000fcdb11b5584396c7856b85a4be5. The later report was published at afbc7b3eb5d6d431739242da671480a0341f4ecf. This recovery adds packaging, documentation and audit portability; it does not reconstruct experimental tensors or synthesize substitute results.

The unavailable cohort source revision is c4be984e247f738a63433c7a84d4aa7a2c5918ef, with tree d9b7759444790f050fbf0c1126c15e54b9eb5c6b. Neither object was found in the retained repository or GitHub. The recorded cohort ZIPs were also absent from the retained workspace and saved-file inventory. Their earlier hashes and sizes remain in experiments/text_regions/results/cohort_diversity/recovery.json for future identification.

The next development step remains a full-rank linear byte decoder fitted on the expanded cohort, compared with identity/GELU residuals and measured for inference cost. Rebuilding the missing cohort tooling and rerunning its fits is required before that comparison can reuse the later experiment.
