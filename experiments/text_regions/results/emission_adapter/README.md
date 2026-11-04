# Shared emission adapter study

The aggregate comparison covers twelve frozen regional graphs, three adapter
initializations per graph, raw and mixed features, and parameter-matched identity
and GELU heads. Each head is one common 256-byte decoder for all queries.

`comparison.json` contains per-initialization and per-graph results, group means,
fit metadata, and artifact hashes. `manifest.json` identifies the original
checkpoints and reference heads. `verification.json` records the independent
audits; `study_run_log.json` preserves settings and commands.

The downloadable package restores the feature caches, fitted heads and complete
synthetic prediction rows listed in `artifact_inventory.json`. Raw rows remain
outside the public repository. The cached features permit further decoder
experiments without running the experts again.

After extracting the complete package, run from the repository root:

```bash
python -m experiments.text_regions.emission_compare \
  --source artifacts/emission_adapter_v1 \
  --out artifacts/reproduced_emission_comparison.json
```

The audit reconstructs synthetic cases, checks model and source identities,
reproduces all saved predictions from the fitted heads and cached features,
checks exact agreement with the previous linear baselines, and reconstructs
each fitted training objective. No evaluation scores select a head or seed.
