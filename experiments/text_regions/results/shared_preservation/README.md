# Shared-answer preservation during residual training

Adds a training-only squared margin shortfall penalty on correctly answered
shared examples. Two fixed penalty coefficients (1 and 10), identity/GELU
controls, and three initializations give 144 fits across twelve frozen graphs.
Validation chooses penalty and residual scale under shared-answer accuracy guards.
Four longer-fit diagnostics were chosen only by training gradients.

`comparison.json` contains all aggregate and per-head metrics, training diagnostics,
and validation choices. `manifest.json` identifies prerequisite files. The complete
ZIP supplies weights, features, and raw predictions outside the public repository.

```bash
python -m experiments.text_regions.preservation_compare \
  --source artifacts/shared_preservation_v1 \
  --convergence artifacts/preservation_convergence_v1 \
  --out artifacts/reproduced_preservation_comparison.json
```
