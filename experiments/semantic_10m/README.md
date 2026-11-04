# Semantic SHoE: 10M scaling experiment

> Public release: run `python scripts/prepare_data.py` from the repository root before training. Recorded metrics and samples are included; checkpoints are created by training and are excluded from git. Commands below run from this experiment directory.

This scales the November 3, 2026 semantic-lace 1M experiment to a wider,
six-layer SHoE while preserving its training exposure and sparse selection
budgets. See [recorded results](../../docs/results.md) for measured results, and `PROVENANCE.json` for the
exact parent archive and changes. The original 1M metrics are in `../semantic_1m/results`.

## Architecture

| Component | Original 1M | Scaled 10M |
|---|---:|---:|
| Trainable parameters | 1,000,732 | 10,027,388 |
| Transformer / semantic layers | 3 | 6 |
| Residual width | 100 | 180 |
| Attention heads | 4 | 6 |
| Total experts | 1,024 | 4,096 |
| Experts per layer | 342 / 341 / 341 | 683 / 683 / 683 / 683 / 682 / 682 |
| Expert hidden width | 4 | 6 |
| Semantic groups per layer | 32 | 64 |
| Address dimensions | 16 | 32 |
| Learned lace factors | 64 | 256 |
| Selected groups per token / layer | 2 | 2 |
| Selected experts per token / layer | 16 | 16 |
| Selected lace factors per token / target layer | 2 | 2 |
| Lace refinement rounds | 2 | 2 |
| Context length | 128 | 128 |
| Attention slots per query | 16 local + 16 remote | 16 local + 16 remote |

The scaled bank has 8,847,360 expert matrix parameters. A token computes with
207,360 of those across its six layers (2.34375%). The 1M bank used 38,400 of
819,200 (4.6875%). These are expert-weight fractions, not fractions of total
model FLOPs or total shared computation.

Each layer organizes its experts independently into balanced stable groups.
Learned prototypes represent group geometry; residual expert addresses resolve
individual experts within selected groups. Each selected group provides eight
experts, giving sixteen tiny 180 → 6 → 180 GELU specialists per layer.

The lace controller stores a 256 × 6 × 64 tensor of group-relation logits.
It receives per-layer pilot evidence derived from the embedding and a causal
prefix mean. For a target layer, each refinement round combines the other
five layers' evidence with a learned context query. Two selected lace factors
send a soft group prior to that layer. The six transformer blocks then execute
in sequence, resolving the priors against their actual local hidden states.
This preserves the original sum of other-layer log support without introducing
a new depth-normalization rule.

Controller feedback is between pilot routing beliefs; completed expert outputs
propagate forward and do not rerun earlier blocks. The learned groups have no
manual semantic labels. Increasing lace count from 64 to 256 increases the
learned factor bank, not an explicit enumeration of all possible group paths.

Each attention key has a group assignment. Causal posting lists retrieve up to
eight older keys from each of two selected groups, excluding a sixteen-token
local window. Attention uses exact QK products plus address and group scores on
these selected keys. It creates no dense token-by-token attention matrix.

Shared QKV/output projections, embeddings, layer norms, group scoring and
AdamW state remain dense. Selected expert weights are gathered for arithmetic;
PyTorch's backward pass still creates dense bank gradients. This is the same
unfused reference approach as the 1M experiment.

## Controlled experiment

Two new 10M models start from identical parameter tensors and see identical
minibatches: one uses laces, one disables them throughout training and
evaluation. Their initial-parameter and input-stream SHA-256 hashes are saved
and checked. The no-lace model retains the same allocated architecture;
disabled controller parameters do not contribute to its task computation.

Both train for 600 steps × 4 sequences × 128 characters = 307,200 character
presentations. This matches the original 1M exposure and data sampling, not a
fixed compute budget. Tiny Shakespeare uses a contiguous 90% training / 5%
validation / 5% untouched test split. Vocabulary comes from the training split.

Optimizer: AdamW, betas (0.9, 0.95), weight decay 0.01, clipping norm 1.
Learning rate warms to 0.002 over 40 steps, stays flat to step 480, then declines
linearly to 0.0004 by step 600. The auxiliary coefficient is 0.01.

Periodic validation uses 2,048 character presentations; the final comparisons
all use the same 8,192 presentations (seed 333). Windows are sampled and can
overlap. The final training probe uses a separate fixed seed 334. The test
partition remains untouched. Losses are natural-log cross-entropy per character.

Frozen-model diagnostics disable laces or scramble the fourth layer's lace
endpoints (zero-based layer 3). These diagnose learned dependence; the
separately trained no-lace run is the matched training comparison.

`probe_endpoints.py` extends the frozen-model diagnostic by permuting each
layer individually and then independently permuting all six layers. It uses
the same final evaluation windows and verifies exact parameter restoration
afterward. Its results are saved as `results/full/endpoint_probes.json` and
summarized in the repository results guide.

The runs execute concurrently with two CPU threads each. Reported training
times therefore include contention. Isolated timing results, when present in
`performance.json`, measure both architectures in one fresh sequential process.
The experiment uses one shared training seed, as in the parent experiment.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_model test_scaled
python train.py --scale 10m --steps 600 --threads 2 --eval-every 50 --out results/full
python train.py --scale 10m --steps 600 --threads 2 --eval-every 50 --no-lace --out results/no_lace
python generate.py --checkpoint results/full/checkpoint.pt --prompt 'ROMEO:'
```

Training writes `checkpoint.pt`, including model, optimizer, configuration, vocabulary and RNG state. Checkpoints are generated locally and excluded from git. The trainer starts a new run; it does not implement a resume flag.

`Config()` retains the 1M defaults for compatibility. `Config.scaled_10m()`
constructs the scaled configuration; the training CLI defaults to `--scale 10m`.
The generation CLI reads the checkpoint's configuration automatically.

`test_model.py` retains the ten original correctness tests. `test_scaled.py`
adds exact counts / memberships, six-layer causality, task gradients through
all six lace endpoints, generalized middle-layer scrambling, full-depth
diagnostics, and checkpoint identity. All sixteen tests must pass.

Raw histories, protocols, summaries, routing counts and unedited samples are included. After reproducing the full run, `python probe_endpoints.py` reruns the endpoint diagnostic.
