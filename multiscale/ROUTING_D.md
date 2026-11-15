# Routing D selected for main

D is the selected default for fresh models in the public API and CLI. This
promotion uses the frozen D control, including its prerequisite learner and
recall fixes. Native implementation files match that control byte-for-byte.
The experimental context routers and added-depth arms remain outside this release.

## Recorded result

| Seed | FineWeb development NLL after 15M targets |
| ---: | ---: |
| 1 | 1.7796373790853632 |
| 7 | 1.7712335445338496 |
| 19 | 1.7676447414715426 |
| Mean | 1.772838555030252 |

The prior serial comparison measured 30,669.996 training bytes/s median for D
on one core of an AMD EPYC 9V74, using three 200K-target continuations. This
promotion reuses those throughput and training results. It reruns source parity,
fresh 100K-target replay per seed, full development evaluation of all three 15M
checkpoints, and 2K-target checkpoint continuation. The recorded and recomputed
losses, fresh states and continued states agree exactly.

D remains the selected loss/time baseline. In the later extra-depth comparison,
the residual depth arm's mean 15M loss was 1.768953, but it won only one of three
paired seeds and trained 4.1% slower; at equal measured training time it was
approximately tied with D. Those extra-depth changes are not silently included.

## Mechanism

Every level centers routing inputs using a training-only running mean and
normalizes the cue. Prototypes are initialized deterministically from observed
training cues. The gate is `1 + 0.25*tanh(semantic_margin)`; utilization biases
influence selection, but do not saturate this semantic gate. Exact prototype
search is used at all four levels.

Every 256 selections, bias correction is updated. Uniform-use warmup fades as
the least-used expert reaches 256 selections; accumulated bias then decays and
only a below-10%-of-uniform usage deficit receives a weak starvation nudge.

At L1–L3, a fixed ordered code selects the sixteen largest absolute coordinates
from stable child content, preserving signs and values. A detached normalized
previous state supplies a distinct context contribution. Routing adds 0.5 times
the SDR and 0.125 times permuted context; the expert input adds 0.25 times the
SDR and 0.125 times context. A shared 64-by-64 head predicts the next observed
same-scale SDR. Its local loss teaches selected experts and recurrent computation.
Context reset clears pending prediction eligibility, while preserving learned
means, prototypes, predictor weights and utilization history.

The nominal 1M setting retains expert banks 98/49/29/20 and adds three predictor
heads, giving 1,011,903 fitted parameters. Defaults retain separate learning
rates and full constituent credit. Semantic-lace dispatch is disabled in D.

## FineWeb protocol

The corpus is UTF-8 FineWeb `sample-10BT`. It contains 15,000,000 training
next-byte targets from 4,947 documents and 262,144 development targets from
70 disjoint documents. Exact binary hashes, source revision and preprocessing
are recorded in [selection metadata](results/routing_d_selection.json).

Seeds 1/7/19 use the same document-aware 4,096-target chunk permutations as
the recorded comparison. Recurrent memory is cleared at each chunk boundary;
no next-byte pair crosses a document boundary. The first 5M targets use a 50K
warmup and cosine decay from rates fast/query 0.008, slow 0.08 and head 0.16 to
one tenth of those values. After 5M, rates become one quarter of the terminal
rates: fast/query 0.0002, slow 0.002 and head 0.004. Prototype attraction stays
0.002, contrastive auxiliary weight 0.1 and reconstruction weight 0.05.

Evaluation retires pending updates on a clone, freezes the model, and resets
context for each development document. Loss is weighted by next-byte targets.
The training checkpoint itself is not modified by evaluation.

## Reproduce promotion verification

The recovered experiment package supplies `baseline_v9/model.py`, its native
headers, the manifest and binary splits, and D checkpoint/result pairs. With
those files outside the repository:

```bash
python -m multiscale.tests.run_all
python -m multiscale.tests.verify_routing_d \
  --historical-model /path/to/baseline_v9/model.py \
  --data-dir /path/to/data \
  --checkpoints /path/to/fineweb_runs \
  --out /tmp/routing_d_verification.json
```

On ptrace-managed hosts, use `--no-leak-check` for the regression suite. This
explicitly disables LeakSanitizer, whose process inspection is unsupported
there; AddressSanitizer and UndefinedBehaviorSanitizer remain active.
GitHub CI runs the complete suite with LeakSanitizer enabled.

Historical versions 1–4 use preserved native profiles. Versions 5–9 are loaded
with saved learning/routing semantics; v8 pilot balancing is not migrated to
v9. The C++ constructor retains its internal historical construction policy
for these loaders and native tests. Fresh public models select D explicitly
through the Python wrapper before observing any data.
