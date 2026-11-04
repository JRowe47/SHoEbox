# SHoE semantic-lace nanoGPT demo

> Public release: run `python scripts/prepare_data.py` from the repository root before training. Recorded metrics and samples are included; checkpoints are created by training and are excluded from git. Commands below run from this experiment directory.

A runnable three-layer character language model implementing **semantic groups → cross-layer lace relationships → context-resolved experts and attention keys**.

## Default shape

| Component | Configuration |
|---|---|
| Trainable parameters | 1,000,732 |
| Transformer / SHoE blocks | 3 |
| Experts, total | 1,024: 342 / 341 / 341 |
| Active experts | 16 per layer, 48 total expert executions per token |
| Width / heads | 100 / 4 |
| Individual expert | 100 → 4 → 100 GELU MLP |
| Semantic groups | 32 independently organized groups per layer |
| Selected groups | 2 per layer; 8 experts selected inside each |
| Address dimension | 16 |
| Learned lace factors | 64; 2 selected per target layer per routing round |
| Routing refinement | 2 rounds of bidirectional controller messages |
| Attention reads | 16 local keys + up to 8 older keys from each selected group |
| Context / vocabulary | 128 characters / 65 characters |
| Embedding and output weights | Tied |

The 1,024 count is **across the complete model**, not per layer. These are deliberately tiny experts: 16 selected four-unit specialists give 64 active hidden units per block. There are 819,200 expert matrix parameters, of which 38,400 participate for one token across all three layers (4.6875% of the bank). This does not count shared computation.

## Run

Python 3.10+ and PyTorch 2.5+; the bundled run used Python 3.12 and PyTorch 2.15.0+cpu. Run from this directory:

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_model
python model.py
python train.py --steps 600 --threads 2
python generate.py --checkpoint results/demo/checkpoint.pt --prompt "ROMEO:" --tokens 300
```

Matched training with the relationship controller disabled:

```bash
python train.py --steps 600 --threads 2 --no-lace --out results/no_lace
```

`--steps` counts optimizer steps, with 512 character tokens per default step. The first training command overwrites its output directory's run files; use `--out` to preserve a prior run. Checkpoints contain model weights, optimizer state, configuration, vocabulary, RNG states and step count. Generation loads the checkpoint with `weights_only=True`. The trainer starts new runs; continuation can use the checkpoint state but is not exposed as a CLI option in this demo.

## How the addressing works

1. **Stable, independent group identities.** Each layer learns 32 prototype vectors. Expert IDs are independently shuffled into balanced groups of 10–11 members. An expert's address is the normalized sum of its group's prototype and a learned bounded residual. Expert membership is fixed in this first demo; the geometry and the computation specialize during learning. No group-ID alignment is assumed across layers.
2. **Context proposes local interpretations.** The current embedding plus a causal prefix mean is projected into each layer's address space. These three provisional group distributions are the initial evidence supplied to the controller. The prefix mean is a cheap causal contextual feature, not a recurrent long-term memory.
3. **Laces express joint group relationships.** A lace is a learned factor with a categorical group profile at each of the three layers. The factor bank does not enumerate all 32³ group combinations. A contextual query and evidence from the other layers select two factors for each target layer. Those factors send a distribution over its candidate groups.
4. **Two rounds coordinate routing.** Evidence from the receiving layer is excluded from its incoming factor score. This lets one layer's evidence change another's prior. Incoming messages are refined twice before the three transformer blocks execute. This is feedback in the lightweight routing controller; later expert outputs are not recurrently replayed into earlier blocks.
5. **Local state binds the address.** Each actual transformer layer combines its current causal state with the incoming group prior, selects two groups, and scores only the 10–11 expert members of each. Eight members per group execute. Group and member gates remain differentiable with respect to their selected-score paths.
6. **The same groups address token memory.** Current token keys are assigned to their nearest layer prototype. Per-group posting lists contain their sequence positions. A query reads at most eight preceding keys per selected group outside its local window, then performs exact QK attention only on those keys plus its local window. Cumulative prefix counts enforce causality, even though posting lists are constructed in parallel for the training chunk.

The concrete address is therefore **(layer, selected group, context-selected expert ID / token position)**. Expert keys are persistent learned addresses; token keys are contextual representations of current sequence positions. They share prototype anchors but remain distinct objects.

### Controller equation

Let A[m,l,g] be a lace's softmax-normalized group profile and p[l,g] a layer's current evidence. For a message addressed to layer l:

```
support[m,j] = log(sum_g p[j,g] * A[m,j,g])
score[m -> l] = context_score[m] + sum_{j != l} support[m,j]
selected = top2_m(score[m -> l])
prior[l,g] = sum_{m in selected} softmax(score)[m] * A[m,l,g]
new_evidence[l,g] = initial_evidence[l,g] + 0.75 * centered_log(prior[l,g])
```

The learned factors provide cross-layer correspondences. The receiving layer uses its own prototype geometry, so no global x/y coordinate system or common group numbering is required.

## What is sparse

* Only 16 expert MLPs per layer execute, selected after scoring two group neighborhoods.
* Expert lookup scores 32 prototypes, then at most 22 expert addresses; it does not scan all 341–342 expert addresses.
* Attention allocates 32 score slots per query, of which some are masked when insufficient history exists. No T×T attention matrix or causal mask is created in the model.
* Shared QKV/output projections, residuals, norms, embedding/head, group scoring, and the small lace controller are dense.
* Posting-list construction uses O(BTG) counts and storage, and lookup uses a bounded number of reads. G=32 is fixed here. This is a stable-prototype index, not the LSA paper's Ward clustering or growing tree index.
* PyTorch gathers only selected expert weights for arithmetic, but backward allocates dense bank gradients and AdamW maintains/updates dense optimizer state. Weight gathering and invalid attention slots also have real overhead. This implementation establishes the mechanism; it does not claim a speedup over dense nanoGPT.

## Training and evaluation

Tiny Shakespeare is prepared with the root data script. Original dataset source: https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt . `protocol.json` records the exact SHA-256. The contiguous split is 90% training, 5% validation, 5% untouched test. The vocabulary is derived only from the training split. Losses are character cross-entropy in nats, not directly comparable with the earlier GPT-2-tokenized 124M run.

Training uses AdamW (β₁=.9, β₂=.95), peak LR .002, 40-step warmup, a plateau through 80% of the run, then linear cooldown to .0004. Gradient norm is clipped at 1. The loss adds .01 times group-load balancing, key/prototype alignment and lace-load balancing terms. Hard top-k choices and token bucket assignments are discrete; gradients flow through gates, attention scores, selected template mixtures and auxiliary probabilities.

Periodic validation uses fixed 2,048-character probes. Final validation and train probes use 8,192 characters each. All final validation variants use the same windows. Disabling laces and scrambling their middle-layer endpoints on a frozen trained model measure dependence on the controller. The separately trained `no_lace` run uses the same seed and minibatches and is the relevant small matched comparison. It retains the dormant controller parameters, but executes fewer operations. Timing comparisons are not isolated hardware benchmarks when the two runs execute concurrently.

## Verification

`test_model.py` checks exact parameter/expert counts; complete balanced expert membership; task-loss and total-loss gradients; suffix and prefix causality; causal posting-list lookup against an independent loop; sparse attention against a masked dense reference; expert arithmetic against individual MLP execution; zero task gradients to unused expert matrices; cross-layer interventions; context-sensitive routing for the same current token; ablation isolation; and checkpoint round-trip identity.

## Files

* `model.py`: model, semantic index, sparse expert banks and lace controller.
* `train.py`: reproducible trainer, validation, routing diagnostics and checkpoints.
* `generate.py`: load a checkpoint and generate text.
* `test_model.py`: correctness tests.
* `results/demo/`: lace-enabled raw metrics, protocol and sample.
* `results/no_lace/`: matched controller-disabled run.
* `test_model.py`: runnable verification suite.
* [Recorded results](../../docs/results.md): experiment comparisons and interpretation.

Conceptual source: [Latent Sparse Attention, arXiv:2610.42147](https://arxiv.org/abs/2610.42147), especially semantic clustering/search in sections 3.1–3.2. This code is a new readable reference implementation of the conversation's stable-anchor cross-layer lace extension; it does not import the earlier MFS native extension or claim to reproduce the paper's benchmarks.
