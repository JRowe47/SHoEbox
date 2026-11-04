# Three-SHoE hierarchy with shared experts and upward semantic reinforcement

> Public release: run `python scripts/prepare_data.py` from the repository root before training. Recorded metrics and samples are included; checkpoints are created by training and are excluded from git. Commands below run from this experiment directory.

A semantic-only parent SHoE connects to two expert SHoEs. Each child retains
the tested 1M model's learnable modules and tensor shapes: three layers, width
100, four attention heads, 1,024 expert MLPs, 32 groups per layer, 16-dimensional
addresses, and 64 lace factors. Each token recruits sixteen experts per layer.
The parent has three semantic axes and 64 laces, and no expert bank.

## Exact parameter accounting

| Item | Count |
|---|---:|
| Logical parameters in each child | 1,000,732 |
| Expert weights in each child | 819,200 |
| Shared experts | 512 |
| Dedicated experts per child | 512 |
| Unique physical experts | 1,536 |
| Shared expert matrix parameters | 409,600 |
| Dedicated expert matrix parameters per child | 409,600 |
| Unique expert matrix parameters | 1,228,800 |
| Both children, counting shared weights once | 1,591,864 |
| Parent plus input interface | 44,920 |
| Complete unique parameter count | 1,636,784 |

The original 1M model includes attention, embeddings and routers. Thus two
original-shape children sharing half of their *experts* produce a 1.637M
complete hierarchy, with 1.229M unique expert matrix weights.

Shared counts across the three layers are 171 / 171 / 170. Each child's total
counts remain 342 / 341 / 341; its private counts are 171 / 170 / 171.
Sharing is actual Python module/Parameter identity, preserved after checkpoint
loading. AdamW receives each physical parameter exactly once. Child-specific
group memberships, prototypes and expert addresses remain independently learned.

## Continuous module interface and causal dispatch

`forward(tokens, targets)` performs character encoding only at the external
input boundary. It then calls `forward_payload(payload, positions, targets)`.
The latter accepts continuous `[batch, time, 100]` messages, allowing another
module to supply the input directly. Each child's vocabulary embedding table
serves as its output decoder in this mode; no intermediate tokenization occurs.
The original standalone child token API remains available through inheritance.

The parent selects one child for every source token. Each child executes its
assigned tokens as an ordered subsequence, preserving their original position
IDs. Its local attention and experts see only that assigned stream. The compact
source header also contains a causal prefix mean over the complete source
stream. The hierarchy therefore changes the context graph relative to two
independent full-context language models while retaining the original child
parameter shapes and internal mechanisms.

Only the selected child executes a token. No attention or expert computation
occurs during `route_only`. Per-token expert work is 48 MLP activations across
three layers, touching 38,400 matrix parameters: 3.125% of the unique expert
bank. Shared projections, controller scoring, and PyTorch bank gradients are
dense. Generation currently recomputes its context rather than caching K/V.

## Multilevel semantic routing

1. **Parent interpretation.** The parent forms group evidence on three local
   semantic axes. Its lace controller combines those axes into a group prior.
2. **Established route or discovery.** A route can use one child probe when
   confidence exceeds 0.75, its past gain is positive, sufficient eligibility
   mass has accumulated, and 80 feedback updates have occurred. Otherwise both
   child controllers are probed. A low local match rejects a shortcut and
   triggers the other probe.
3. **Child bids.** A child runs its pilot group scorers and lace controller on
   the incoming header. It returns a local endpoint distribution and match
   score. It runs neither attention nor experts.
4. **Upward interpretation.** A learned relation tensor of shape
   `2 × 3 × 32 × 32` maps each parent's group axis into the corresponding local
   child group axis. Child endpoint beliefs send reverse compatibility messages
   through this tensor. The parent refines its own lace interpretation.
5. **Selection and downward message.** The parent combines semantic agreement,
   context, predicted benefit and learned route value to choose one child. Its
   translated prior biases the selected child's lace controller before expert
   selection.
6. **Execution.** Each child runs its three original-style blocks on only its
   assigned causal subsequence. Outputs are scattered back to source order.

Hard top-1 selection uses a selected-gate straight-through task gradient.
During training, 10% of tokens explore a uniformly chosen child after probing
both. Inference selects deterministically. A parent load-balancing penalty is
applied to its unmasked preference probabilities; it remains trainable even
when an established path probes only one child. The discovery warmup, exploration
and balancing were added after an initial ten-update smoke run showed early
one-child lock-in.

## How executed lower laces teach the parent

Three complementary signals are implemented:

**Executed-lace semantic targets.** The child collects group scores from the
three blocks that actually ran. A cheap final lace-controller pass turns them
into an executed endpoint posterior. That detached posterior teaches the
parent's translated endpoint prediction through KL loss. The loss is weighted
more strongly when the child's prediction improved on the parent's cheap
baseline. Task gradients also train the live bid and downward-message pathways.

**Outcome calibration.** A lightweight parent head predicts each child's gain.
The realized target is the parent's cheap next-character cross-entropy minus
the executed child's cross-entropy, clipped to [-2, 2]. Only the selected child
receives an outcome target. Smooth-L1 loss trains the gain estimate.

**Next-pass route memory.** The parent endpoint activity provides an eligibility
trace for the selected connection. After backpropagation and the optimizer step,
`commit_feedback` updates endpoint-to-child running value sums and masses with
decay 0.97. Positive and negative outcomes are retained. The next forward pass
can use these values to favor established useful paths.

Forward calls never update this memory, even when labels are supplied.
Evaluation never commits feedback. Labels participate only in losses and the
subsequent training update, so predictions cannot read their targets. There is
no explicit replay buffer, replay pass, or unexecuted-expert outcome estimate.

Loss coefficients: base next-character CE + 0.01 original-style routing auxiliary
loss + 0.05 cheap parent-baseline CE + 0.05 parent load-balancing penalty;
the full variant additionally uses 0.03 executed-lace KL and 0.03 utility loss.

## Controlled training and interventions

`results/full` trains all three feedback mechanisms. `results/no_reinforcement`
retains the same architecture, initialization, live bottom-up bids and top-down
semantic signaling, but disables explicit semantic/outcome reinforcement losses,
route-memory commits and fast-path reuse. This tests that reinforcement bundle;
it is not a no-lace model and does not isolate its individual components.

Both runs start from scratch with the same seed, minibatches, and 600 × 4 × 128
= 307,200 character presentations. AdamW uses peak LR 0.002, beta (0.9, 0.95),
weight decay 0.01, norm clipping 1, a 40-step warmup, plateau to step 480, and
linear cooldown to 0.0004. Parameter and minibatch hashes verify the match.

Tiny Shakespeare uses the original contiguous 90% train / 5% validation / 5%
untouched-test split. Periodic validation uses 2,048 character presentations;
the final comparison and frozen interventions use the same fixed sampled
8,192-character probe. Windows can overlap. Training is one seed per variant;
the test split remains unused. CPU runs may overlap, so elapsed timings describe
this execution rather than an isolated architecture benchmark.

Frozen interventions force broad discovery, disable the upward messages,
scramble the observed child endpoint coordinates, or disable learned route
memory/shortcut reuse. They measure dependence without retraining.

## Run and inspect

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_hierarchy
python train.py --steps 600 --out results/full
python train.py --steps 600 --no-reinforcement --out results/no_reinforcement
python generate.py --checkpoint results/full/checkpoint.pt --prompt 'ROMEO:'
```

The fourteen hierarchy tests cover exact counts and aliasing, expert arithmetic,
probe isolation, one-child execution, shared updates/private isolation, causal
prefixes and original positions, target isolation, upward gradients and
interventions, signed delayed feedback, shortcut rejection/discovery, continuous
payload equivalence, and checkpoint identity.

Checkpoints preserve weights, route memory, optimizer state, vocabulary,
configuration and RNG states. The trainer currently starts a fresh run; a resume
CLI is not provided. `semantic_core.py` is an unmodified copy of the tested 1M
implementation; its source hash and lineage are recorded in `PROVENANCE.json`.
