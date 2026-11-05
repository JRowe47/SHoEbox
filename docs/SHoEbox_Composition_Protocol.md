# Regional composition protocol

This phase trains the seven-node network itself to combine relations across regions. It closes the decoder investigation with an optional common affine byte head over full P/Q publications. The existing mixed readout remains the default for older experiments and checkpoints.

## Task and information access

The task is `g(f(x))`. B receives the four-byte table for permutation `f`; C receives the table for permutation `g`. Keys and values are the four bytes A-D. Tables list their values in A/B/C/D key order. A receives the query byte `x` after the tables have been encoded. B and C receive a constant `?` read cue at query time; neither receives `x`.

During the four prefix events, only B and C execute, independently, with regional feedback disabled. All seven nodes execute on each query round. P and Q have no previous valid publication before that query. First-round P can combine the query with B; first-round Q can combine B and C but has no access to the query. The next round makes P/Q exchange and descending feedback available.

The model receives bytes and ordinary graph state. Answers and intermediate lookup values are used only by the data generator and loss/scoring code. There is no symbolic solver, per-task output head, retrieval cache or replay store in inference. The decoder is one affine projection from the 64 concatenated P/Q values to all 256 byte logits. No output restriction to A-D is applied during scoring.

## Split and learning

There are 24 possible permutation tables per source and 576 table pairs. A fixed SHA-256 rule assigns whole pairs to train, validation or test. All four queries for a pair stay together. Every individual table appears in every split, so held-out cases require combining familiar relations in new pairs.

| Split | Table pairs | Unique queries |
| --- | ---: | ---: |
| Train | 296 | 1,184 |
| Validation | 132 | 528 |
| Test | 148 | 592 |
| Full domain | 576 | 2,304 |

Every complete pair contributes one instance of each answer, so cohorts and training batches are answer-balanced. Evaluation does not repeat cases to inflate the episode count. This first composition fixture uses a fixed four-symbol alphabet; natural-language rendering, fresh entity names, distractors and longer chains are later difficulty extensions.

Training starts from fresh weights. The first 200 optimizer steps draw only training pairs with one identity table, making the same composition operation initially reduce to a lookup. Subsequent steps use all training pairs. The graph, targets, query format and output interface do not change between phases. AdamW uses a 20-step learning-rate warmup followed by constant LR 0.003, clipping at 1 and the existing routing/balance losses. No frozen regional backbone or fitted decoder correction is used.

The follow-up training path adds two independently selectable mechanisms. `--copy-warmup 100` first draws training pairs whose composition is the identity, so the same task initially copies the query. `--source-reconstruction 0.5` adds a shared 528-parameter training-only head that reconstructs each source's four observed table entries from its own latent publication, both before and after query processing. The reconstruction targets are input bytes, not intermediate answers. Gradients update the source/regional network. Inference still uses only the common byte head. With both options, the schedule is 100 copy steps, 200 lookup steps, then full composition; the training allocation is 253,006 parameters and the inference graph remains 252,478.

Validation selects the lowest answer NLL at the trained round count after the lookup warmup. Test evaluation follows checkpoint selection. Checkpoints save model and optimizer tensors, sampling and torch RNG state, step, history, protocol and source hashes. Episode state resets between independent training batches; source state remains differentiable from description to query inside each batch. Resuming requires the same protocol and source.

## Comparisons

The same trained recurrent model is evaluated at K=1,2,4, with each query starting from the same immutable encoded prefix state. External query bytes are ingested once regardless of K. Controls disable delayed exchange at K=2, reset memory before the query, or shuffle whole B tables while preserving C and the query. For shuffling, both the original answer and the answer implied by the replacement table are scored.

A separate one-pass control activates twice as many experts per block, has the same allocated parameter count and starts with identical initial parameter tensors for its matched seed. It has no delayed regional exchange. Its query expert budget equals K=2; its prefix costs more because it also uses twice as many experts while encoding the tables. Report both costs, rather than labeling the entire run compute-matched.

| Configuration | Prefix calls/case | Query calls/case | Total calls/case |
| --- | ---: | ---: | ---: |
| Sparse K=1 | 96 | 84 | 180 |
| Sparse K=2 | 96 | 168 | 264 |
| Sparse K=4 | 96 | 336 | 432 |
| Wider one-pass control | 192 | 168 | 360 |

Each intact case supplies eight table bytes and three query-time bytes, independent of K. Each model has 252,478 parameters and 384 physical experts. Independent initializations are used for recurrent seeds; the feedforward query graph is paired with the first seed. A successful bounded composition result does not by itself complete M7's combined memory/shared-private/natural-text demonstration.

## Commands

```bash
python -m experiments.text_regions.composition_train \
  --out artifacts/composition_seed1 --steps 1000 --batch 64 \
  --seed 20261105 --validate-every 100

python -m experiments.text_regions.composition_eval \
  artifacts/composition_seed1/best.pt --split test --controls \
  --out artifacts/composition_seed1/test.json

python -m experiments.text_regions.composition_train \
  --out artifacts/composition_feedforward_seed1 --steps 1000 --batch 64 \
  --seed 20261105 --rounds 1 --no-feedback --active-groups 4

python -m unittest -v experiments.text_regions.test_composition

# Train the representation-learning follow-up on the same task and splits.
python -m experiments.text_regions.composition_train \
  --out artifacts/composition_reconstruction_seed1 --steps 1000 --batch 64 \
  --seed 20261105 --copy-warmup 100 --lookup-warmup 200 \
  --source-reconstruction 0.5
```

To continue an interrupted run, repeat its exact training command with `--resume`. To stop at a planned checkpoint without changing the schedule, add `--stop-after N`. Recurrent repetitions use seeds 20261106 and 20261107 with separate output directories.
