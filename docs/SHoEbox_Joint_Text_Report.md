# SHoEbox: joint text and retained-memory learning

Completed November 5, 2026 (America/Denver). Locked phase targets: **not all passed**. Three parent/data pairs, one eleven-node graph and one shared 256-byte decoder per pair.

## Result

| Run seed | Selected update | Text NLL before | Text NLL after | Memory minimum |
| --- | --- | --- | --- | --- |
| 20261310 | 40 | 32.6135 | 7.8371 | 90.82% |
| 20261311 | 40 | 32.6135 | 8.5279 | 87.70% |
| 20261312 | 80 | 32.6135 | 8.0587 | 91.99% |

Mean held-out text loss is **8.1412 nats/byte** (11.7453 bits/byte). The unigram reference scores **3.3220 nats/byte** on the same bytes. The initial parent was specialized for four-value composition, so its large text loss is a transfer starting point, not a language-model baseline.

Each test evaluates 4,096 natural-text target bytes and 512 fresh episodes for every structured condition. Selection uses validation only. The three parents share the same frozen composition ancestor; their producer weights and new synthetic training episodes differ.

The validation trajectory records the full training run, including checkpoints rejected by the memory guard.

| Update | Mean text validation NLL | Worst memory validation score |
| --- | --- | --- |
| 0 | 32.3650 | 96.88% |
| 20 | 18.6666 | 93.75% |
| 40 | 8.8207 | 87.50% |
| 60 | 6.6649 | 84.38% |
| 80 | 6.0678 | 85.94% |
| 100 | 6.0614 | 79.69% |
| 120 | 6.2608 | 73.44% |
| 140 | 5.9146 | 75.00% |
| 160 | 5.6914 | 76.56% |

The fixed-budget final weights are evaluated separately to show what continued joint learning did. These diagnostics were declared during training after observing validation interference, and do not replace the locked primary checkpoint selection. The memory columns compare the same five recall/correction tasks on the same episodes.

| Run seed | Final text NLL | Selected five-task minimum | Final five-task minimum |
| --- | --- | --- | --- |
| 20261310 | 4.7840 | 91.60% | 85.35% |
| 20261311 | 3.4657 | 90.62% | 96.09% |
| 20261312 | 9.0211 | 91.99% | 88.48% |

Final diagnostics meeting both the text NLL and five-task memory thresholds: run 20261311: 3.4657 nats/byte, 96.09% memory minimum, versus 87.50% memory on validation. These final memory diagnostics cover five recall/correction tasks; they do not include the four withheld-correction conditions in the selected primary evaluation.

A post-test coverage diagnostic on run 20261311 replayed its weakest final validation condition (shared left). The original 64-episode predictions reproduce exactly. Expanding that condition to 512 episodes changes its minimum field score from **87.50% to 95.90%**. The original validation and test-subset predictions remain identical across batch sizes. This diagnostic does not change checkpoint selection or the locked primary result.

| Run seed | Text target | Memory target | K2 gain over K1 | Ownership / resume |
| --- | --- | --- | --- | --- |
| 20261310 | fail | pass | 67.90 pp | pass |
| 20261311 | fail | fail | 67.12 pp | pass |
| 20261312 | fail | pass | 68.62 pp | pass |

The byte-stream interface is implemented and verified, but the locked joint-learning acceptance target is not met; M7 remains open. The next stage should broaden retention validation, extend text learning, and measure language/memory gradients by subsystem to explain the variation between runs. The final-checkpoint diagnostics expose useful progress that the current guard rejects; they remain separate from the selected primary results. Natural-language fact addressing follows reliable joint learning.

## What changed

Natural-text prediction and fresh memory episodes now update the same composer, expert banks, connections and byte decoder in one optimizer step. All four memory producers and their edge value projections continue learning; the inherited memory-edge context and gate weights remain fixed. The graph still has 510,958 parameters and 768 physical experts; no additional decoder or language model was added.

The byte-stream API carries graph state across chunks and saves the next-byte logits plus a private sampling RNG. Resuming generation consumes only new bytes. Text uses the A/B/C views; the four memory owners stay inactive and their outgoing edges are masked. The three source encodings are counted separately. Structured queries retain an explicit field address. Natural-language fact parsing and query addressing remain the next interface work.

Fixed 128-byte sample from the first run, prompted with ROMEO: and a newline, temperature 0.8. The literal below escapes line breaks and non-ASCII bytes; the artifact package also preserves the raw output.

```text
CCCCDDDDDDCCCDCDCDCCDCCCABBDDDDDDDDDDDDDDDDDDDDDDDDDDDAAAAAAAAADDDDDDDDDDDDAADDADDAADAAAAADDDDDDDDDDDDDDDBCAAAAAADDDDDDDDDDDDAAD
```

## Retained capabilities

| Condition | Shared | Left private | Right private |
| --- | --- | --- | --- |
| recall | 91.73% | 92.71% | 92.19% |
| shared left | 92.06% | 92.64% | 92.25% |
| shared right | 91.67% | 92.84% | 92.06% |
| left private | 91.86% | 92.97% | 92.12% |
| right private | 91.73% | 92.77% | 92.32% |
| withheld shared left | 91.34% | 92.84% | 92.25% |
| withheld shared right | 91.47% | 92.84% | 92.12% |
| withheld left private | 91.80% | 93.49% | 92.12% |
| withheld right private | 91.80% | 92.77% | 91.86% |

The primary minimum includes every field, correction direction and withheld update separately. Answers remain composed results g(f(x)); the retained value x is never decoded into an intermediate byte.

| Available sources | Shared | Left private | Right private |
| --- | --- | --- | --- |
| left only | 92.64% | 91.99% | 23.83% |
| right only | 92.77% | 25.91% | 91.67% |
| erase left | 92.32% | 25.78% | 91.67% |
| erase right | 92.58% | 91.99% | 23.83% |

Unavailable private facts use a conditional 25% prior. Erasure controls preserve intact source execution counts; source absence skips the absent producers.

## Interventions

| Memory condition | Shared | Left private | Right private |
| --- | --- | --- | --- |
| shared right | 91.67% | 92.84% | 92.06% |
| k1 | 23.63% | 24.22% | 25.07% |
| k4 | 45.51% | 47.33% | 45.77% |
| no exchange | 23.83% | 24.28% | 25.00% |
| no edge | 25.85% | 23.83% | 26.69% |
| reset | 25.85% | 23.83% | 26.69% |
| shuffle | 23.96% | 17.32% | 13.09% |
| always | 23.57% | 25.85% | 28.12% |
| long recall | 91.21% | 92.38% | 91.93% |
| long correction | 90.56% | 92.58% | 91.41% |
| repeated | 92.17% | 92.80% | 91.56% |

The one-, two- and four-round comparisons use identical retained observations and weights. No-exchange, blocked-memory, reset-memory, shuffled-memory and unconditional-publication controls match the intact two-round execution count. Longer gaps are reported separately from the primary target.

| Text condition | Mean nats/byte | Expert calls/byte | Seconds / 4,096 bytes |
| --- | --- | --- | --- |
| intact | 8.1412 | 168 | 115.46 |
| k1 | 7.8347 | 84 | 57.74 |
| k4 | 8.2965 | 336 | 208.39 |
| no exchange | 10.1921 | 168 | 83.54 |
| reset chunks | 10.2052 | 168 | 85.89 |

Text state-reset controls reset at each 16-byte chunk. Each test starts with an independent state and cursor. Natural-text K=1/2/4 comparisons are measurements of this checkpoint, not separately trained models. Timings are means from the CPU reference implementation while the three study workers run concurrently.

## Text interleaved with memory

| Run seed | Shared | Left private | Right private |
| --- | --- | --- | --- |
| 20261310 | 91.60% | 92.97% | 92.77% |
| 20261311 | 91.41% | 91.41% | 91.02% |
| 20261312 | 91.99% | 94.14% | 92.38% |

These episodes insert 32 natural-text bytes after fact acquisition and before querying, across 512 episodes per seed. Every retained owner tensor is preserved exactly. Each seed also exports a 128-byte autoregressive session with facts resident, followed by all three field queries. Split sampling at byte 41 reproduces the remaining 87 bytes, RNG, logits and complete state exactly.

## Learning, cost and verification

Each run performs 160 joint updates, consuming 20,480 natural-text bytes and 2,560 fresh memory episodes. Eight ordered language rows carry state across 16-byte backpropagation chunks. The memory objective cycles five tasks and gaps 0, 4, 8 and 16. Four alignment targets are fixed from each parent before training; no replay store is introduced.

AdamW uses betas (0.9, 0.95), weight decay 0.01, clipping at 1 and five warmup updates. Core, decoder and producer learning rates are 0.0003, 0.002 and 0.0001. Validation every 20 updates requires every memory task/field to remain at least 90%, then selects the lowest language loss.

| Run seed | Expert executions | Training minutes | Peak RSS MiB |
| --- | --- | --- | --- |
| 20261310 | 11,919,360 | 83.28 | 1137.6 |
| 20261311 | 11,919,360 | 79.70 | 1140.0 |
| 20261312 | 11,919,360 | 87.77 | 1140.6 |

Training time sums completed updates and excludes validation and audits. Forward source encodings and repeated expert executions are counted. All **408 regression tests pass**. Audit outputs reconstruct training bytes, synthetic batches, tables, source hashes, cursor positions, guarded selection, held-out predictions and the inference export.

## Reproduction

Extract the source and artifact ZIPs into the same directory; both use the SHoEbox/ root. The artifact package includes the three parent checkpoints, initial/selected/final joint checkpoints, the separate development pilot with its exact source, predictions, samples, audits and the hash-verified Tiny Shakespeare data for offline reproduction.

```bash
python scripts/prepare_data.py
python scripts/check_all.py
python scripts/audit_joint_text.py artifacts/joint_text/20261310
python scripts/diagnose_joint_text_validation.py artifacts/joint_text/20261311 \
  --out artifacts/joint_text/diagnostics/validation_20261311.json
python scripts/report_joint_text.py
python -m experiments.text_regions.byte_stream \
  --checkpoint artifacts/joint_text/20261310/best.pt \
  --prompt "ROMEO:" --count 128 --out sample.bin
```

Repeat the audit for the other two run directories. A fresh three-pair study uses `python scripts/run_joint_text.py --root artifacts/joint_text_fresh`; repeating that command resumes complete optimizer and stream state. A generation session can continue with `--resume-session sample.session.pt` and the same checkpoint.
