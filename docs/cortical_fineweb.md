# Cortical SHoEbox: 10M FineWeb run

The `cortical_10m()` preset has **9,982,290 trainable parameters**. It preserves
the complete Cortical Input Map v2: 18 independent ordinary HAT-SHoE regions,
three sparse expert layers per region, two recurrent rounds, full 1/2/4-byte
ports, and one shared hippocampus accessible only through EC-C and EC-X.

| Setting | Value |
| --- | ---: |
| Regional width | 64 |
| Experts per region / per layer | 192 / 64 |
| Physical experts | 3,456 |
| Expert hidden width | 16 |
| Routing groups / selected groups | 8 / 1 |
| Selected experts per layer | 1 |
| Laces / selected laces | 32 / 2 |
| HAT levels / symbols | 1 / 64 |
| minKron heads / key / value dimensions | 2 / 8 / 8 |
| Hippocampal capacity / returns per stream | 32 / 2 |
| Hippocampal address / active coordinates | 64 / 8 |

There are 102 selected expert executions per external byte: A1/B1 run once,
and the other 16 regions run twice. Expert weights account for 7,077,888
parameters. Routing, recurrent, HAT and output-port work is additional.

## Completed run

| Measurement | Result |
| --- | ---: |
| Training targets / updates | 100,000 / 250 |
| Initial / final holdout NLL | 5.556483 / 3.266079 |
| Final holdout accuracy | 17.65% |
| Training minutes / bytes per second | 17.63 / 94.52 |
| Unigram / bigram count baseline NLL | 3.255803 / 2.680461 |

The model learned the broad byte distribution. Samples remain character-level
fragments, and the final holdout score does not improve on the unigram count
baseline. The best measured holdout NLL was 3.250240 at 40,000 targets. The
reported final checkpoint is the requested 100,000-target endpoint.

All 11 checks pass. A final reload audit verifies 9,982,290 finite parameters,
250 updates, exactly 100,000 targets, 500 observations/hippocampal writes per
stream, 32 retained episodes per stream and the original training source
fingerprint. See [compact results](cortical_fineweb_results.json).

## Data and training

```bash
python -m pip install -r requirements.txt
python scripts/prepare_cortical_fineweb.py --out data/cortical_fineweb
python -m shoebox.train_fineweb \
  --data data/cortical_fineweb --out artifacts/cortical_fineweb_10m \
  --context 2 --threads 2 --lr 0.002 --warmup-updates 10 \
  --checkpoint-every 50 --sample-bytes 96
```

The source is the official `HuggingFaceFW/fineweb` dataset, `sample-10BT`
configuration, through Hugging Face's row-slice endpoint. The preparation
command uses the first 100 returned rows, requires complete text cells,
encodes UTF-8, inserts two newline bytes between documents and records source
IDs, text hashes and exact byte offsets. The manifest pins the downloaded
bytes; its observed Hub revision is informational because the viewer serves
`main` rather than an explicitly pinned revision.

Training consumes exactly 100,000 next-byte targets in one pass, across 200
independent contiguous streams. Each stream has 500 targets and one unscored
prefix byte, so the saved training corpus contains 100,200 bytes. No target
position is reused. The prepared training slice covers source rows 0-38;
validation uses separate source rows 39-40 and contains 4,096 scored targets
plus 32 prefix bytes. Training/validation source IDs and text hashes must be
disjoint.

Every two observed bytes, truncated backpropagation detaches the carried
regional states, two-occurrence transport history and bounded hippocampal
store. State persists across all 500 observations within a stream. There is
no corpus wrap or replay. Document separators are ordinary bytes; the graph
does not reset at them. Two-byte truncation is the gradient horizon across
updates, not the lifetime of the forward recurrent state.

The run uses FP32 on CPU, AdamW with betas (0.9, 0.95), peak learning rate
0.002 after ten warmup updates, constant learning rate thereafter, weight
decay 0.01 excluding biases/norm vectors and marked recurrent controls,
global gradient-norm clipping at 1, and the existing 0.001 auxiliary weight.
There are 250 optimizer updates, each scoring 400 byte positions.

Evaluation resets its own independent state and carries it through the same
fixed holdout. It does not touch training state or update parameters. The
first actual training batch also checks gradients from pure next-byte loss
into every region and the hippocampus, excluding auxiliary losses.

## Checkpoints and continuation

`last.pt` saves parameters, AdamW moments, every runtime state, RNG, exact
stream cursor, history, byte count and the source/protocol fingerprint every
50 updates and at completion. `results.json`, `history.jsonl`, `config.json`,
`data_manifest.json` and `topology.json` accompany it.

For an interrupted run, retain its exact source, corpus and arguments:

```bash
python -m shoebox.train_fineweb \
  --data data/cortical_fineweb --out artifacts/cortical_fineweb_resumed \
  --resume artifacts/cortical_fineweb_10m/last.pt \
  --context 2 --threads 2 --lr 0.002 --warmup-updates 10 \
  --checkpoint-every 50 --sample-bytes 96
```

Resume finishes the original byte budget. It does not implicitly start a
second pass. The source/protocol check rejects incompatible continuation.
The completed checkpoint can also be loaded through `shoebox.checkpoint.load`
for inference or a new explicitly defined training phase.

The optional samples are ordinary external autoregressive byte continuations:
generated bytes become the next external text input. This runner adds no
internal world model or autonomous simulation mechanism to the cortex.

## Validation

```bash
python scripts/check_all.py
```

The existing nine architecture checks cover topology, causal framing,
independent state, gradient connectivity and checkpoint behavior. Two added
checks verify the 10M parameter count, exact byte accounting, disjoint source
documents, rejected truncated source text and bit-exact model updates across
an interrupted/resumed run through the production training command.

Sources: [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) and
[official row-slice API](https://huggingface.co/docs/dataset-viewer/rows).
