# Predictive hierarchy pilot

Implements the November 8 architecture review against `c0bbf64`. The six-region
pilot uses fresh weights; the old 18-region model/checkpoints keep their existing
behavior and checkpoint format.

## Regional organization

| Region | Width | Cadence | Required new input | Committed context |
|---|---:|---:|---|---|
| A1 | 64 | 1 byte | Current observed byte | Previous L2 |
| B1 | 64 | 1 byte | Current observed byte | Previous S2 |
| L2 | 96 | 4 bytes | Four distinct A1 occurrences | Previous L3 |
| S2 | 96 | 4 bytes | Four ordered A1/B1 pairs | Previous L3 |
| L3 | 128 | 16 bytes | Four ordered L2/S2 span pairs | None |
| Y | 48 | 1 byte | New nonlinear local/mid/high conjunction | Cached levels in separate slots |

Each region has its own HAT, three recurrent SHoE layers and MinKron states.
B1 starts with a fourfold slower learned time constant. Each layer has 32
experts with four selected. The total is 3,050,848 parameters, including 398,336
parameters in training-only future heads. The single-region control has
287,096 parameters. These are deliberately different capacities; compare loss
and measured cost together.

Every publication is one 64-dimensional vector. Source identity, occurrence
ordering, inclusive span endpoints and publication age survive transport.
The receiving HAT takes one recurrent step per message, never per coordinate.
Relative span start/end/length and source/sequence-position embeddings are
encoded before the HAT. Context is an additive condition on new occurrences;
it is not inserted as a repeated observation. Context must predate the input
span. No second revision round runs over the same frames.

L2/S2 execute only when all four local spans are complete; L3 executes only
when all four middle spans are complete. Buffers survive chunk boundaries.
Every source required by a conjunction is present. All streams share the fixed
schedule. Sparsity operates inside the ordinary regional expert banks; no
regional balance objective or flat regional top-k can remove ingredients.

The scheduled work is 3.5625 regional executions and 6.5 HAT message occurrences
per external byte. `parameter_cost()` reports scheduled selected-parameter
touches, explicitly distinguished from FLOPs and wall time. AdamW still updates
dense expert banks. The AVX512/native kernels are reused without changes.

## Joint expression and causal supervision

The decoder projects `(observed byte, A1, B1)`, `(L2, S2)`, and `L3` into
separate common-width slots. A nonlinear layer consumes those slots,
local-by-mid/high products, ages and availability masks. Its conjunction enters
small ordinary Y as one new occurrence; a local residual joins Y's output
before the single inference byte head.

`model(values)[:, t]` conceptually predicts the successor of `values[:, t]`;
the actual API returns `result['logits']`. There is no previous-argmax input.
Training, evaluation, and sampling all call the same forward path. Generation
feeds the sampled byte into the next call. Targets never become model state.

At a 4-byte boundary, the joint L2/S2 representation predicts the next four
bytes. At a 16-byte boundary, L3 predicts the next 16. Each scale's CE is
averaged over valid boundary/stream/offset targets, including masked tails.
An optional `future_targets` tail supplies supervision past the gradient
window without putting those future values into the recurrent computation.

```text
loss = next_byte_CE + 0.2 * next_4_byte_CE + 0.1 * next_16_byte_CE
       + 0.001 * internal_expert/HAT_balance
```

The control consists of one A1 HAT-SHoE and the observed-byte/local projection
plus final head. It has no Y, higher regions or future heads. The auxiliary
heads execute only when targets are supplied; sampling never computes them.

## Prepare and run

The input files below are official FineWeb viewer response JSON with `rows`.
No network access is needed after preparing the corpus. Multiple response files
are accepted. Duplicate document IDs and text hashes are removed.

```bash
python -m scripts.prepare_hierarchy_data \
  --rows-json fineweb_rows_0.json fineweb_rows_100.json \
  --reserve-manifest data/cortical_fineweb/manifest.json \
  --out data/hierarchy

python -m shoebox.train_hierarchy --data data/hierarchy \
  --out artifacts/hierarchy --updates 1024
python -m shoebox.train_hierarchy --data data/hierarchy \
  --out artifacts/local --local-only --updates 1024

python -m scripts.compare_hierarchy artifacts/hierarchy/results.json \
  artifacts/local/results.json --out artifacts/comparison.json
```

The default corpus has 194,560 training targets and 65,536 held-out targets.
Validation draws from at least 16 disjoint source documents, at most 4,096
targets each, and resets state between documents. Data and source fingerprints
are recorded. `--reserve-manifest` excludes the old validation documents from
both new splits. Optional `--continuity data/cortical_fineweb/validation.bin`
reuses the old first-4,050-byte/B50 slice for a separate 4,000-target metric;
the runner checks its checksum and refuses training-document overlap.

Training defaults: B8, T128, AdamW, LR 0.001, 16-update warmup, clip norm 1.
State carries between chunks, detaches only at optimizer boundaries, and resets
at corpus wrap. The finite prepared training corpus repeats when needed:
1,024 updates means 1,048,576 exposures, not that many unique bytes. Counts of
updates, exposures, corpus size and epochs are separate in every result.

Partial validation uses 8,192 scored bytes at each 64-update checkpoint.
At 256 updates and at completion, validation uses all 65,536 bytes and fitted
unigram/bigram baselines from exactly the observed training pairs. Failure to
beat unigram by 0.05 nats and input-swap degradation below 0.01 nats stops a
longer run at 256 for local-path inspection. Final evaluation also swaps only
upper decoder summaries while preserving local inputs and all upstream states.
The hierarchy gate requires improvement over the local control and an upper
swap penalty above 0.01 nats. Route use and real external bytes/s are recorded.

## Checkpoints and checks

```bash
# Declare the full protocol, pause, then continue it exactly.
python -m shoebox.train_hierarchy --data data/hierarchy --out artifacts/pilot \
  --updates 1024 --max-updates 32
python -m shoebox.train_hierarchy --data data/hierarchy --out artifacts/pilot \
  --updates 1024 --resume artifacts/pilot/last.pt

python scripts/check_all.py
```

Versioned checkpoints include weights, AdamW moments, RNG, private recurrent
states, cached publications, incomplete spans, cursor, epoch, baseline counts
and protocol. Resumption rejects changed source/data/configuration/settings.
The tests verify future exclusion, target independence, exact chunk/reload and
optimizer continuation, complete recipes, partial-span preservation, gradients
to every scale, document isolation and fused/portable gradient agreement.

Learned regional routing, learned span boundaries, codebook transport and
episodic participation remain the review's later stages. The fixed pilot must
first establish predictive value before those stages are enabled.
