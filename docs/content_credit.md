# Connected content credit

The headcheck at `bd9b14d` found that every 16-byte execution chunk detached
the carried graph. Older eligibility traces reached admission policies and
payload adapters, but could not train the earlier content-producing network.

`ContentCreditRun` is the continuation trainer for this repair. Its default
optimizer window is 384 bytes per stream, executed in 16-byte chunks. All
carried state stays connected between those chunks: ingress, recurrent expert
and contextualizer state, publication, recall, routing and feedback. Inference,
parameter names and the 312,440-parameter parent graph are unchanged.

## Two boundaries, two purposes

| Setting | Default | Effect |
|---|---:|---|
| `--segment` | 16 | Execution chunk size; does not detach gradients. |
| `--horizon` | 384 | Optimizer window and exact gradient horizon. |
| `--batch` | 1 | Independent streams per source. |
| Global gradient cap | 1.0 | Existing cap after summing both source gradients. |

Only window entry detaches incoming state and captures a historical-credit
receipt. `capture()` itself detaches state, so removing the explicit trailing
detach without moving capture would leave part of the original break in place.
Internal chunks advance conditional eligibility but create no new receipts.
The entry receipt is applied once after backward; adding internal receipts would
count credit already carried by exact autograd twice. Document resets still
clear the affected rows and their histories.

Exact credit is bounded by the optimizer window, not by elapsed stream lifetime.
An observation and its later target must lie within the same window for the full
content derivative to connect. Outside it, the existing conditional admission
and payload traces retain their previous scope. Raising the horizon increases
activation memory. No unit-norm gradient normalization or replay is added.

## Training and recovery

Language and task sources keep separate state, eligibility and cursors. They
produce gradients at identical weights before the existing summed update and
global cap. Warm-start transfers the selected parent's weights and AdamW moments;
source states and cursors start fresh, recorded in the initialization metadata.
Resuming a continuation restores its own complete optimizer, source and RNG state.

```bash
python -m experiments.text_regions.content_credit_train \
  --data data/tiny_shakespeare.txt \
  --parent artifacts/state_credit/state/best.pt \
  --out artifacts/content_credit/continuation \
  --horizon 384 --segment 16 --batch 1 --steps 8 --eval-every 4
```

Use `--resume` with the same protocol to continue `last.pt`. `best.pt` is selected
by mean validation language/answer NLL, with earlier ties retained. The optional
`--cut-at-segments` flag is the matched old-path ablation; it requires the segment
size to divide the horizon. Legacy trainers remain available for reproducing
their published experiments.

## Verification

The original observation-to-answer probe spans 148 bytes, from source position
32 to query position 180. Its source gradient is zero with the old cutoff and
0.013164902495110228 with the repair. The repaired gradient and forward logits
match the uncut reference exactly. The probe uses a 192-byte window, with padding
only after the measured query so the legacy divisor requirement is satisfied.

Thirty-five relevant checks cover all-parameter gradient agreement, uneven final
chunks, older credit applied once, document resets, explicit optimizer-boundary
detachment, source isolation, exact resume, existing eligibility and runtime
fusion. A real paired-source smoke update exercises the default 384-byte window.
Its checkpoint is a continuation smoke test, not a validation-selected result.
The update processes 768 bytes and 19 answer targets in 217.3 CPU seconds,
with 7.07 GiB peak process memory. All weights remain finite; all 757 parameter
tensors change. Immediate acquisition stays at 25% after this single update.
See [recorded measurements](../experiments/text_regions/results/content_credit/summary.json).

```bash
python scripts/probe_content_credit.py
python scripts/smoke_content_credit.py
python -m unittest -q experiments.text_regions.test_content_credit \
  experiments.text_regions.test_state_credit \
  experiments.text_regions.test_context_balance \
  experiments.text_regions.test_runtime_fusion
```

The smoke command requires a fresh output directory. Its before/after probes
contain 32 immediate-acquisition and 64 natural-byte targets. They check that a
real update completes; subsequent ordinary shared-stream training must establish
reliable value-conditioned recall.
