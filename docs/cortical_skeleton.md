# SHoEbox cortical skeleton

Cortical Input Map v2 implementation - November 8, 2026.

The new active model contains 18 ordinary HAT-SHoEs and one shared hippocampus.
It starts fresh rather than migrating the old seven/eleven-node checkpoints.
The original query/retention experiments remain reproducible under
experiments/text_regions; none of their controllers or regional archives is
imported into the active model.

## Capacity

| Item | Default |
| --- | ---: |
| Parameters | 635,466 |
| Ordinary regions | 18 |
| Physical experts | 864 |
| Expert layers per region | 3 |
| Selected experts per layer | 1 of 16 |
| Hidden width | 16 |
| HAT levels / symbols | 1 / 16 |
| minKron heads / key / value | 1 / 4 / 4 |
| Recurrent rounds | 2 |
| Episodic capacity per independent stream | 32 |
| Maximum returned episodes per round | 2 |
| Hippocampal address / active coordinates | 16 / 4 |

A1 and B1 execute once per observation; the other 16 execute twice. That is
102 selected expert executions per external byte at the default setting.
HAT, routing and byte-port projections still perform ordinary dense work.

## Exact ordered recipes

Every symbol below names a complete learned output port. A 2-byte port is
independently learned; it is never a truncation of a 4-byte token. Two frames
from a producer are consecutive observation occurrences, oldest first.

| Region | Current-round inputs | Previous-round context |
| --- | --- | --- |
| A1 | External bytes | None |
| B1 | Same external bytes; initially slower ordinary dynamics | None |
| L2 | 2 x 4B A1 | 1B L3 |
| S2 | 4B A1 + 4B B1 | 1B S3 |
| R2 | 2 x 4B A1 + 4B B1 | 1B R3 |
| F2 | 2 x 4B B1 + 2B A1 | 1B F4 |
| L3 | 2 x 4B L2 + 2B S2 | 1B F3 + 2B EC-C |
| S3 | 2 x 4B S2 + 2B L2 | 1B E4 + 2B EC-C |
| R3 | 2 x 4B R2 + 4B S2 | 1B F3 |
| F3 | 2 x 4B F2 + 4B L2 + 4B R2 | 1B G5 + 1B M4 + 2B EC-X |
| E4 | 2 x 4B S3 + 2 x 4B R3 + 4B L3 | 1B G5 + 2B EC-X |
| F4 | 2 x 4B F3 + 4B R3 + 4B L3 | 1B G5 + 1B M4 |
| V4 | 2 x 4B S3 + 4B F3 + 2B B1 | 1B G5 |
| M4 | 4B R3 + 4B F3 + 2B B1 | 1B F4 |
| G5 | 2 x 4B E4 + 2 x 4B F4 + 4B V4 | None |
| EC-C | 4B L3 + 4B S3 + 2B S2 | 1B EC-X + framed hippocampal return |
| EC-X | 4B E4 + 4B F3 | 1B G5 + 1B M4 + 1B EC-C + framed hippocampal return |
| Y | 2 x 4B L3 + 4B F4 + 2B S2 + 2B EC-C | Preceding emitted byte |

The backbone is acyclic inside a round. EC-C to Y is a same-round convergence
edge. Reciprocal EC links and EC-to-language/task/situation feedback use the
previous snapshot. Hippocampal lookup uses the EC features committed before
the round and episodes stored before the current observation.

## Regional computation and framing

The HAT receives the complete ordered sequence, with immediate source, original
producer, frame ordinal, byte position and port width represented separately.
A joint ordinary minKron sequence allows symbols from different producers to
interact. The final HAT representation enters three sparse SHoE blocks.
The next-byte objective reaches upstream producers through hard-forward,
straight-through byte ports. At inference each port carries actual integer
bytes. C0-C10 categories are documentation only.

Every region has separate HAT, sparse expert and recurrent state. The initial
B1 decay rate is four times slower, with all time constants remaining learned.
No cortical component has an archive, write/hold retention policy, memory
request decoder or query-specific lace head.

Two recent complete occurrences per port support the multi-frame recipes.
A new recurrent revision replaces the current occurrence. It does not append
another observation or another episode. The trace records the source event
and revision at every edge. Missing startup frames are absent rather than
invented zero tokens. Ordinary state is never forced to expire at a fixed age.

## Shared hippocampus

Only EC-C and EC-X publish to the subsystem, once after the last round.
The skeleton stores detached EC content/context frames and their joint features,
with FIFO capacity 32. A learned sparse address retrieves up to two distinct
episodes. A learned null competes with the selected associations; each return
has its own frame, source and original per-batch occurrence/revision metadata.
Stored features are re-encoded with current address weights at read time.

Returns enter the same EC HAT input path as other framed sources. Downstream
regions see ordinary EC publications. This is a small bounded associative
implementation for shakedown; lateral sharding and selective admission remain
future work inside the shared subsystem.

## Learning and execution

~~~bash
python -m pip install -r requirements.txt
python scripts/check_all.py
python -m shoebox.shakedown --steps 24 --out artifacts/cortical_skeleton
~~~

The default fixed fixture contains short ordinary text and tests optimization
plumbing. It uses batch 2, four observed bytes per row, two rounds, one CPU thread,
AdamW at 0.003, a 0.001 routing/HAT balancing term, and global norm clipping at 1.
Fresh state is used for each fixture update.

The completed local run reduced next-byte NLL from 5.571365 to 3.629863 and
accuracy rose from 0 to 25% on its eight training positions. Training took
59.27 seconds (3.24 batch bytes/second). These are smoke-fit measurements.

~~~bash
python -m shoebox.shakedown --data your_text.txt --context 8 --batch 2 \
  --steps 100 --out artifacts/cortical_stream
python -m shoebox.shakedown --resume artifacts/cortical_stream/last.pt \
  --data your_text.txt --context 8 --batch 2 --steps 20 \
  --out artifacts/cortical_stream_continued
~~~

With --data, rows are independent contiguous streams. State crosses chunk
boundaries with ordinary truncated backpropagation and resets on corpus wrap.
Resume restores the model, full runtime state, AdamW, RNG, corpus checksum and
stream cursor. A new output directory prevents overwriting a previous run.
Resume inherits optimizer settings from the checkpoint.

The forward API accepts external byte observations and optional preceding
output bytes on a separate continuation input. Current targets are never read
as model inputs. This skeleton does not include autonomous world rollouts or a
free-running generation scheduler.

## Validation and files

Nine behavioral checks cover the exact topology/removal boundary, once-only
ingress, committed feedback, chunk equivalence, causality, batch independence,
frame/source/order sensitivity, hard byte transport gradients, task-loss
gradients through every region and the hippocampal address, bounded episodes,
memory intervention, output/observation separation, input validation, and exact
next-update checkpoint resume. The pure task-loss check excludes balancing
losses so auxiliaries cannot conceal disconnected regions.

The delivery contains the complete source, initial.pt, last.pt, topology.json,
trace.json, results.json, and Markdown/PDF reports. The initial checkpoint is
the clean shakedown starting point; last.pt is the small fixture-fit example.
The executable recipes are in shoebox/config.py. Optional expert/codebook
sharing is left off so all state ownership is explicit.
