# EXP03/05: retention and interference in existing bounded state

The public E8 runtime is unchanged. A1 is a separately named research policy
that prioritizes learned record windows during replacement of the existing
16 L1 retained units. It adds no fitted parameters or persistent sequence state.
It improves diverse-filler retention but fails the registered development gates.
The fresh audit remains CLOSED. A2 query-first selection and its three-value
update-fixture amendment are preregistered, not fitted.

Read `reports/exp0305/SHoEbox_EXP0305_Report.md`, the diagnosis/A1 registrations,
`A2_REGISTRATION.md`, and `A2_FIXTURE_AMENDMENT.md`. The binary A1 overwrite
fixture has a perfect latest-value shortcut and is diagnostic only. Its original
failed scores and costs remain intact. All outputs/checkpoints belong outside git.

```bash
python -m unittest experiments.interference_retention.test_research -v
python -m experiments.interference_retention.check_native
python -m experiments.interference_retention.diagnose \
  --checkpoint /path/inherited_e8/local_gate.qr --out /tmp/new-diagnosis
# Fresh reproduction charges another 20.25M bytes; never use for A2 implicitly:
python -m experiments.interference_retention.run --out /tmp/new-a1
python -m experiments.interference_retention.extended \
  --runs /tmp/new-a1 --out /tmp/new-a1-extended
python -m experiments.interference_retention.verify_cursors \
  --runs /tmp/new-a1 --out /tmp/new-a1-cursors.json
python -m experiments.interference_retention.update_metrics \
  --runs /tmp/new-a1 --out /tmp/new-a1-updates.json
# Only after every other local CPU job has stopped:
python -m experiments.interference_retention.benchmark \
  --runs /tmp/new-a1 --out /tmp/new-a1-speed.json
```

`run` writes the source/data registration before compiling/fitting and refuses
an existing destination. Native/E8/A1 each gets its own 250K native warmup plus
2M continuation across seeds 1,7,19. Every byte, including padding, is charged.
The fixed local observed-window labels are separate from the next-byte loss;
generic streams must pass no labels. The inference alphabet always has 256 bytes.

The new A1 header is composed from the unchanged research E8 header and the
checked admission hook. A1 magic is `0x3141353050584553`; research E8 remains
`0x38455041514853`. Cross-policy loads fail. The nested core remains v19.
No public constructor, numerical policy, historical source or checkpoint changes.
`backend.Model(policy='a1')` selects A1 explicitly; `policy='e8'` is the unchanged
research E8 with read-only instrumentation. The external cache-freeze diagnostic
is not called by candidate inference and is rejected on trainable models.

## Resume an actual fitted cursor

```python
import json
from pathlib import Path
from experiments.interference_retention.backend import Model
from experiments.interference_retention.stream import advance

path=Path('../exp0305/a1/7/a1.qr')
cursor=json.loads(path.with_suffix('.cursor.json').read_text())
with Model.loads(path.read_bytes(), policy='a1') as model:
    # A new, explicitly charged 1,000-byte extension; never overwrite the source.
    cursor,loss=advance(model,cursor,1000)
    Path('../continued-a1.qr').write_bytes(model.dumps())
    Path('../continued-a1.cursor.json').write_text(json.dumps(cursor,indent=2)+'\n')
```

Use `policy='e8'` for E8 and native control files. Save the model and JSON cursor
together. The cursor preserves the partial episode, both RNGs, unused shuffle
order, record/query variant, condition, spacers, byte position and scheduled
finish events. Never reset, finish, replay a warmup or discard a partial example
at an arbitrary resume boundary. Finish is executed by `stream.advance` at the
registered byte events. `verify_cursors` checks all 18 actual fitted pairs with
1,000 uninterrupted versus 371 + save/load +629 discarded updates.

The archive includes pinned generated worlds, all actual snapshots/cursors,
initial and corrected diagnostic sources, raw logs and process ledgers. No new
generic corpus was used; E1/E7 generic audit documents/tasks remain excluded
from subsequent qualification. Reproduce the original frozen diagnosis with
the included inherited E8 checkpoint and its original loader, not A1.
