# Fresh-data joint continuation

The next phase starts from the stronger 512-update joint checkpoint. Its held-out
NLL is 2.392272, compared with 2.448039 for the freeze/thaw schedule. The joint
monitoring curve is still improving. The selected policy therefore keeps all
regions learning together at AdamW 0.001 and expands data and temporal credit.

| Setting | Policy |
|---|---|
| Initial weights and optimizer | `artifacts/joint_seed512/last.pt` |
| Training | Joint AdamW, betas (0.9, 0.95), matrix decay 0.01, clipping at 1 |
| Batch / credit window | 8 streams / 256 observed bytes |
| Higher-level temporal credit | 16 L3 updates per gradient window |
| New data | 524,288 targets from 195 previously unused FineWeb documents |
| Budget | 256 additional optimizer updates; one pass through the new corpus |
| Lifetime progress | 768 optimizer updates / 1,048,576 byte-target exposures |
| Validation | Original 65,536 targets across 23 documents; evaluation chunks stay at 128 |
| Future prediction losses | Four-byte weight 0.2; sixteen-byte weight 0.1 |

The new source rows are official FineWeb `sample-10BT` viewer rows 1000-1299.
Complete responses and their hashes are retained in `data/fineweb_phase2_raw`.
Preparation excludes prior training, validation and reserved document IDs and
text hashes; validation bytes and metadata are copied exactly. Only 195 of the
downloaded documents are needed to fill the new corpus.

Changing the data is an explicit `--new-stream` initialization. Model parameters,
Adam moments, update ages and accumulated training-pair counts are preserved.
Recurrent state, corpus cursor and corpus epoch start fresh for the unrelated
streams. The controller remains in joint mode. A resumed run restores the same
stream state and does not reset it again.

Lifetime byte counts now accumulate the actual sizes of earlier phases. Changing
the gradient window does not retroactively double earlier exposure counts.
`--eval-context` separates monitoring from training horizon, keeping the 8,192-
target monitoring window comparable with the preceding phase.

```bash
python -m scripts.prepare_hierarchy_phase \
  --rows-json data/fineweb_phase2_raw/rows_1000.json \
              data/fineweb_phase2_raw/rows_1100.json \
              data/fineweb_phase2_raw/rows_1200.json \
  --previous data/hierarchy_disjoint --out data/hierarchy_fresh524k

python -m shoebox.train_hierarchy --data data/hierarchy_fresh524k \
  --initialize-from artifacts/joint_seed512/last.pt --new-stream \
  --out artifacts/joint_fresh256 --training-mode joint --cycle-warmup 0 \
  --context 256 --eval-context 128 --updates 256 --threads 1 \
  --checkpoint-every 8 --validate-every 32 --log-every 16
```

For exact resume, replace `--initialize-from ... --new-stream` with
`--resume artifacts/joint_fresh256/last.pt`; retain the other options.

The checkpoint bundle extracts into the repository's `artifacts/` and `data/`
directories. For a fresh reproduction, choose unused output directories rather
than overwriting the delivered data or completed run.

```bash
python -m scripts.audit_hierarchy_checkpoint artifacts/joint_fresh256 \
  --data data/hierarchy_fresh524k

# Reporting additionally uses Matplotlib and ReportLab.
python -m scripts.report_hierarchy_phase --seed artifacts/joint_seed512 \
  --run artifacts/joint_fresh256 --data data/hierarchy_fresh524k \
  --out reports/fresh_joint
```

This phase measures progress under the selected policy. Fresh data and a longer
credit window change together, so it does not separately attribute their effects.
