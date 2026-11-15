# 100M routing D: 40M plateau diagnosis

This is an isolated diagnostic harness for the 40M FineWeb checkpoint. It does
not change production training or checkpoint semantics. Neutral diagnostic
controls reproduce the original runtime exactly; alternative bias and auxiliary
policies are process-local and are restored explicitly by `run_arm.py`.

The run compares 11 continuations on the same 1M fresh byte targets, then extends
the control and two selected candidates to 3M. A separate 1M-byte audit set is
not scored until this confirmation. Development and fresh-screen losses select
the two candidates. `selection.json` records that decision before confirmation.

The first nine arms screen rates, prototype attraction, frozen existing biases,
and auxiliary objectives. Frozen probes then motivated two additional arms:
zero utilization bias with updates disabled, and a tenfold reduction of L2/L3
contrastive loss. These additions are diagnostic follow-ups, not a pre-registered
11-arm experiment. The original 40M model remains unchanged.

Required base-directory layout:

- `run_100m/latest.shoe`: original 20M checkpoint.
- `run_20m_40m/latest.shoe`: original 40M checkpoint.
- Both directories' original `results.json` files.
- `data/manifest.json`, `data/development.bin`.
- `diagnostic_40m/data/{train.bin,manifest.json,development.bin}`: the new 5M
  source pool and original development split. This pool is disjoint from all
  earlier training and development documents.

`setup.py` builds the neutral diagnostic runtime, partitions the source pool by
document, and checks exact continuation parity. `run_probe.py` evaluates frozen
ablations and reads activation/route state outside the training byte loop.
`gradient_probe.py` decomposes pending content gradients by objective;
`setup_followup.py` validates their sum against the actual native flush.
`snapshot_controls.py` localizes bias effects and tests an older centering mean.

Run from the repository with the corresponding base directory:

```bash
python experiments/plateau_100m/setup.py --base /path/to/base
python experiments/plateau_100m/setup_followup.py --base /path/to/base
python experiments/plateau_100m/launch.py --base /path/to/base
python experiments/plateau_100m/gradient_probe.py --base /path/to/base
python experiments/plateau_100m/snapshot_controls.py --base /path/to/base
python experiments/plateau_100m/confirm.py --base /path/to/base
python experiments/plateau_100m/finalize.py --base /path/to/base --out /path/to/output
```

The original execution added the two follow-up arms separately using
`runtime_followup`; `launch.py` now includes them for a complete replay. Its
neutral arithmetic matches the original runtime. Full per-document scores,
sampling plans, parent hashes and gradient checks are retained in the results.
Training snapshots and datasets are kept outside git.

Rate changes leave prototype attraction at 0.002. The quarter-rate setting is
fast/query 0.0002, slow 0.002 and head 0.004, matching D's documented continuation
rates. The zero-bias arm is an explicit experiment, not a persistent production
policy encoded by the `.shoe` checkpoint alone.
