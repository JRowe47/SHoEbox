# EXP01-E: joint paired readout and applicability

Research adapter on minimal corrected-C v19. Public `shoe_multiscale.Model`
remains unchanged. Read `reports/exp01e/PROTOCOL.md` before running fits.

The current two-byte stable constituent feeds a 65-weight sigmoid gate. It
scales the D3 paired residual before the native 256-byte tree distribution.
Combined next-byte loss updates the core, output residual and gate online.
Local presence/key/value reconstruction is separately supplied fixture training
supervision; generic streams never construct those labels. There is no inference
parser, oracle source index, exact store, new retained slot or history tape.

`paired` and `paired_latest` in the historical constituent adapter are unchanged.
This adapter uses a distinct E checkpoint magic; no saved D checkpoint is
silently reinterpreted. Total gated readout: 9,965 fitted weights, deducted from
the nominal budget. Its extra sequence state is zero. Scratch and training
counters are disclosed by `inventory()`.

```bash
python -m experiments.joint_readout.run pilot --out /tmp/e1-pilot --workers 1
python -m experiments.joint_readout.run train --out /tmp/e1 --workers 3
python -m experiments.joint_readout.run audit --out /tmp/e1 --workers 3
python -m experiments.joint_readout.prepare_data --out /tmp/e1-data
python -m experiments.joint_readout.regressions train --out /tmp/e1-regression --data /tmp/e1-data
python -m experiments.joint_readout.regressions audit --out /tmp/e1-regression --data /tmp/e1-data
python -m experiments.joint_readout.regressions retention --out /tmp/e1-retention --data /tmp/e1-data --binding /tmp/e1
python -m experiments.joint_readout.run train --parameters 1000000 --out /tmp/e1-1m
python -m experiments.joint_readout.run train --parameters 5000000 --out /tmp/e1-5m
python -m experiments.joint_readout.run audit --out /tmp/e1-1m
python -m experiments.joint_readout.run audit --out /tmp/e1-5m
```

Registrations are written before fitting and cannot be overwritten. Recipes are
fixed before audit; changed source is rejected. Put all fitted outputs outside
git. Pinned FineWeb text hashes, document boundaries and binary hashes allow
checking the corpus returned by the external row API, whose main can change.

Binding checkpoints have JSON companion cursors with both RNG states, remaining
shuffled order, partial episode and byte position. `stream.advance` continues
without replaying any fitted update. Evaluation-time `finish()` is a training
schedule event; exact original schedule reproduction includes it at each 500K
continuation boundary. Arbitrary interruptions between those events do not call
`finish()`. Language cursors record the document index/offset and corpus hash;
task cursors additionally record decoder updates and the final partial episode.

```bash
python -m unittest experiments.joint_readout.test_adapter -v
python -m experiments.joint_readout.check_native
python -m multiscale.tests.run_all --no-leak-check
```

Sources: [Miller et al. 2016](https://aclanthology.org/D16-1147/),
[See et al. 2017](https://aclanthology.org/P17-1099/),
[Arora et al. 2024](https://proceedings.mlr.press/v235/arora24a.html).
The learned switch is adapted to residual logits; it is not a pointer network.

The completed campaign preserves E2 logit switching, E3 probability switching,
E4 calibrated bits, E5 evidence inputs, E6 query-presence gating, and E7 observed
answer-request gating in separately named `experiments/joint_*` modules.
Each policy has a distinct checkpoint magic. E7's sixth local label is derived
only from the observed request byte; it does not inspect the answer mask.
Its generic fits have no local labels. See the full report for which development
and audit gates failed; no experiment here changes the public v19 default.
E8 is preregistered in `reports/exp01e/AMENDMENT_E8.md` and has not been fitted.

To rebuild the charts and Markdown from an exported run workspace:

```bash
python -m experiments.joint_readout.report --workspace /path/to/runs --out /tmp/report
python -m experiments.joint_readout.write_report --workspace /path/to/runs --out /tmp/report
```
