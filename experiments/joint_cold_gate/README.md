# EXP01-E8: fixed cold-start local applicability

E8 changes only the local applicability intercept from zero to `-log(255)`,
so the untrained local gate is 1/256. E7's numerical policy and every historical
checkpoint are preserved. E8 has magic `0x38455041514853`; cross-policy loading
is rejected. There are 7,670 fitted adapter weights within the requested
budget, no extra persistent sequence state and no optimizer moments.

Read the original [E8 registration](../../reports/exp01e/AMENDMENT_E8.md) and
[final qualification registration](../../reports/exp01e8/FORMAL_E8.md).
The local gate and key/value recognizers learn from explicit observed-prefix
fixture labels. Generic next-byte objectives never supply those labels and
leave those fitted parameters frozen. Native mixture-responsibility credit
remains E7's. The fixed prior is this experiment's design choice.

Use the pinned `exp01e-data` retention corpus and `exp01e7-data` fresh generic
corpus from the evidence archive. Their original hashes and document boundaries
are checked. E8 reused E7's then-unopened generic split; the final audits are now opened. Do not
run the inherited `prepare_data.py` helper to substitute a new corpus.

```bash
python -m experiments.joint_cold_gate.pilot --out /tmp/e8-pilot --data /path/exp01e-data --arm local
python -m experiments.joint_cold_gate.regressions train --out /tmp/e8-regressions --data /path/exp01e7-data
python -m experiments.joint_cold_gate.qualify train --out /tmp/e8-formal --data /path/exp01e-data
python -m experiments.joint_cold_gate.qualify train --parameters 1000000 --out /tmp/e8-1m --data /path/exp01e-data
python -m experiments.joint_cold_gate.qualify train --parameters 5000000 --out /tmp/e8-5m --data /path/exp01e-data
```

Training creates write-once source registrations and final checkpoints with
outer cursors. Audits require final locks and unchanged fitting dependencies.
Use `benchmark_qualification`, after all other CPU work has finished, for
capacity-matched qualification; the inherited `benchmark.py` is not the E8
campaign entry point. `measure_command` records whole-process CPU, including
child fitting/evaluation/compilation. Fit records also retain separate CPU costs.

```bash
python -m unittest experiments.joint_cold_gate.test_adapter experiments.joint_cold_gate.test_request experiments.joint_cold_gate.test_prior -v
python -m experiments.joint_cold_gate.check_native
python -m experiments.joint_cold_gate.check_request
python -m experiments.joint_cold_gate.verify_cursors --workspace /path/runs --out /tmp/exact-cursors.json
```

`stream.advance` resumes binding without replay. `resume.language_advance`
continues exact document positions, explicitly starting the next pinned epoch
when the saved corpus has ended. `resume.task_advance` finishes partial generic
examples and preserves the separately trained reconstruction decoder. Correctness
checks compare 1,000 updates with 371 + serialization + 629 on discarded copies;
they do not extend selected research fits. The registered evaluation `finish()`
events are part of the original fitting schedule.

Primary sources: [Miller et al. 2016](https://aclanthology.org/D16-1147/) for
paired key/value encodings and common weights; [See et al. 2017 §2.2](https://aclanthology.org/P17-1099/)
for generation/retrieval probability mixtures. The adapter combines two byte
distributions; it is not a pointer network. No parser, exact store or label
enters inference. E8 now passes all registered research gates. Public v19 remains unchanged pending
[versioned integration and parity](../../reports/exp01e8/PUBLIC_INTEGRATION_PLAN.md).
