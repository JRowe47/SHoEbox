# EXP01-E7: preserved research policy

Explicit local observed-answer-request applicability, frozen on generic text. The development pilot retains binding through 1M language targets. Fresh generic qualification is recorded separately.

Read [the registration](../../reports/exp01e/AMENDMENT_E7.md) and
[the complete report](../../reports/exp01e/SHoEbox_EXP01E_Report.md).
Each module uses its own checkpoint magic; old policies are never silently
converted. The pilot conversion functions name and hash the source format.
All trained output belongs outside git. Production remains minimal v19.

```bash
python -m experiments.joint_request_gate.pilot --out /tmp/e7 --data /tmp/e1-data --arm local
python -m unittest experiments.joint_request_gate.test_adapter
python -m experiments.joint_request_gate.check_native
```

The adapters have bounded computation and no additional persistent sequence
state. Local key/value/presence labels are fixture-specific observed-window
supervision. Generic next-byte loss is a separate objective. The E6/E7 local
gate is frozen when those labels are absent; it is not learned from generic
byte loss. Inference contains no task parser, target label, source oracle or
external exact store. Kind aliases in these modules are internal controls;
use `experiments/constituent_readout` to reproduce historical D3.

References: [Miller et al. 2016](https://aclanthology.org/D16-1147/) for paired
key/value representations; [See et al. 2017](https://aclanthology.org/P17-1099/)
for learned generator/retrieval switches. These are byte-distribution adapters,
not pointer networks.
