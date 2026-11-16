# EXP01-E3: preserved research policy

Normalized probability mixture with exact online posterior-responsibility credit. This transferred-policy pilot fails before generic continuation; no from-scratch E3 result is claimed.

Read [the registration](../../reports/exp01e/AMENDMENT_E3.md) and
[the complete report](../../reports/exp01e/SHoEbox_EXP01E_Report.md).
Each module uses its own checkpoint magic; old policies are never silently
converted. The pilot conversion functions name and hash the source format.
All trained output belongs outside git. Production remains minimal v19.

```bash
python -m experiments.joint_mixture.pilot --out /tmp/e3 --data /tmp/e1-data --source /tmp/e2/binding.qr
python -m unittest experiments.joint_mixture.test_adapter
python -m experiments.joint_mixture.check_native
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
