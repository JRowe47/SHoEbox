# Signal and gradient audit

Runtime: `9f813f7677b175ea75faca852391b5673a8f3327`. The selected state-credit
checkpoint is `artifacts/state_credit/state/best.pt`; JSON records its SHA-256.

The network is connected: answer loss alone reaches all 757 parameter tensors.
Global norm clipping is active and verified. Native acquisition queries still
predict `1` throughout (8/32 correct). Historical autograd is cut every 16 bytes;
explicit traces add credit only to 45 policy and 20 payload-adapter coefficients.
A full 148-byte FP64 observation-to-answer derivative has norm 0.0131649, while
16-byte detachment makes it disconnected with exactly unchanged forward values.

Use `scripts/audit_signal_flow.py --phase PHASE` for `archive`, `forward`,
`telemetry`, `gradient`, `answer` and `derivative`. Each invocation writes to
`artifacts/headcheck` by default, without changing checkpoint weights. Supply
`--checkpoint`, `--data`, `--root` and `--out` to use alternate paths.
The checkpoint and corpus come from the existing state-credit artifact package.

`forward.json` preserves the initial run, including its exploratory telemetry.
`telemetry.json` is the separately verified capture and has zero deviation from
the uninstrumented run. The final harness separates these phases; raw records
retain the audit-script digest at their original run. `gradient.json` contains
the complete tensor-level inventory. `checks.json` and `checks.log` record the
63 passing regression checks. Production model and trainer code were not changed.

Next priority: broaden content-level delayed credit and establish reliable
value-conditioned short recall. Retain cap 1.0 and existing controls.
