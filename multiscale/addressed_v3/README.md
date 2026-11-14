# Preserved version-3 runtime

The native sources here are byte-identical to revision
`51030d6f7fdd8030c65cb6b2c3e5a10529fe90fe`. They preserve residual matrix writes,
original expert allocation, inference bindings and exact pending-block stream
continuation. Version-3 checkpoints select this backend automatically.
Use `Model(memory_profile="addressed_v3")` for the previous-policy control.
New models use version 4, including retained ordered-pair recall.
