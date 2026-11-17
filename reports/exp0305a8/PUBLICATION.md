# Publication provenance

The original pre-fit commit is `d622eb08e4ab5ee72315c739f5439255665b373f`.
It was created locally with passing source/data/control receipts before any
research fit. The plan and all pre-fit/fitting timestamps and hashes are unchanged.

The command-line push failed because this environment has no GitHub Git credentials.
Its failed log and process cost are retained. The connected GitHub API publishes
identical source trees with server-assigned commit metadata/IDs. The pre-fit
snapshot publication is explicitly a later mirror, not a claim that its GitHub
commit was made before fitting. Publication preserves the previous remote main
as an ancestor and uses only fast-forward ref updates.

`PREFIT_COMMIT.txt` contains the original raw Git commit object, including its
original timestamp, parent and tree. Its Git object hash is the original pre-fit
SHA. The archive's complete `source_history.bundle` preserves both original
local research commits and the published history. The publication manifest
identifies the final published SHA and the exact matching local source tree.
No source/runtime/data/outcome lock or research result was rewritten for transport.

A fresh ordinary GitHub clone can reconstruct the original pre-fit object from
its published tree and the preserved raw object before the delivery verifier:

```bash
git hash-object -t commit -w reports/exp0305a8/PREFIT_COMMIT.txt
```

It must print `d622eb08e4ab5ee72315c739f5439255665b373f`. The included bundle
already contains this object and needs no reconstruction.
