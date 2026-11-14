# Addressed version-2 checkpoint backend

The three native source files here are byte-identical to commit
`4e52beede8ef800e5f7a52ecac612cd75ebb644b`. They preserve the original additive
memory write, energy normalizer, parameter allocation, and unfinished-block
semantics for saved version-2 models. Do not silently load their matrices into
the new residual-write policy.

`Model.loads` selects this backend automatically. Use
`Model(memory_profile="addressed_v2")` for a fresh historical comparison.
New `Model()` instances use version 3. At the Python API boundary, cold
inference now preserves persistent association records for all three backends;
`clear_memory=True` explicitly requests their removal. This caller-selected
lifecycle operation does not change checkpoint continuation arithmetic.
