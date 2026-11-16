# API boundary correction, registered before repair

The staged-default final suite passed 58/59 cases. The historical public API
rejects nominal budgets above 100,000,000. E8 deducted its 7,670 readout weights
before invoking the native core's cap check, so 100,000,001 was accepted.
The explicit historical constructors still rejected it; qualified 500K/1M/5M
models and all saved E8 snapshots were unaffected. No change was published.

Restore the historical nominal maximum at the public Python constructor before
any library allocation, preserving the RuntimeError rejection contract. Leave
native source, readout arithmetic, budgets of valid models, initialization,
training objectives and serialization untouched. This is API validation only,
not a numerical revision or research fit; fresh folds/prior tuning are inapplicable.

Return the default to v19 while checking the correction. Verify explicit E8
rejects 100,000,001 and restore staged E8 only after that check. Rerun the final
public/runtime Python suite and retain both the failed test log and CPU cost.
No C++ or timed inference code changes, so native/sanitizer and isolated-speed
results remain applicable. Publish only after the final staged-default suite passes.
