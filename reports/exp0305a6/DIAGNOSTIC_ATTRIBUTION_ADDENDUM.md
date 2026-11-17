# Additional read-only attribution, before execution

The registered A6 traces show 100% diverse256 target survival, zero observed
true-record false negatives, and 96.875% teacher-fed surviving-target accuracy
on the first 32 worlds, while complete autoregressive primary diverse256 is
84.375%. Competing nearest identity remains 96.875% for matched A5/A6.

Before selecting the next revision, summarize existing locked outcomes by the
first 32 versus remaining development worlds and compare the registered trace
with the identical primary subset. Do not infer full-world performance from the
small trace. Count primary autoregressive episodes only; no new inference.

Separately traverse the exact registered continuation cursor with a data-only
counter (no model/weights), keeping the last eight input bytes within each episode.
For each completed record window, count local positive key events and distinct
raw windows, including how many contain padding. Count fully diverse eight-byte
negative windows, their repetitions, and the number of unique raw windows.
Retain per-key/phase counts. This measures curriculum coverage, not separability,
learnability or a counterfactual candidate's performance. No alternative objective,
coefficient, codebook, policy or augmented example is evaluated. Charge process
CPU; zero fitted/inference bytes. Keep source and input-file hashes with output.
