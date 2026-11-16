# Next implementation: expose and integrate the qualified E8 policy

E8 has completed its registered three-seed binding, retention, generic,
1M/5M, throughput and engineering qualification. The current implementation
is `experiments/joint_cold_gate`; public `shoe_multiscale.Model` remains v19.
No further E8 prior tuning or audit selection is authorized by this plan.

1. Add an explicitly versioned public E8 policy with the same 1/256 local
   applicability prior, parameter deduction, bounded state and numerical
   behavior. Preserve v19 and all full-core/historical constructor and saved
   checkpoint meanings. Give new public-policy checkpoints their own identity.
2. Expose optional training-only observed-window labels without adding a task
   parser, byte equality test, source oracle or exact store to inference.
   Generic byte calls supply no fixture labels and freeze the local readout
   and applicability parameters exactly as E8 does. Document this distinction.
3. Provide an explicit importer for the qualified E8 research checkpoints,
   recording source/destination hashes. Prove full-256-byte prediction,
   training, partial-constituent and outer-cursor continuation parity through
   that import; never silently reinterpret a historical checkpoint.
4. Verify the complete public API lifecycle, parameter/state inventories,
   native/Python parity, causality and ASan/UBSan, and remeasure isolated
   throughput. Use the pinned development data and exact qualified snapshots
   for integration checks. Promote the public default only after compatibility
   and numerical equivalence pass; otherwise keep v19 and publish the findings.
5. Treat any numerical/objective/state change as a new research revision with
   its own preregistration and budget. The E2-family binding and E7 generic
   audits are now OPENED. Exclude those worlds/document groups from new
   qualification folds and never tune against their scores.

All E8 results establish the explicitly fixture-supervised mechanism, including
its generic fallback and retention. They do not establish that generic byte
objectives learn the local extraction/applicability heads. Broader task learning
remains separately named research.

Direct commits and pushes of completed, verified work to main are authorized,
including unsuccessful research. Preserve unrelated changes and remote history;
never force-push. Keep datasets, checkpoints and binaries outside git. Deliver
downloadable Markdown, primary PNGs, a resumable evidence archive and another
copy/paste prompt carrying this authorization. Do not use Pages or Spaces.
