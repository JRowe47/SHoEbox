# A10 resume and handoff

A10 is complete. Exactness passes; execution qualification fails at
76.020287% native throughput, below the unchanged 80% gate. Candidate throughput
is 11.182846% above paired scalar A9. Preserve all twelve measurements; do not
rerun timing. Research fitting, accuracy rescoring and audit access are zero.
E8 stays public. A9's development failure remains unchanged.

Read AGENTS.md, DEVELOPMENT_HANDOFF.md, the capability matrix, E8/public evidence,
all A1–A9 reports/diagnoses, original A9/A9-F1/A10 registrations, this report,
ACCOUNTING.json, pre_timing.lock.json, timing_worlds.lock.json and the complete
A11_REGISTRATION.md. Historical source/control receipts and budget constraints
are authoritative. The old A9-F1 20.25M fitting budget is exhausted.

## Retained evidence

- Starting main: `8270e0d60bbefd75f7c7c9604d5961702615aac2`.
- Pre-timing published commit: `3bc9d54f84626c3c117f88419f8cdcec0564e325`;
  tree `68e09052a14c3c09bc0d1e780475953202f3b543`. All controls were committed
  before any timing. Local original `dd1435e` has the same tree.
- Explicit backend: `experiments/row_parallel_affine/`. A9 scalar and every
  prior computational source remain unchanged. Checkpoint policy/magic remains
  A9; execution backend identity is external metadata, never saved state.
- Complete original pre-fit failure, F1 allocation, research checkpoints,
  manifests, raw outcomes and costs are under `a9_f1_work/` in the archive.
  All 18 original model-cursor pairs are in `a9_f1_work/a9/{1,7,19}/`.
- A10 controls, per-byte traces, exact frozen file hashes, generated test-only
  counter sources, source locks, raw timings and process receipts are in
  `a10_work/`. Compiled binaries are excluded; source is sufficient to rebuild.
- The stream SHA-256 is
  `2c26341b8f06acc74ebda3b1f118a6f6c32141cfa829df20de44e0965e3b6290`.
  Its 16 timing identities are opened for future exclusions. Labels/answers
  were never passed to timing and no accuracy was reported.
- The failed initial environment collector and 1.312434 CPU seconds are
  preserved. It executed zero model bytes; a /proc/cpuinfo fallback passed.
  Legacy microtest individual byte/update counters are incomplete, as recorded
  explicitly in ACCOUNTING.json; their source/assertions/full CPU are retained.

## Restore without executing research

Extract the archive, verify SHA256SUMS, then clone `SHoEbox.git.bundle` into a
new working directory. `publication.json` gives the verified final main SHA.
The `source/` directory also contains the final tracked source; the
`starting_source/` directory preserves the independent historical comparator.
Keep the two work directories outside the repository. No archive restore
command reruns a control, timing, accuracy evaluation or fit.

The existing benchmark rejects its retained timing_started.json marker. Do not
delete the marker to rerun. The source/control lock remains the original
pre-timing lock, not a retroactive lock over final report prose.

## Sole next work

A11 is a preregistration only: one fixed eight-row AVX2 affine kernel, exact
row arithmetic, zero research fitting. No A11 allocation, code, fitting,
control, timing or diagnosis has run. Execute only after reading that complete
registration, with fresh identities excluding A10, unchanged model state and
policy, complete inherited-event instrumentation before controls, committed
passing locks and only its five registered timing arms.

An execution pass cannot repair A9 accuracy or authorize promotion. Preserve
every development threshold and keep audits closed. End with one focused next
preregistration without executing another experiment. Completed verified work,
including failure, is explicitly authorized for commit/push directly to main
without confirmation; preserve unrelated changes/history and never force-push.
Keep data, checkpoints and compiled binaries outside git. Deliver findings,
verified commit link, downloadable Markdown, primary PNGs, resumable archive
and a copy/paste prompt preserving this authorization. No Pages or Spaces;
export and attach without confirmation.
