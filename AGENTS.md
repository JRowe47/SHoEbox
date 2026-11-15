# SHoEbox development

- The selected CPU runtime is corrected routing C, exposed by `shoe_multiscale.py` and
  implemented in `multiscale/`. Preserve `shoebox_reference.py` as the historical
  cortical comparator. New public models use exact search with routing C and v10 coverage-only
  balancing. Explicit C/D controls retain v9 unless warmup-only is selected.
- Implemented architectural repairs belong in the default reference path.
  Enable compatible reference features and full credit coverage by default.
  Throughput regressions do not justify leaving a completed repair opt-in.
- Preserve explicitly selected historical kernels and saved checkpoint policies
  for reproducible comparisons. Do not silently reinterpret old stream state.
- Report effective capabilities. Experimental controls without a current
  implementation must not be presented as enabled reference features.
- Upper language-credit gains and retained-value producer credit are experimental
  controls. Numerical correctness does not establish expert specialization;
  require matched loss and active-expert permutation evidence before promotion.
- Validate changes with the relevant native regression tests, checkpoint
  continuation checks and sanitizer harness. Record throughput when measured.
- Keep trained models, datasets, compiled binaries and temporary snapshots out
  of git. Record compact source, tests and measurement summaries.
