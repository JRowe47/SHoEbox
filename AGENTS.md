# SHoEbox development

- The selected CPU runtime is routing D, exposed by `shoe_multiscale.py` and
  implemented in `multiscale/`. Preserve `shoebox_reference.py` as the historical
  cortical comparator. New public models use exact search with routing D.
- Implemented architectural repairs belong in the default reference path.
  Enable compatible reference features and full credit coverage by default.
  Throughput regressions do not justify leaving a completed repair opt-in.
- Preserve explicitly selected historical kernels and saved checkpoint policies
  for reproducible comparisons. Do not silently reinterpret old stream state.
- Report effective capabilities. Experimental controls without a current
  implementation must not be presented as enabled reference features.
- Validate changes with the relevant native regression tests, checkpoint
  continuation checks and sanitizer harness. Record throughput when measured.
- Keep trained models, datasets, compiled binaries and temporary snapshots out
  of git. Record compact source, tests and measurement summaries.
