# Regional runtime profiling

Matched CPU measurements compare the optimized runtime with the untouched source at `10f87459377d5d2062c51be2c14fa8feaf67ebd0`. Three measured pairs follow one warmup per backend; the order alternates. The same checkpoint or seeded weights and workloads are used by both implementations.

`measurements.json` contains individual timings, median speedups, configurations, hardware/runtime metadata, and output/state hash comparisons. Text inference measures supplied-byte processing. `checkpoint_verification.json` records frozen-model equivalence, gradient error and three matched joint updates. The text profiles are instrumented diagnostic samples; reported speedups use the uninstrumented timings.

The downloadable bundle includes the selected checkpoint and binary cProfile files. See `docs/SHoEbox_Runtime_Profiling_Report.md` for the interpretation and reproduction commands.
