
The first native routing test passed the E8 numerical, request and A1 admission
sub-suites twice (optimized and ASan/UBSan), then failed its synthetic value
assertion because the test had retained randomly initialized value coefficients.
The artificial test now zeros those coefficients before installing its expected
bias-only values. Only the changed selection sub-suite is rerun in optimized and
sanitized builds. No candidate arithmetic changed. Original test source/log and
all CPU costs are preserved.
