# Diagnostic freeze boundary correction

The first diagnostic attempt completed padding measurements, then stopped in
its repeat16 freeze trace because the final record could straddle the start of
the distractor span. Freezing the cache before that window completed prevented
the trace from capturing its signature. This is a measurement error, not a
fitted failure. Preserve that run, source lock, partial results and process CPU.

Start the externally timed freeze at the first four-byte completion boundary
at or after the distractor start (0–3 neutral bytes later). This admits the
complete record-containing window before freezing further cache replacement.
Keep constituent frontier updates active throughout. Re-run the full fixed
matrix under a new source lock; charge all repeated inference. No fitted bytes,
new settings, audit access or production numerical change.
