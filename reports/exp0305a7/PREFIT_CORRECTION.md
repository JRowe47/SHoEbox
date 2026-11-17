# Pre-fit assertion correction

The initial A7 Python/native invocations expected one checkpoint magic occurrence.
A6 has two, in save and load. The candidate construction stopped on this source
assertion before any A7 model was allocated or fitted. Change the expected count
from one to two; both occurrences receive the registered A7 magic. Model arithmetic
remains unchanged. The failed initial backend, logs and CPU receipts are retained.
The complete corrected Python/native/sanitizer controls pass before fitting.
The initial Python invocation completed E8/A6 parity and cursor control updates;
those discarded bytes are separately charged by exact data-only reconstruction.
