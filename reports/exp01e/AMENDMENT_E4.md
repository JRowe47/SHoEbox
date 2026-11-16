# EXP01-E4: calibrate the existing learned byte code as a complete distribution

Registered from E2/E3 development diagnostics. E2's restored readout with its
gate forced on recovers 99.41% restricted two-value exactness but 0% native
256-byte exactness. E3's explicit probability-mixture transfer starts at only
36.52% native exactness before language fitting, so it is not a successful
mixture-training experiment; its conditional fresh-fit Pilot B is not triggered.
The measured issue is full-alphabet calibration and subsequent decoder drift,
not lost keys, values or constituent pairing.

D3 already reconstructs each observed value byte as eight independent signed
bit coordinates with targets +/-0.7. E4 uses their original scale to form
`p(bit=1)=clip(0.5+0.5*h/0.7, 1e-6, 1-1e-6)` and multiplies the eight Bernoulli
probabilities to define all 256 byte probabilities. No byte, symbol, query,
record index or task alphabet is tested by inference. This is a generic
continuous byte-code decoder, not a lookup parser. Its 2,295 redundant residual
output weights are removed. The learned gate retains E3's normalized mixture
and posterior responsibility credit for the native core and applicability.
No generic target rewrites the bit-code decoder through a separate output head;
local reconstruction still explicitly trains the existing key/value projections
when causal local labels are supplied. No generic key/value learning is claimed.

Readout parameters: 7,605 local presence/key/value weights +65 applicability
weights =7,670. No optimizer moments or persistent sequence state. The fixed
calibration follows the existing +/-0.7 target convention, not a threshold sweep.
Normalized next-byte mixture loss trains core/gate on generic streams.

Pilot A: explicit E2 weight conversion strips its unused output weights, records
both hashes, and preserves its inherited 2.25M binding fit. Evaluate complete
E2 DEVELOPMENT binding gates, then 1M pinned FineWeb targets without local
labels and repeat all development gates. Separately charge 1M new bytes.
If both initial and retained development primary exactness are >=95%, run
Pilot B from scratch: nominal 500K total, E2's unopened grouped folds, 250K
native warmup +2M joint continuation +1M generic language. Register its source
hashes before fitting. No E2/E3/E4 audit is opened by these pilots.

No gates are weakened. Three-seed fresh qualification, independent language /
copy / reconstruction regressions, 1M/5M capacity and >=80% inference throughput
remain necessary for E4 production promotion. Keep all previous policies and
failed fits. Literature: See et al., ACL 2017 §2.2 for a learned mixture switch;
Miller et al., EMNLP 2016 for common weights on paired key/value encodings.
