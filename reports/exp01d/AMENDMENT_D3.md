# D3: teach constituent readout independence from neighboring framing

D2 source snapshot 6ff405b. Its audit passed native binding (99.41%), both query
orders (98.44%), all offsets (100% on the registered 16-world sweep), causal
ablations, correctness, and throughput (90.77% of native). It failed the
independent full-block delay gate: about 86% exact, below 90%.

Additional DEVELOPMENT-only diagnostics find current-query decoding still 100%
at +64/+256, but both-key reconstruction falls from 87.5% to 40.625%. Padding
changes the non-record bytes inside a retained eight-byte pair after the final
record. D2's local projection training saw a fixed adjacent query frame there.
The source is present; its learned decoder depended on neighboring context.

D3 changes training coverage, not the runtime, optimizer, parameter count or
memory. Independently insert 0..7 PAD bytes after each complete record during
training. These short spacers vary every outside-record position of an eight-
byte pair. Adjust leading padding to keep all 64 answer phases balanced.
No +64/+128/+256 delay appears in training. All bytes, including spacers, are
charged. Local labels are still only already observed window contents.

Pilot: seed 7, 250K warmup plus at most 2M continuation, report development
curves and development-only offset/delay results. No D3 audit access.
Formal: seeds 1/7/19; native and pooled controls trained with short spacers,
the D2 paired recipe without spacers, and D3 paired recipe with spacers.
Each branch gets 2M continuation; native, pooled and paired capacities each
have their own 250K warmup. Paired arms share the same reduced-core warmup.
Warmup for the paired comparison is unspaced; the two head continuations
differ only in short spacers. Formal total: 26.25M charged bytes.

Fresh grouped salt SHoEbox-EXP01D-context-variation-v2, data RNG 20261119;
2,048 train / 512 development / 1,024 audit presentations. Exclude all worlds
from the already opened D2 primary audit from the new audit. Never exclude
based on success/failure. All siblings remain in the same split. Lock selection
before opening the new audit; preserve all original gates and thresholds.

Repeat positive controls on the new fold: 250K bytes per seed, separately
charged. Previously opened D1/D2 audits remain historical evidence, not D3's
qualification. No default promotion without remaining regression qualification.
