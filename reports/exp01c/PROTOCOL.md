# EXP01-C / EXP22 preregistered protocol

Starting source: bda7bc8. Production baseline: explicit minimal corrected C v19.
Register this file and source hashes before any pilot fitting. No promotion or
production runtime change is implied by this experiment.

## Hypothesis and views

Separate the query representation from key/value evidence, without adding
sequence state. Retain the EXP01-B rank-16, five-lane selector and native
256-byte residual decoder (9,535 fitted parameters). Compare:

1. `native`: full-budget minimal v19.
2. `l3`: unchanged EXP01-B shared-L3 local-credit head, frozen reduced core.
3. `split`: query from a recursively completed *provisional* upper constituent;
   key/value heads from the last completed L3 state. Provisional composition
   reads only valid current children; it never commits a new constituent.
4. `pool`: five learned, separate views over existing retained stable ordered
   pairs, current completed constituents and valid partial children. Soft pooling
   uses each lane's first projection row, tied to that row's decoding role, so
   there are no extra weights. A second query/key selection returns a value.

The pool's 60 content coordinates and four structural coordinates (scale, bounded
occurrence age, source type and partial-child role) contain no task symbols,
parsed record positions, query labels, or answer labels. Source sets are bounded
by the existing four scales and 16 retained pairs per scale. Pooling is a new
readout of existing state, not additional stored records. This experiment does
not change write/admission policy. All candidate cores freeze after warmup.

## Data, budgets and selection

- Seeds: 1, 7, 19. Nominal total capacity 500,000. Deduct 9,535 from candidate
  core requests, report actual allocated weights and discrete-bank rounding.
- Fresh grouped fold salt: `SHoEbox-EXP01C-query-access-v1\0`; data RNG 20261117.
  Train/dev/audit buckets 60/20/20. All record/query-order and offset siblings
  remain together. Two distinct keys from 48; two distinct binary values.
- 2,048 train presentations, 512 dev presentations, 1,024 audit presentations
  before offset augmentation; four balanced order variants per concrete world.
- Training cycles through all 64 leading padding offsets. Leading padding shifts
  hierarchy phase without changing record-to-query distances. Extra full-block
  delays are excluded from training. Charge all padding and observation bytes.
- Each core: 250,000 warmup bytes. Shared reduced warmup for the three candidates;
  separate full native warmup. Each formal branch: 2,000,000 continuation bytes.
  Evaluate development at 500K, 1M, 1.5M, 2M. Use final checkpoints only.
- Pilot: seed 7, one 250K reduced warmup, then 500K each for l3/split/pool,
  fixed head rate .16 and auxiliary weight 1. Maximum pilot budget 1.75M bytes.
  Pilot checks viability; no audit access. Any design amendment requires a new
  registered document and separate charged costs before further fitting.
- Formal actual budget: 25.5M bytes (24M continuation plus 1.5M warmup).
  Positive control: record-encoded learned attention, 250K bytes/seed, separately
  accounted 750K bytes; inference oracle features disqualify native comparisons.
- Select among split/pool by final mean development native exactness; tie break
  lower inference cost. Audit every registered arm once; no audit-based tuning.
  Software tests are correctness checks, separately logged, not fitted pilots.

## Endpoints and interventions

Primary: full-256-byte autoregressive *two-answer episode* exactness over a
balanced 64-offset audit. Report each seed, first and second answer separately,
teacher forcing, restricted-alphabet diagnostic, and NLL. Report accuracy and
NLL against charged training bytes and CPU seconds.

Also report offset zero and all 64 offsets on identical 64 presentations / 16
worlds per seed. Independent +0/+64/+128/+256-byte delays are inserted *after*
records, preserving answer phase. Evaluate every delay across all offsets on
16 presentations / four worlds per seed, independently of the primary audit.

Frozen counterfactuals clone an identical record-prefix state, then swap only
query order, preserving leading padding. Score both first answers and all four
answers. Native inference receives bytes/state only. Mechanism ablations:
no head, no query, zero source, uniform/reversed final selection, no completed
sources, no partial sources, and uniform source pooling. Auxiliary labels derive
only from already observed prefix bytes and never include a selected-record index.

## Acceptance gates (all required for promotion)

- Mean primary native binding exactness >=95%; every seed >=90%.
- First and second answer accuracy each >=95% mean and >=90% per seed.
- Frozen both-query-orders exactness >=90% mean, >=85% every seed.
- Across-offset mean >=95%; worst offset (seed mean) >=90%.
- Independent delay mean at every tested delay >=90%, no seed <85%.
- Query removal drops exactness >=20 percentage points; reversed selection
  reduces accuracy. No answer-label or oracle-index path into inference.
- Gradient checks, causal prefix/future independence, bounded source inventory,
  exact checkpoint continuation, old native regression and ASan/UBSan pass.
- Candidate isolated inference throughput >=80% of native; no added persistent
  sequence state; report scratch, normalizers, weights and checkpoint bytes.
- Only if these gates pass: register and run joint-credit, language/copy/
  reconstruction, 1M/5M capacity regressions before production promotion.
  Until then those capability regressions are explicitly pending.

Publish implementations and negative results. Preserve old checkpoints and
minimal v19 defaults. Keep datasets, model checkpoints and binaries out of git.

## Primary literature and intended adaptation

- Miller et al., EMNLP 2016, section 3,
  https://aclanthology.org/D16-1147/: distinct addressing and returned-value
  encodings. Here all inputs are existing neural constituents, not parsed records.
- Dai et al., ACL 2019, sections 3.2–3.3,
  https://aclanthology.org/P19-1285/: access to completed context across segment
  boundaries. Here existing retained/partial state replaces a new attention cache.
- Sun et al., 2023, section 2.1,
  https://arxiv.org/abs/2307.08621: separation of within-chunk contribution and
  recurrent history motivates the provisional/completed comparison. We do not
  claim equivalence to RetNet or change SHoEbox's recurrence.
- Yang et al., ICLR 2025,
  https://openreview.net/forum?id=r8H7xhYPwz: distinguishes memory erasure from
  targeted correction. Defer write-rule changes until readout failure versus
  evidence loss is identified; no delta-rule memory is added here.

All numerical claims in the resulting report must come from this experiment.
