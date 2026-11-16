# EXP01-D / EXP22: constituent key/value pairing

Starting commit: fadf56f6e2cd4b48204ac57848d73e12f5d09874. Explicit minimal
corrected C v19 remains production. This registration precedes new fitting.

## Mechanism and staged budget

Keep keys and values in the same existing retained eight-byte constituent pair.
Evaluate four phase-specific learned projections before any value mixing.
Learn a local record-presence score; select the two highest scored candidates,
then match a separately decoded current query to their keys. Decode the value
with the same selection weights. Query recognition reads existing two-byte
retained pairs with an occurrence-age preference for the latest query.
All recognizers and byte projections are fitted online. Inference has no parser,
symbol constants, target labels, record indices, exact lookup, or new sequence
state. Hard source routing has no gradient; its local presence loss is explicit.

Training-only labels describe each newly observed two/eight-byte window. They
mark local query/record syntax and reconstruct the already observed key/value
bytes. They never execute a lookup or identify the requested record. This is
additional local supervision, so report it explicitly and include a no-local-
supervision control. Gate and projection training uses normalized SGD; no
optimizer sweep or optimizer moments. Scratch source views never persist.

Pilot D1: seed 7, 250K charged-byte native warmup, at most 500K continuation
bytes for one stable-content candidate; fresh grouped development worlds,
all 64 phases. Record pilot cost separately. If it fails, diagnose on development
only and register an amendment before each additional fitted mechanism.

Formal recipe is locked after development: three seeds 1/7/19, 500K nominal
total parameters with head weights deducted from banks; 250K warmup and up to
2M continuation per arm. Native, prior pooled control, selected paired readout,
and no-local-supervision control. Report actual capacity and training costs.
Fresh grouped salt SHoEbox-EXP01D-constituent-pairing-v1, RNG 20261118;
2,048 train, 512 dev, 1,024 audit presentations. No audit-based tuning.

## Gates and audit

Retain EXP01-C gates unchanged: >=95% mean native full-256-byte autoregressive
two-answer exactness, >=90% each seed; same first/second thresholds; >=90%
both-query-orders mean, >=85% each seed; all-offset mean >=95%, worst offset
seed-mean >=90%; independent +0/+64/+128/+256 delays >=90% mean and >=85%
each seed. Query ablation must drop >=20 points; reversed selection must hurt.
Audit all 64 phases on identical worlds, clone identical prefix state for query
counterfactuals, and feed generated first answers into the second query.

Require causality, native parity, exact checkpoint continuation, bounded memory,
gradient checks where differentiable, existing regression and ASan/UBSan checks.
Isolated full-distribution inference must retain >=80% native throughput.
No production promotion until joint-core, language, copy, reconstruction and
1M/5M capacity qualifications also pass. A passed binding gate is reported as a
research-stage solution, separately from default promotion.

## Primary sources

- Miller et al. (EMNLP 2016), sections 3.1–3.2,
  https://aclanthology.org/D16-1147/: distinct key/value encodings, one common
  attention weight for each associated pair; position-sensitive window encoding.
  Adaptation here reads pre-existing neural constituents, not parsed records.
- Arora et al. (ICML 2024), https://proceedings.mlr.press/v235/arora24a.html:
  multi-query associative recall and recall/throughput accounting. No new cache
  or BASED implementation is introduced.
- Wang, Shi and Fox (JMLR 2026), https://jmlr.org/papers/v27/25-0903.html:
  separating memorization from retrieval and analyzing query/key normalization.
  Motivates diagnosing source availability separately from learned decoding.

Publish compact source and positive/negative evidence. Preserve historical
reports and checkpoints; exclude binaries, datasets and fitted weights from git.
