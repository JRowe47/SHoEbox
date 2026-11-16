# EXP01-E2: make the learned fallback independent of native decoder drift

Registered after E1 DEVELOPMENT-only retention failure; no E1 binding audit has
been opened. E1 reaches 99.02% development exactness, but 1M generic FineWeb
updates erase native binding. Gate-forced-on does not repair it. Restoring the
old residual plus gate on the language-trained core recovers only 27.54% exact
(30.27% with gate forced on); restoring only the old core recovers 56.25%.
Both output branches drift. This is a measured integration failure, not missing
constituents or a reason to add slots.

E2 changes the learned residual to `g * (readout_logit - native_logit)`.
The resulting logit is `(1-g)*native_logit + g*readout_logit`. The readout learns
an absolute output, and a gate near one can shield a retrieval answer from
native decoder drift. Core gradients multiply by `(1-g)`; head gradients by
`g`; gate gradients use their logit difference. Same 65 gate weights, 9,965
budgeted readout weights, no persistent sequence state. No gradient through
source selection. Keep E1 and all D policies/checkpoints unchanged in their
own modules; E2 uses a new checkpoint magic. This is a logit switch inspired
by See et al. (ACL 2017), not their probability-mixture pointer architecture.

Pilot seed 7: fresh grouped salt `SHoEbox-EXP01E-switch-v2`, RNG 20261121;
exclude D2/D3 and E1 primary audit worlds from all splits, independent of scores.
Same 2048/512/1024 presentations, 250K warmup, 2M continuation, every-byte
loss and D3 local labels. Then 1M pinned FineWeb targets, no local labels;
development curves and complete binding/offset/delay/intervention tests before
and after. Total additional pilot fitting: 3.25M charged bytes.

No threshold changes, no audit access for this pilot. A passing development
revision needs three-seed native/D3/E1/E2 matched fresh-fold qualification and
all production checks; an E2 failure remains in the ledger. Do not transfer
old residual weights to the absolute-output policy as if they had that meaning.
