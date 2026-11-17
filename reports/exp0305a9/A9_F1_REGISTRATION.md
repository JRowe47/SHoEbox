# A9-F1: binary development allocation repair (registered, not implemented or fitted)

A9 cannot complete pre-fit registration. With the exact v9 salt, 64-key pool,
world RNG 20261201 and all mandatory exclusions, only 96 binary worlds fall in
its development bucket. The requirement is 128. Independent exhaustive
enumeration agrees with the inherited generator. No research fit, development
score, throughput measurement or audit evaluation has run. This is a data
allocation blocker, not a measured failure of selective record-key credit.

## One fixed change

Change only the allocation of the fresh **binary development** sample. Keep the
original A9 three-value train/development/closed-audit sets exactly unchanged.
Retain the entire historical opened binding-fold/E2 exclusion and every reserved
A1–A8 audit and A2–A8 development/binary world. Also exclude all 896 selected
A9 primary worlds (512 train +128 development +256 audit) from the new binary
pool. This leaves 336 eligible binary worlds before selection.

Enumerate the same 64 keys and binary values in the original order:

```python
eligible = []
for key0 in range(64):
    for key1 in range(key0 + 1, 64):
        for value0 in range(2):
            for value1 in range(2):
                if value0 == value1:
                    continue
                world = World('binding', ((key0, value0), (key1, value1)))
                if historical_A9_excluded(world.group_key):
                    continue
                if world.group_key in selected_A9_primary_groups:
                    continue
                eligible.append(world)
random.Random(20261201).shuffle(eligible)
binary_development = eligible[:128]
```

`historical_A9_excluded` means the unchanged A9 `tasks.excluded`, which already
includes A1–A8. `selected_A9_primary_groups` is the union of the unchanged
`worlds('train',512)`, `worlds('dev',128)` and `worlds('audit',256)` with three
values. The binary sample uses the entire eligible pool instead of requiring
membership in the v9 SHA-256 development bucket. Its assignment is explicitly
binary-development-only. It cannot supply a training example or an audit.
All order/operation/offset siblings remain grouped. No salt search or world
selection based on outcomes is permitted. No A9 model outcome exists.

This is the sole amendment to `../exp0305a8/A9_REGISTRATION.md`. It preserves
128 distinct binary worlds and every original threshold and sample count;
no sample reduction, duplicate presentations as replacement worlds, reopened
worlds, key-pool expansion or altered primary fold is allowed. The v9 primary
salt and RNG remain exact. This document has not changed the executable
allocation, and no F1 binary identities have been selected or scored.

## Preconditions before any fitting

Read the original A9 registration and this amendment together. Implement the
binary-only allocation and independent byte-for-byte reference. Prove the 128
selected identities are unique, disjoint from all 896 primary groups and every
historical exclusion; lock their exact order and identity hashes. Retain the
original 96-world failure, all failed/passing controls, stream/credit manifests
and all CPU receipts. Preserve the original primary world/stream hashes.

Rerun affected pre-fit checks against the final source. Require the complete
20-test A9 suite to pass, all prior Python/native/ASan/UBSan/policy/history checks,
scalar A8 parity and masked-credit isolation, causality, shortcut controls,
64-phase/block boundaries and exact partial-episode/spacer resumes. No fake
passing receipt or sample-size bypass. Commit the passing source/world/stream/
credit/correctness lock before starting the fitting driver.

Then execute only the original matched E8/A8/A9 campaign, seeds 1/7/19, nominal
500K, independent 250K native-only PAD warmups +2M continuations per arm/seed:
**exactly 20.25M research fitting bytes**. A8/A9 inputs and local labels remain
identical. A9 alone suppresses positive record-key updates in diverse blocks;
all other training/inference rules stay unchanged. No extra fit, pilot, replay,
inherited warmup, sweep or compensation for omitted updates. Preserve all-256
inference, generic/local separation, exact cursors and disabled FMA.

All original primary/position/offset/order/delay/lesion/binary gates apply,
including **>=80% isolated native throughput** on one pinned CPU, three
interleaved 65,536-byte repeats and 1,024-byte warmups, with all other jobs
stopped. Account for executed and omitted gradients/comparisons, all bytes,
failures, repeated correctness and whole-process CPU. Lock fits, checkpoints,
cursors and every full outcome before the registered A8/A9 frozen diagnosis.
Keep subset teacher-fed measurements separate from full autoregression.

Failure retains E8 and closed A1–A9 audits and requires one focused next
preregistration. Passing development requires a separate fresh generic/capacity
qualification with every original margin. No gate is relaxed.

Completed verified work, including unsuccessful research, is explicitly
authorized for commit and push directly to main without further confirmation.
Preserve unrelated changes and remote history; never force-push. Keep datasets,
checkpoints and compiled binaries outside git. Deliver ordinary downloadable
Markdown, primary PNGs, a resumable archive and another copy/paste prompt
preserving this authorization. No Pages or Spaces; export without confirmation.
