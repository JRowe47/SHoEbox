# SHoEbox composition and credit repair

12 November 2026

## Result

The repaired runtime implements explicit constituent identities, learned event boundaries, separate content and context, and credit through upper-region retained input. All 19 automated checks pass. Two fresh 1M-parameter runs, each trained on 1M byte targets, reduce mean held-out loss from 2.060284 to 2.035142 nats/byte (1.22% lower).

| Seed | Original held-out NLL | Repaired held-out NLL |
| --- | --- | --- |
| 1 | 2.069381 | 2.040394 |
| 2 | 2.051188 | 2.029891 |
| Mean | 2.060284 | 2.035142 |

In six interleaved trials per model on the same CPU core, median training throughput was 22,564 bytes/s for the original and 17,098 bytes/s for the repair: 24.2% lower throughput, or about 14.2 additional seconds per million bytes at these measured rates. The new dictionary, event and credit state uses 2.96 MiB, independent of expert count. Neural parameter counts are unchanged.

## Implemented changes

| Component | Delivered behavior |
| --- | --- |
| Reusable identity | A separate local dictionary records ordered constituent IDs and relative geometry. IDs survive context changes, byte shifts and table eviction; recognition confidence reflects observed recurrence. |
| Events and boundaries | A1/B1 emit byte atoms. Regions L2 through G5 consume each primary child event once and accumulate up to eight children. Previously learned transition strength determines update, hold and completion. |
| Content and context | Pure constituent codes form the recognition/routing input and future-content teacher. Child neural outputs, feedback and other regional ports remain separately weighted context. |
| Unrecognized input | A singleton transparently carries its identity, context and original neural producer. It does not execute an upper expert or occupy neural eligibility. Unseen combinations of familiar compounds are explicitly provisional. |
| Learning and credit | Dictionary recurrence learning is independent of forecasting. Error-driven teaching multipliers now scale expert, router and relevant state updates together. Retained-input credit follows the input-state Jacobian. |
| Memory and execution | EC stores transitions between stable content cues. Raw A1 clock cues are excluded from EC in this mode. Y still predicts every byte. Local forecasting retains the original one-in-four neural-event schedule. |

A pattern slot identifies an exact ordered composition. Neural experts learn how that content and its context predict subsequent observations. A bounded four-way associative dictionary provides 2,048 pattern entries per region, independently of compute experts; two additional bounded tables hold transition numerators and denominators. The default recognition threshold is three observations and the transition continuation threshold is 0.35. Content IDs are 128-bit structural fingerprints; timestamps and absolute byte offsets are excluded, while child order and relative geometry are included.

## Retained-input credit

The retained state already affected forward predictions. The missing part was its learning dependency. For upper regions, m_t = (1-alpha)m_(t-1) + alpha(q_t+c_t). The repair sends the retained contribution to the prior INPUT state, bypassing the prior expert's output Jacobian. A fabricated self-output edge would teach the wrong computation. Transparent event relays retain the actual neural producer, so held context receives credit at its source. In seed 1, 8,277,023 retained-input credit deliveries were recorded and 8,046,714 forwarding operations completed, with 0 expired retained-input edges and 0 overwritten input-credit accumulators.

Eligibility remains bounded to 16 neural publications per region and three learned credit hops. Ordinary source-publication lookup misses still occurred (867,683 over the 1M-target run); the runtime counts them explicitly. The new path repairs the absent recurrence dependency, while retaining the existing delayed local-learning approximation and finite eligibility budget.

## Verification and regional contribution

The 19 checks cover stable IDs and vectors across context/byte alignment, hold behavior, ordering, explicit unknown status, held-out combinations of familiar constituents, contiguous nonduplicated spans, frozen-evaluation isolation, exact checkpoint resume, batch independence, target independence, complete teaching-scale application, retained-input Jacobians against finite differences, correct source events, legacy parity, policy validation, document reset, transparent neural provenance and source ablation. AddressSanitizer and UndefinedBehaviorSanitizer also pass for training, evaluation, draining, reset and state import/export. The 5M build passes training and exact save/resume; the 100,198,144-parameter build passes a 2,400-byte training smoke check with finite weights.

The following seed-1 interventions report change in held-out NLL. Positive values mean the intervention damaged prediction.

| Intervention | Original delta | Repaired delta |
| --- | --- | --- |
| Replace Y with its fitted mean | +0.089392 | +0.156742 |
| Remove L3 output | +0.069705 | +0.381476 |
| Remove E4 output | +0.000418 | -0.000007 |
| Remove G5 output | -0.000159 | -0.000002 |

The L3 lesion causes a much larger loss increase after repair, and replacing the entire Y decoder feature with its fitted mean also hurts more. These results show increased use of the repaired lower hierarchy. E4 and G5 still have negligible measured influence after this short run. Their predictive usefulness remains an open training result; the repair does not claim that every higher region is already doing useful language work.

## Method

Both architectures use 18 regions, width 64 and 12 experts per region (1,023,744 neural parameters). Seeds 1 and 2 start from the same respective neural initializations and fixed transforms. Each receives the same first 1,000,000 next-byte targets from the existing Tiny Shakespeare training split. The full 111,539-target held-out split is evaluated at initialization and every 100,000 targets using fresh stream state and frozen learned dictionaries. The learning rate follows the existing cosine schedule from 0.08 to 0.008 over 1M targets; local rate is main/3 and prototype rate is 0.002. Training uses periodic mode with one-in-four local teaching on actual neural executions. No structural labels or word boundaries are supplied to these language runs. The separate structural tests use a staged synthetic curriculum to establish familiar lower constituents before testing new combinations.

Timing uses an Intel Xeon Platinum 8573C CPU, no GPU, one core, six alternating-order trials per architecture, 32,768 training targets per trial and a 1,024-target warmup from an independent snapshot. It includes forward computation, routing and learning updates, and excludes compilation, snapshot construction, checkpoint I/O and evaluation. Raw whole-run timings are also preserved, but the interleaved same-core measurement is the throughput comparison.

## Running and continuing

The default `fresh_model()` selects the repaired `composition` kernel. `fresh_model(kernel="native")` retains the original baseline. `finish_document()` drains pending constituents without adding input bytes or synthetic targets; `reset=True` then clears working context. `snapshot_copy()` retains the complete stream and `frozen_copy()` starts a fresh stream while preserving fitted dictionaries. Checkpoint version 28 includes composition-state ABI 2. Legacy weights can be explicitly converted with the supplied migration script, which starts new constituent, working-memory and hippocampal state. The package supplies trained repaired 1M checkpoints. Full 5M/100M language training was not performed.

```bash
python -m unittest discover -s tests -v
python src/benchmark.py --kernel composition --seed 1 --core 0 --tag rerun
python src/migrate_checkpoint.py old_checkpoint.npz new_checkpoint.npz
```

Set `SHOEBOX_EXPERTS=12`, `64` or `1312` before import for the 1M, 5M or uniform 100M configuration. This repair targets width 64 and uniform regional banks. The README contains the document-boundary API and conversion details. `SHA256SUMS.json` identifies every packaged file, and the result JSON files retain source-data and checkpoint hashes.

Use the delivered source and repaired checkpoints for the next continuation. Keep the structural tests as invariants, retain loss and throughput tracking, and repeat the higher-region contribution measurements after additional training. The original long-trained models require explicit conversion before using the new content and event semantics.
