# SHoEbox: acquisition-to-retention

November 6, 2026. Implementation `3d63e23`; preceding retention base `0bbe3c1`.

The acquisition curriculum and longer gradient window are implemented and tested. In the completed pilot, stored publications reach **1.183 exact-answer NLL versus 1.199 for trained fresh outputs**. Direct sensitivity to older stored content remains very small. Additional seed confirmations became inaccessible when the execution workspace went offline; they are not counted as completed results.

## Completed pilot

NLL is nats per exact answer byte. Selected checkpoints minimize all-answer NLL on the same target validation distribution throughout. Each test evaluation uses 32 independent documents and 216 exact questions.

| Arm | Selected step | Exact NLL | Exact accuracy | Final exact NLL | Long NLL |
| --- | --- | --- | --- | --- | --- |
| Curriculum, 64 bytes | 500 | 1.183 | 40.7% | 1.176 | 1.208 |
| Trained fresh | 600 | 1.199 | 38.4% | 1.199 | 1.185 |
| Curriculum, 16 bytes | 500 | 1.226 | 38.0% | 1.185 | 1.360 |
| Direct target | 600 | 1.190 | 37.0% | 1.190 | 1.228 |

The main arm's exact-NLL gain over fresh is 0.015798 at the selected checkpoints and 0.022853 at the final budget. Long NLL uses selected checkpoints with tripled configured delays. Final longer-delay NLL is 1.180945 for the main arm versus 1.185395 for fresh.

All arms have 549,059 parameters, 600 optimizer updates, 307,200 training bytes and 18,432,000 sparse expert executions. Initialization/data seeds are 29/229. Pilot training plus validation took 1,462-1,473 seconds per arm with four concurrent CPU jobs.

## What changed

Training uses 200 acquisition updates, 200 bridge updates with interference and delays 2/16, then 200 target updates with delays 2/48. The stream contains four changing entities, four possible values, corrections and context questions. Inputs remain ordinary bytes; answers and provenance stay outside the model.

A 64-byte live graph trains acquisition payloads, features and recurrence together. The matched truncation control detaches every 16 bytes but aggregates losses across the full window before one optimizer update. The fresh control reads current producer candidates. Direct training stays on the target stream; its comparison includes the curriculum's change in question density.

Weights and optimizer moments persist across stages; new independent documents restart runtime state. Actual full-observation coverage in the pilot bridge stage rises from 6.8% with truncation to 41.3% with the longer window. Immediate acquisition reaches 64/64 validation questions by update 100.

## Stored-content checks

Forks begin before the entire query and preserve earlier recurrent state. Normal query-time writes continue. Positive NLL delta means changing stored content worsened prediction.

| Main-arm intervention | Selected delta | Final delta |
| --- | --- | --- |
| Shuffle pre-query publication content | +0.000353 | +0.000523 |
| Shuffle content older than 16 bytes, changed-query subset | +0.000148 | -0.000138 |

The selected older-content intervention changed 212 slots across 159 exact questions. Its small effect changes sign at the final checkpoint. The pilot demonstrates acquisition learning and a small task gain; reproducible usefulness of older publications remains open.

## Verification and handoff

All **497 checks passed** across the five preserved suites. Regional modules ran in separate processes after the combined process stalled. New checks cover payload-gradient connectivity, identical forward values and budgets under truncation, answer isolation, and exact resume across stage changes and output directories.

Confirmation pairs 31/231 and 37/237 were last observed at update 300. Both pairs had already reached 64/64 immediate acquisition questions, but their final held-out evaluations and causal audits were not completed or verified. The completed pilot's checkpoints, raw traces, logs and hashes were preserved before the workspace outage.

Next: finish paired confirmation, then extend delayed credit into the stored payload representation. Historical payload learning still stops at the live gradient window; the detached eligibility trace trains write heads only. Keep the trained-fresh and content-intervention controls.

Reproduction commands are in [SHoEbox_Acquisition_Protocol.md](SHoEbox_Acquisition_Protocol.md). Completed pilot evidence is summarized in [pilot_summary.json](../experiments/text_regions/results/task_acquisition/pilot_summary.json).
