# Acquisition-to-retention curriculum

November 6, 2026. Source base: `0bbe3c1`.

## Question and implementation

Can short acquisition tasks and a longer live gradient window teach useful payload representations that transfer to four-entity retention under interference?

The preceding run used 16-byte gradient chunks. Its detached trace trains write heads, but does not train historical payload encoders. This round keeps the 549,059-parameter competitive model unchanged and uses 64-byte optimization windows. Gradients follow committed payloads, their features and recurrence within each window. Conditional write-head traces continue across window boundaries. There is no replay store, frozen teacher, task identifier, supplied slot address or forced task-specific write.

All curriculum arms use these fixed stages, with all weights and AdamW moments carried forward:

| Updates | Stage | Stream |
| --- | --- | --- |
| 1-200 | Acquisition | Four entities; a short exact question after each initial observation; delays 0/2; no interfering updates |
| 201-400 | Bridge | Four entities; no extra acquisition questions; delays 2/16; interfering updates |
| 401-600 | Target | Original retention stream; delays 2/48; interfering updates |

Stage changes start new independent documents and reset only runtime state, eligibility and stream cursors. Ordinary within-stage chunks and query boundaries do not reset state. Short acquisition answers occur five bytes after the source value. They are targets only; they never appear as query inputs. Counts of source values and complete observations inside the actual gradient segment are recorded separately from nominal fact delay.

## Paired pilot

Initialization/data seeds are 29/229. Batch 8, context 64, 600 optimizer updates and 307,200 external bytes per arm. Each byte executes 60 sparse experts, giving 18,432,000 expert executions per arm. All four arms share initial model weights, parameter counts, optimizer settings, and executed byte/expert budgets.

| Arm | Training stream | Read interface | Gradient segment |
| --- | --- | --- | --- |
| Curriculum | Three stages | Stored publications | 64 bytes |
| Fresh | Three stages | Current producer candidates | 64 bytes |
| Truncated | Three stages | Stored publications | 16 bytes |
| Direct | Target throughout | Stored publications | 64 bytes |

The truncated control makes four detached forwards per update and aggregates answer/text losses over the entire 64-byte window before one optimizer step. Each boundary receives the existing delayed write-head credit once. Thus its comparison preserves examples, answer normalization, expert count and optimizer schedule. It removes longer payload/feature and recurrent gradient paths together; their individual contributions are not separated by this arm.

The curriculum also changes question density. Its direct-training comparison measures the combined data schedule; the fresh and truncated curriculum controls receive the same questions and byte order as the main arm.

Training uses AdamW 0.002, betas 0.9/0.95, weight decay 0.01, gradient norm cap 1, answer CE + 0.1 ordinary-byte CE + 0.001 routing auxiliary loss. No write penalty. Fixed target-distribution validation uses 16 documents every 100 updates and at stage ends. The best checkpoint minimizes all-answer NLL on that same target throughout; selected and final checkpoints are both reported. Stage validation separately measures acquisition progress and does not select the checkpoint.

## Evaluation and decision

Each selected/final checkpoint scores 32 held-out test and 32 longer-delay documents. Exact-value answer NLL is the primary retention comparison; report exact accuracy, corrected questions, semantic/context results, writes, read ages and costs separately. Whole-query normal-write content shuffling and an age-filtered shuffle of publications older than 16 bytes test direct use of pre-query content. Record actually changed cells and matched changed-query effects; holding writes remains a separate diagnostic. All causal forks preserve pre-query recurrence.

First compare curriculum with trained fresh on exact test NLL, then compare with matched truncation and direct training. A useful curriculum advance needs a task gain over trained fresh plus positive normal-write content sensitivity. Older stored content must have a measured task contribution to support longer-lived retention. If the pilot supports advancement, repeat curriculum/fresh on two additional paired seeds before describing the gain as reproducible. Preserve failed comparisons and selected/final discrepancies. Do not scale to 25M merely because publications are older or writes are fewer.

## Reproduction

From the repository root:

```bash
python scripts/run_task_curriculum.py --out artifacts/acquisition/29
```

The runner saves resumable checkpoints, acquisition/bridge checkpoints, fixed-target and stage histories, source-gradient coverage counts, selected/final evaluations and full query audits. Existing arms resume through the trainer's configuration checks. Use `--workers 1` for sequential execution. PyTorch CPU threads are one per arm.

For replications, change `--seed`, `--data-seed` and `--out`, and use `--arms curriculum fresh`. Separate output roots are required. Training schedule boundaries are explicit update counts, independent of the requested final budget; resuming across a boundary is tested against uninterrupted training.

The conditional confirmation pairs are fixed as initialization/data seeds 31/231 and 37/237.
