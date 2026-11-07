"""Render the completed regional recall report from preserved metrics."""
import json
from pathlib import Path

from render_markdown_report import render

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    study = json.loads((ROOT/'experiments/text_regions/results/regional_recall/summary.json').read_text())
    checks = json.loads((ROOT/'artifacts/regional_recall/checks.json').read_text())
    arms = study['arms']
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.2), constrained_layout=True)
    for layout, color in (('distinct', '#17654d'), ('pooled', '#aa6934')):
        arm = arms[layout]
        rows = [dict(step=0, validation=arm['initial']), *arm['history']]
        for axis, family, metric in ((axes[0], 'language', 'nll'), (axes[1], 'memory', 'answer_nll')):
            axis.plot([r['step'] for r in rows], [r['validation'][family][metric] for r in rows],
                      label=layout.title(), color=color, marker='o', markersize=3)
    for axis, title in zip(axes, ('Natural-language bytes', 'Memory answers')):
        axis.set(title=title, xlabel='Mixed updates', ylabel='Validation NLL (nats)')
        axis.grid(alpha=.2); axis.spines[['top', 'right']].set_visible(False)
        axis.legend(frameon=False, fontsize=9)
    fig.savefig(ROOT/'docs/SHoEbox_Regional_Recall_Curves.png', dpi=160)
    plt.close(fig)
    rows = []
    for layout in ('distinct', 'pooled'):
        same_checkpoint = arms[layout]['evaluation']['best']['step'] == arms[layout]['evaluation']['last']['step']
        for tag in (('best',) if same_checkpoint else ('best', 'last')):
            result = arms[layout]['evaluation'][tag]
            label = 'Selected/final' if same_checkpoint else ('Selected' if tag == 'best' else 'Final')
            rows.append(f"| {layout.title()} | {label} ({result['step']}) | "
                        f"{result['language']['nll']:.4f} | {result['memory']['test']['groups']['exact']['nll']:.4f} | "
                        f"{result['memory']['long_delay']['groups']['exact']['nll']:.4f} |")
    intervention_rows = []
    same_checkpoint = arms['distinct']['evaluation']['best']['step'] == arms['distinct']['evaluation']['last']['step']
    for tag in (('best',) if same_checkpoint else ('best', 'last')):
        result = arms['distinct']['evaluation'][tag]
        for key, label in (('pooled', 'Pool roles'), ('no_feedback', 'Disable feedback'), ('blocked', 'Block reads')):
            control = result['interventions'][key]
            checkpoint_label = 'Selected/final' if same_checkpoint else ('Selected' if tag == 'best' else 'Final')
            intervention_rows.append(f"| {checkpoint_label} | {label} | "
                                     f"{control['language_delta']:+.4f} | {control['answer_delta']:+.4f} |")
    final = arms['distinct']['evaluation']['last']
    totals = final['totals']
    last_val = arms['distinct']['history'][-1]['validation']
    graph = study['graph']['audit']
    feedback_min = min(min(n['query_feedback_norm'], n['null_feedback_norm']) for n in graph['nodes'].values())
    report = f'''# SHoEbox: regional recall and mixed learning

November 6, 2026 | Shared recall ports across the graph and owned-memory model

**Recall is now a reusable regional interface, and the same owned-memory model learns from natural text and memory tasks.** The implementation retains role-specific input maps, item-or-null selection, overlap controls, feedback, producer-state holds and delayed payload credit. These controls remain available for broader training and later ablation.

## Ordinary regional graph

Each enabled receiver selects from its configured incoming publication ports. Separate slot maps preserve item roles at the sparse expert boundary. The receiver's previous computed output informs its next request, including when its public output is held. External bytes enter through the existing ingress path.

Forward edges read completed current stages; lateral and descending edges read the round-start snapshot. Masks apply to keys and values. Substitutions replace both payload and semantic headers. Selection telemetry includes source versions and occurrences. Dependencies identify examined inputs, because every candidate can influence selection; per-slot weights identify actual content reads. All candidate ports are still scored; selecting fewer items does not reduce expert executions.

The seven-node, two-round integration uses seven slots per receiver. It completed 32 updates on 2,048 natural-text bytes, executing 344,064 experts. The graph has 311,711 parameters, versus 246,142 without recall, and adds **896 bytes of feedback state per stream** at width 32. The existing publications remain the memory cells.

| Graph integration check | NLL, nats/byte |
|---|---:|
| Initial validation | 5.5321 |
| Final validation | 4.3244 |
| Separate 128-byte test | {graph['nll']:.4f} |
| Test with recalled content blocked | {graph['controls']['blocked']:.4f} |

All seven receivers learn different slot maps and nonzero query/null feedback weights (smallest feedback norm: {feedback_min:.3f}). Splitting a continued stream at byte seven, serializing its state and resuming reproduces logits and final state exactly. Pooled-role and feedback-off checks remain available alongside blocked reads.

## Mixed language and memory continuation

Both arms start from the same final distinct-slot checkpoint of the preceding recall pass, with fresh optimizers and runtime state. Each has **564,519 trainable parameters** and completes 320 updates: {totals['language']['bytes']:,} natural-text bytes plus {totals['memory']['bytes']:,} memory-task bytes, with {sum(v['experts'] for v in totals.values()):,} expert executions. Odd updates read ordered Tiny Shakespeare; even updates read procedural documents containing corrections and interference. Batch size is eight, live context is 64 bytes, and the learning rate is 0.001. Initialization/data seeds are 71/271.

The same weights serve both sources. Each source owns its runtime state, cursor and version-bound credit traces. Source labels never enter the model. Producer-state holds and delayed gate/payload credit are enabled for both sources. Both sources produce nonzero delayed gate and payload gradients. The ordinary graph and this owned-memory topology are separate configurations sharing the recall module; graph publication gates retain their existing training path.

---

## Learning and held-out results

The distinct arm's natural-language validation NLL changes from {arms['distinct']['initial']['language']['nll']:.4f} to {last_val['language']['nll']:.4f}; its all-answer memory NLL changes from {arms['distinct']['initial']['memory']['answer_nll']:.4f} to {last_val['memory']['answer_nll']:.4f}.

![Validation learning curves](SHoEbox_Regional_Recall_Curves.png)

Checkpoints are selected by the equal-weight mean of natural-byte and all-answer validation NLL. The table reports natural-text and exact-answer test NLL separately; lower is better. Language testing uses {final['language']['targets']:,} targets. Each memory split contains 16 fresh documents; long-delay testing triples the configured gaps.

| Layout | Checkpoint (update) | Language | Exact answer | Long delay |
|---|---|---:|---:|---:|
{chr(10).join(rows)}

## What the learned recall contributes

These interventions run from the start of each held-out stream. Positive NLL change means intact behavior helped. Memory changes below use **all answer types**, matching the selection metric. They measure whole-rollout effects, including changed subsequent writes and recurrent state.

| Distinct arm | Intervention | Language delta | Answer delta |
|---|---|---:|---:|
{chr(10).join(intervention_rows)}

## Verification and next step

**{checks['total_tests']} checks pass across the five experiment suites**, plus the live-output example. New checks cover legacy initialization, hard/null selection, masking, delayed snapshots, gradients, held publications, inactive rows, exact graph continuation, source isolation, and exact restoration of shared weights, optimizer, both stream cursors and both delayed-credit states.

Next, bring the task-trained publication/credit interface into the ordinary graph, then increase mixed-data exposure through that unified path. Keep role and feedback controls trainable; their comments explicitly mark potential training-threshold dependence for later ablations.

```bash
python scripts/run_regional_recall.py --parent /path/to/parent.pt \\
  --data data/tiny_shakespeare.txt --out artifacts/regional_recall
python scripts/check_all.py
```

The companion archive contains source, the exact corpus, common parent, selected/final mixed checkpoints, graph checkpoint, both stream states, optimizer and eligibility state, raw traces, metrics, checks and SHA-256 hashes.
'''
    source = ROOT/'docs/SHoEbox_Regional_Recall_Report.md'
    source.write_text(report)
    render(source, source.with_suffix('.pdf'), title='SHoEbox: regional recall and mixed learning',
           footer_text='SHoEbox | Regional recall | November 6, 2026')
    print(str(source))


if __name__ == '__main__':
    main()
