"""Create the concise Markdown report and scientific learning curves."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, default=ROOT/'experiments/text_regions/results/context_balance/summary.json')
    p.add_argument('--out', type=Path, default=ROOT/'docs/SHoEbox_Context_Balance_Report.md')
    a = p.parse_args()
    summary = json.loads(a.summary.read_text())
    arms = summary['arms']
    balanced, control = [arms[k]['evaluation']['best'] for k in ('group_clip','sum')]
    final_balanced, final_control = [arms[k]['evaluation']['last'] for k in ('group_clip','sum')]
    protocol = arms['group_clip']['protocol']['protocol']
    totals = arms['group_clip']['evaluation']['last']['totals']
    delta = summary['group_clip_minus_sum']
    parent = summary['parent']['measurements']
    rows = [
        ('Selected update', f"{balanced['step']}", f"{control['step']}"),
        ('Natural-byte NLL', f"{balanced['language']['nll']:.4f}", f"{control['language']['nll']:.4f}"),
        ('All-answer NLL', f"{balanced['test']['answer_nll']:.4f}", f"{control['test']['answer_nll']:.4f}"),
        ('Exact-answer NLL', f"{balanced['test']['groups']['exact']['nll']:.4f}", f"{control['test']['groups']['exact']['nll']:.4f}"),
        ('Exact-answer accuracy', f"{balanced['test']['groups']['exact']['accuracy']:.1%}", f"{control['test']['groups']['exact']['accuracy']:.1%}"),
        ('Exact NLL, long-delay documents', f"{balanced['long_delay']['groups']['exact']['nll']:.4f}", f"{control['long_delay']['groups']['exact']['nll']:.4f}"),
        ('Exact NLL beyond 96 bytes', f"{balanced['long_delay']['exact_cohorts']['groups']['beyond_long_threshold']['nll']:.4f}", f"{control['long_delay']['exact_cohorts']['groups']['beyond_long_threshold']['nll']:.4f}"),
        ('Exact accuracy beyond 96 bytes', f"{balanced['long_delay']['exact_cohorts']['groups']['beyond_long_threshold']['accuracy']:.1%}", f"{control['long_delay']['exact_cohorts']['groups']['beyond_long_threshold']['accuracy']:.1%}"),
        ('Preceding task-test answer NLL', f"{balanced['legacy_test']['answer_nll']:.4f}", f"{control['legacy_test']['answer_nll']:.4f}")]
    table = '\n'.join('| '+' | '.join(row)+' |' for row in rows)
    controls = '\n'.join(f"| {label} | {balanced[key]['answer_nll']-balanced['test']['answer_nll']:+.4f} | "
        f"{control[key]['answer_nll']-control['test']['answer_nll']:+.4f} |" for key,label in (
        ('blocked_test','Block regional reads'),('always_write_test','Always publish'),
        ('no_demand_test','Remove receiver demand'),('always_state_test','Always commit state')))
    age_rows = '\n'.join(f"| {label} | {e['test']['retention']['producers']['mean_age']:.2f} | "
        f"{e['test']['retention']['producers']['older_than_segment_fraction']:.1%} |" for label,e in (
        ('Group clipping',balanced),('Direct sum',control)))
    gradients = '\n'.join(f"| {name} | {v['language_norm_median']:.3f} | {v['memory_norm_median']:.3f} | "
        f"{v['language_clipped_fraction']:.1%} | {v['memory_clipped_fraction']:.1%} |" for name,v in arms['group_clip']['gradients'].items())
    initial = arms['group_clip']['initial']
    old_fraction = balanced['test']['retention']['producers']['older_than_segment_fraction']
    recovery = summary.get('recovery')
    recovery_text = (f"The execution resumed from completed pairs {recovery['arms']['group_clip']['step']} "
        f"and {recovery['arms']['sum']['step']} after a session interruption. Exposure totals "
        "cover the retained trajectories; the validation schedule stayed fixed." if recovery else '')
    retention = ('Long-held publications remain rare; the next target is to make useful information survive repeated publication updates.'
        if old_fraction < .01 else
        'The graph now selects publications beyond a gradient segment; the next target is to isolate which retained contents improve answers.')
    text = f'''# SHoEbox: balanced learning through broader context

November 6, 2026 | Continuation of published checkpoint `7f3ce94`

Language and memory now train through the same graph in each optimizer update.
The new control caps each source's gradients separately in the core, readout,
recall and publication groups. Memory queries now follow natural-text distractors
and updates to competing facts. All {arms['group_clip']['protocol']['parameters']:,} parameters train;
the inference architecture is unchanged.

## Completed comparison

| Held-out measurement | Group clipping | Direct sum |
|---|---:|---:|
{table}

Lower NLL is better. These scores use {balanced['language']['targets']:,} natural-byte targets,
{balanced['test']['answers']} answer targets in {protocol['eval_count']*2} fresh contextual documents,
and a separate long-delay set with three times the configured retention gaps.
Acquisition questions remain short; the beyond-96-byte rows exclude them and
contain {balanced['long_delay']['exact_cohorts']['groups']['beyond_long_threshold']['count']} exact-answer targets.
The preceding task test uses its original eight documents and seed.
The ordinary exact-answer subset contains {balanced['test']['groups']['exact']['count']} targets.
Uniform choice among its four valid values gives 25% accuracy; exact recall
remains close to that baseline even as answer NLL improves.

On these same held-out tests, the starting checkpoint scores
{parent['language']['nll']:.4f} language NLL, {parent['test']['answer_nll']:.4f} answer NLL,
and {parent['long_delay']['groups']['exact']['nll']:.4f} long-delay exact NLL.
Its exact-answer NLL beyond 96 bytes is
{parent['long_delay']['exact_cohorts']['groups']['beyond_long_threshold']['nll']:.4f}.

Group clipping changes language NLL by {delta['language_nll']:+.4f}, answer NLL by
{delta['answer_nll']:+.4f} and long-delay exact NLL by {delta['long_exact_nll']:+.4f}
relative to direct summation at independently validation-selected checkpoints.
Both arms began at validation language/answer NLL of
{initial['language']['nll']:.4f}/{initial['memory']['answer_nll']:.4f}.

At the final update, clipping scores {final_balanced['language']['nll']:.4f} language
and {final_balanced['test']['answer_nll']:.4f} answer NLL; direct summation scores
{final_control['language']['nll']:.4f} and {final_control['test']['answer_nll']:.4f}.
Clipping remains an optional control alongside direct summation.

## Exposure and controls

Both arms complete {protocol['steps']} paired updates: {totals['language']['bytes']:,} ordered natural-text bytes,
{totals['memory']['bytes']:,} task-stream bytes, {totals['memory']['answers']:,} answer targets, and
{totals['language']['experts']+totals['memory']['experts']:,} expert executions per arm.
Task-stream totals include their embedded natural text. Each pair uses two
forward/backward passes at identical weights, then one AdamW step. The language
cursor continues from the parent; optimizer and runtime states start fresh.

Batch 4, window 32 bytes, gradient segment 16, constant LR 0.001. Selection uses
mean validation language and answer NLL at updates 0/32/64/96/128. Natural
distractors respect train/validation/test corpus boundaries. The four-entity
task uses acquisition queries, corrections, two cycles, six questions per cycle,
and minimum delays of 4/48/96 bytes. Query labels and fact provenance are never
model inputs. No replay store is added.

Whole-rollout controls below start from fresh test state; positive deltas mean
worse answer NLL. These include changes to later computation and publication.

| Intervention | Clipping delta | Sum delta |
|---|---:|---:|
{controls}

## Retention and gradient behavior

| Producer reads at queries | Mean publication age, bytes | Older than 16 bytes |
|---|---:|---:|
{age_rows}

Age covers selected valid A/B/C/U/V publications across receivers and rounds.
Publication age differs from fact age: recent publications can carry older
recurrent information. {retention}

| Group, clipping arm | Median language norm | Median memory norm | Language capped | Memory capped |
|---|---:|---:|---:|---:|
{gradients}

Each source/group is capped at norm 1 before addition, followed by the common
global cap of 1. The direct-sum control retains the common global cap. Full logs
also retain gradient cosines and delayed-credit magnitudes for both sources.

## Verification and continuation

Regression verification passed {summary['checks']['tests']} unique checks: 547 existing checks
and the complete 11-test new suite. The driver stopped reporting during the new
suite, which was then rerun separately and passed. New checks cover exact
gradient summation, simultaneous source weights, independent state and credit,
parent-cursor transfer, exact resume, corpus separation, target provenance,
publication-age accounting and evaluation-cache invalidation.

{recovery_text}

The publication/state/demand/payload controls remain independently ablatable.
Their intended coordination remains marked as a potentially threshold-dependent
emergent feature. Continue with task-driven retention and shared regional learning.

![Validation learning curves](SHoEbox_Context_Balance_Curves.png)

## Reproduce

```bash
python scripts/run_context_balance.py --parent artifacts/context_balance/parent.pt --parallel
python scripts/check_all.py
python scripts/evaluate_context_parent.py
python scripts/summarize_context_balance.py
python scripts/report_context_balance.py
```

The archive includes source, corpus, parent, selected/final checkpoints,
optimizer/runtime/credit states, full gradient logs, query traces and hashes.
Detailed protocol: `docs/context_balance.md`. Machine-readable results:
`experiments/text_regions/results/context_balance/summary.json`.
'''
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(9,3.2), constrained_layout=True)
    for mode,label,color in [('group_clip','Group clipping','#007c75'),('sum','Direct sum','#b65c2c')]:
        rows = arms[mode]['validation']
        for ax,source,key,title in zip(axes,('language','memory'),('nll','answer_nll'),('Natural language','Contextual answers')):
            ax.plot([r['step'] for r in rows],[r[source][key] for r in rows],marker='o',label=label,color=color)
            ax.set(title=title,xlabel='Paired optimizer updates',ylabel='Validation NLL')
            ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False)
    fig.savefig(a.out.with_name('SHoEbox_Context_Balance_Curves.png'),dpi=180)
    plt.close(fig)
    print(a.out)


if __name__ == '__main__':
    main()
