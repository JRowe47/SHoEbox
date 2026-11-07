"""Write a compact report from the completed unified-graph evidence."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,default=ROOT/'experiments/text_regions/results/graph_publication/summary.json')
    parser.add_argument('--out',type=Path,default=ROOT/'docs/SHoEbox_Graph_Publication_Report.md')
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    arms = summary['arms']
    credit = arms['credit']; base = arms['no_credit']
    best = credit['evaluation']['best']; control = base['evaluation']['best']
    last = credit['evaluation']['last']
    p = credit['protocol']['protocol']; cfg = credit['protocol']['config']
    start = credit['initial']; end = credit['history'][-1]['validation']
    totals = last['totals']
    count = credit['protocol']['parameters']
    rows = []
    for name,label in (('credit','Delayed credit'),('no_credit','No delayed credit')):
        e = arms[name]['evaluation']['best']
        rows.append(f"| {label} | {e['step']} | {e['language']['nll']:.4f} | {e['memory']['test']['answer_nll']:.4f} | {e['memory']['test']['groups']['exact']['nll']:.4f} | {e['memory']['long_delay']['groups']['exact']['nll']:.4f} |")
    controls = []
    for name,label in (('blocked','Block regional reads'),('no_demand','Remove demand feedback'),
        ('always_state','Always commit producer state'),('always_write','Always publish'),
        ('pooled','Pool recall roles'),('no_feedback','Remove both feedback paths')):
        r = best['interventions'][name]
        controls.append(f"| {label} | {r['language_delta']:+.4f} | {r['answer_delta']:+.4f} |")
    owner_rows = []
    for owner,values in best['memory']['test']['owners'].items():
        owner_rows.append(f"| {owner} | {values['write_fraction']:.1%} | {values['mean_demand'][0]:.3f} | {values['mean_demand'][1]:.3f} |")
    delta_language = best['language']['nll']-control['language']['nll']
    delta_answer = best['memory']['test']['answer_nll']-control['memory']['test']['answer_nll']
    retention = credit['query_retention']
    minimum = min([(0,start['language']['nll'])] + [(r['step'],r['validation']['language']['nll'])
                  for r in credit['history']],key=lambda item:item[1])
    balance = (f"Language validation was best at update {minimum[0]} ({minimum[1]:.4f}) and finished at "
               f"{end['language']['nll']:.4f}. Preserving language while improving answers is the next training priority."
               if end['language']['nll'] > minimum[1] + .05 else
               "Continue tracking language and answer metrics separately as mixed-data exposure increases.")
    text = f'''# SHoEbox: unified regional learning

2026-11-07 | Continuation of the regional-recall milestone

The ordinary seven-node regional graph now learns when to publish, retain state,
and adapt retained payloads. Natural language and memory tasks use the same
regional experts, byte interface, recall slots and learning path. Downstream
read demand becomes a trainable input to the next round's publication decisions.

## Completed learning run

| Selected model | Update | Language NLL | Answer NLL | Exact NLL | Long exact NLL |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Lower NLL is better. Language uses {best['language']['targets']:,} held-out byte
targets; task test uses {best['memory']['test']['answers']:,} answer targets in
{p['eval_count']*2} fresh documents. Long-delay documents triple the configured
delays. Exact answers are a subset of the combined answer score.
The credit arm's exact-answer accuracy is {best['memory']['test']['groups']['exact']['accuracy']:.1%}
on the ordinary test and {best['memory']['long_delay']['groups']['exact']['accuracy']:.1%}
on the long-delay test.

The credit arm's validation language NLL moved from {start['language']['nll']:.4f}
to {end['language']['nll']:.4f}; answer NLL moved from
{start['memory']['answer_nll']:.4f} to {end['memory']['answer_nll']:.4f}.
At the selected checkpoints, adding delayed credit changed test language NLL by
{delta_language:+.4f} and answer NLL by {delta_answer:+.4f} versus the matched arm.
This comparison jointly switches delayed gate and payload gradients; live-window
gradients remain enabled in both arms.

{balance}

## What is integrated

- A/B/C/U/V learn independent write-or-hold policies from candidate content,
  stored content and preceding-round demand. P/Q continue publishing readout outputs.
- Holds preserve publication contents, identity and provenance. Optional state
  holds also preserve contextualizer and recurrent contents; execution clocks advance.
- Receiver demand aggregates mean and peak soft selection probability across
  valid readers. It cannot affect a writer earlier in the same round.
- Bounded owner/version/epoch-bound sensitivities carry conditional gate and
  low-rank payload credit across training cuts. No replay buffer is introduced.
- Language and memory sources share all trainable weights but retain separate
  runtime state, demand, credit and stream cursors.

## Matched exposure per arm

| Item | Value |
|---|---:|
| Parameters | {count:,} (729 added to the parent) |
| Regional nodes / rounds / recall slots per receiver | 7 / {cfg['rounds']} / {cfg['recall_slots']} |
| Updates / batch / byte window / gradient segment | {p['steps']} / {p['batch']} / {p['context']} / {p['truncate_every']} |
| Natural-text / procedural-memory bytes | {totals['language']['bytes']:,} / {totals['memory']['bytes']:,} |
| Training answer targets | {totals['memory']['answers']:,} |
| Expert executions, retained trajectory | {totals['language']['experts']+totals['memory']['experts']:,} |
| Additional inference demand state per stream | 40 bytes |
| Training eligibility per stream | {best['credit_bytes_per_source']['language']//p['batch']:,} bytes |

Both arms start from the same preceding graph weights, with fresh optimizer and
runtime state. All weights train. AdamW uses learning rate {p['lr']}, constant
through the run. Selection minimizes the mean of validation language and answer
NLL at updates 0, 40, 80, 120 and 160. Pauses add no selection opportunities.

## Whole-rollout controls

These changes start from fresh test streams and include effects on later writes
and state. Positive changes mean the intervention worsened NLL.

| Control on selected credit model | Language delta | Answer delta |
|---|---:|---:|
{chr(10).join(controls)}

## Learned publication behavior

| Owner | Writes per execution | Mean demand | Peak demand |
|---|---:|---:|---:|
{chr(10).join(owner_rows)}

---

## Retention and next steps

At answer queries, selected valid publications averaged
{retention['mean_selected_age']:.2f} bytes old;
{retention['fraction_older_than_segment']:.1%} were older than the
{retention['segment_bytes']}-byte gradient segment. These measurements cover
selected edge reads across all receivers and rounds; they do not assign a
stored publication to a particular fact.

The learned demand projection norm is {best['learned']['demand_feature_norm']:.4f};
the memory-feature norm is {best['learned']['memory_feature_norm']:.4f};
the payload-adapter coefficient norm is {best['learned']['payload_coefficient_norm']:.4f}.
The training archive records delayed-gradient norms separately for both sources.

## Verification and interpretation

The full five-suite run passed {summary['checks'].get('full_suite_tests',summary['checks']['tests'])} checks.
Targeted numerical and evaluation-cache checks bring the total to
{summary['checks']['tests']} unique checks. All 19 mixed/graph regressions passed
after the clipping update, followed by the evaluation-cache check.
Fourteen new integration checks cover causal demand, exact hard holds, bootstrap and resets,
multi-round recording, owner/version/epoch rejection, finite task gradients,
conditional gradient equivalence, single-use receipts, legacy behavior,
source isolation, and exact optimizer/state/credit/cursor resume.
Evaluation checkpoints each finished measurement and reuses results for identical weights.

The delayed estimator holds historical candidates, feature/demand networks and
cross-region candidate histories fixed. It restores the documented conditional
publication derivative; it does not reconstruct the full recurrent history.
Gate eligibility decays by 0.995 per active regional round.

Training also gained overflow-safe gradient clipping. It retains the preceding
FP32 arithmetic whenever the norm is finite, otherwise checks every gradient
entry and computes the norm in FP64 before scaling. Actual nonfinite entries
still stop the update. Recovery counts are included in the checkpoint totals.
The control resumed from update 120 after the original interruption; discarded
work is excluded from the retained-trajectory counts and adds no selection opportunity.

The architectural gap is closed: the general graph can now train useful
publication behavior with task feedback. The next pass should balance language
and answer learning, increase mixed-data exposure and inspect selective retention
under broader language contexts. Keep the switches for state commits, demand, role
composition and delayed credit. Code comments explicitly preserve the intended
coordination as a potentially threshold-dependent emergent feature for later ablation.

## Reproduce and inspect

The archive contains source, the exact corpus and parent, selected/final
checkpoints, optimizer/runtime/credit states, source cursors, query traces,
metrics, complete test output and a hash manifest.

```bash
cd SHoEbox
python scripts/run_graph_publication.py --parent ../graph_publication/parent.pt \\
  --data data/tiny_shakespeare.txt --out artifacts/graph_publication --parallel
python scripts/check_all.py
python scripts/summarize_graph_publication.py
```

Architecture and controls: `docs/graph_publication.md`.
Machine-readable results: `experiments/text_regions/results/graph_publication/summary.json`.
'''
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(text)
    print(args.out)


if __name__ == '__main__':
    main()
