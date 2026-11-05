"""Generate result tables from the independently audited producer comparison."""
import json
from pathlib import Path
import re
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/selective_publication'
SEEDS = (20261110, 20261111, 20261112)


def main():
    audit = json.loads((OUT/'evaluation_audit.json').read_text())
    compatibility = json.loads((OUT/'compatibility.json').read_text())
    regression = (OUT/'regression.log').read_text()
    assert audit['all_passed'] and compatibility['all_passed'] and 'All five experiment suites passed.' in regression
    tests = sum(map(int, re.findall(r'Ran (\d+) tests?', regression)))
    records = {name: json.loads((OUT/name/'evaluation.json').read_text()) for name in audit['runs']}
    pct = lambda value: f'{100*value:.2f}%'
    def score(policy, seed, gap, kind, control, runtime=None, checkpoint='best'):
        return records[f'{policy}_{seed}']['groups'][f'{checkpoint}/{runtime or policy}/{gap}/{kind}/{control}']
    def table(headers, rows):
        return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                          *['| '+' | '.join(map(str, row))+' |' for row in rows]])
    tasks = (('recall','carried','Recall'), ('correction','carried','Correction'),
             ('correction','withheld','Withheld update: old value'))
    targets = {str(seed): all(score('learned',seed,16,kind,control)['accuracy'] >= .9
                             for kind,control,_ in tasks) for seed in SEEDS}
    gate_targets = {str(seed): all(score('learned',seed,16,k,c)[field] >= .95
                                  for k,c,_ in tasks for field in ('write_precision','write_recall')) for seed in SEEDS}
    calibrated_targets = {str(seed): all(score('learned',seed,16,k,c,checkpoint='calibrated')['accuracy'] >= .9
                                         for k,c,_ in tasks) and
                          all(score('learned',seed,16,k,c,checkpoint='calibrated')[field] >= .95
                              for k,c,_ in tasks for field in ('write_precision','write_recall')) for seed in SEEDS}
    summary = dict(task_targets=targets, gate_targets=gate_targets, regression_tests=tests,
                   overall_target_passed=all(targets.values()) and all(gate_targets.values()),
                   compatibility=compatibility, calibrated_targets=calibrated_targets, means={})
    for policy in ('learned','always'):
        summary['means'][policy] = {label: mean(score(policy,s,16,k,c)['accuracy'] for s in SEEDS) for k,c,label in tasks}
    summary['long_gap_means'] = {policy: {label:mean(score(policy,s,64,k,c)['accuracy'] for s in SEEDS)
                                         for k,c,label in tasks} for policy in ('learned','always')}
    rows = []
    for seed in SEEDS:
        for policy in ('always','learned'):
            run = audit['runs'][f'{policy}_{seed}']
            rows.append([str(seed), policy, run['selected_step'], *[pct(score(policy,seed,16,k,c)['accuracy']) for k,c,_ in tasks]])
    learned_mean = mean(summary['means']['learned'].values())
    always_mean = mean(summary['means']['always'].values())
    assessment = ('All three learned-publication runs meet the declared task and write-decision targets.'
                  if summary['overall_target_passed'] else
                  f"The task-accuracy target passes in {sum(targets.values())}/3 learned-publication runs; "
                  f"the write precision/recall target passes in {sum(gate_targets.values())}/3.")
    page1 = f'''# SHoEbox: learned publication retention

Completed November 5, 2026. Three paired seeds, extracted sparse producer, common byte decoder.

## Result

{assessment}

The main gain appears beyond the training delay. At gap 64, learned publication reaches **{pct(summary['long_gap_means']['learned']['Recall'])} recall versus {pct(summary['long_gap_means']['always']['Recall'])}** for unconditional publication. When a correction is withheld, it retains the old value at **{pct(summary['long_gap_means']['learned']['Withheld update: old value'])} versus {pct(summary['long_gap_means']['always']['Withheld update: old value'])}**. All three seed pairs improve on each longer-delay task.

At the training-range gap 16, mean accuracy across recall, correction and withheld correction is **{pct(learned_mean)} with learned publication** and **{pct(always_mean)} with unconditional publication**. Each run uses 240 updates and 7,680 fresh training episodes.

The mechanism adds a learned write/hold decision to a node's existing output. On hold, the runtime preserves its payload, semantic header, validity, version and provenance. The node still executes its contextualizer, routers, sparse experts and recurrent-state update. The decision reads only the computed latent.

{table(['Seed','Publication','Selected update','Recall','Correction','Withheld: old'], rows)}

Each cell uses 512 fresh test episodes at gap 16. Selection uses validation only: minimum accuracy, then mean accuracy, then lower NLL. The 90% task threshold applies independently to each condition and seed. Final-endpoint results are also saved, regardless of selection.

The producer has **71,867 parameters**, including a **33-parameter gate**, 96 physical experts, and 12 expert executions per byte. Both paired arms have the same gate and losses; unconditional publication overrides only the write decision. Both inherit the same source-A computation from the selected gradient-horizon checkpoint and initialize a fresh 256-byte decoder. This isolates publication behavior within the producer task.
'''
    rows = []
    for k,c,label in tasks:
        rows.append([label, pct(mean(score('always',s,16,k,c)['accuracy'] for s in SEEDS)),
                     pct(mean(score('learned',s,16,k,c)['accuracy'] for s in SEEDS)),
                     pct(mean(score('learned',s,16,k,c,runtime='always')['accuracy'] for s in SEEDS))])
    gate_rows = [[str(s), pct(min(score('learned',s,16,k,c)['write_precision'] for k,c,_ in tasks)),
                 pct(min(score('learned',s,16,k,c)['write_recall'] for k,c,_ in tasks)),
                 'pass' if gate_targets[str(s)] else 'fail'] for s in SEEDS]
    control_rows = []
    for k,c,label in tasks:
        control_rows.append([label+', gap 64', *[pct(mean(score(p,s,64,k,c)['accuracy'] for s in SEEDS)) for p in ('always','learned')]])
    for c,label in (('reset','Reset before query'), ('no_writes','All writes withheld'), ('absent','Source absent')):
        control_rows.append([label, *[pct(mean(score(p,s,16,'correction',c)['accuracy'] for s in SEEDS)) for p in ('always','learned')]])
    holds = sum(score('learned',s,g,k,c)['hold_pairs'] for s in SEEDS for g in (16,64) for k,c,_ in tasks)
    page2 = f'''## Controls and learned decisions

The following accuracies are means of the three seeds. The last column changes only the selected learned model's runtime publication policy; weights remain fixed.

{table(['Task, gap 16','Trained always','Learned','Learned weights, forced always'],rows)}

{table(['Seed','Minimum write precision','Minimum write recall','95% target'],gate_rows)}

Write scores above take the minimum over recall, correction and withheld correction at gap 16. The data contains bare A-D distractors, invalid '=x' strings, and independently randomized write timing. The model must recognize an observed write in context. Classification labels are never passed to inference.

{table(['Control / extrapolation','Trained always','Learned'],control_rows)}

Gap 64 exceeds the largest training gap, 16. Withheld corrections are scored against the old observed value. Reset clears the complete graph state before the final query. Source absence masks every input occurrence. Four-value chance accuracy is 25%.

All **{holds:,} measured hold transitions** across the carried/withheld gap-16 and gap-64 conditions preserve the latent payload bit-for-bit. Unit tests separately verify header and provenance retention, inactive rows, resets and serialization.
'''
    budgets = []
    for policy in ('always','learned'):
        run = audit['runs'][f'{policy}_{SEEDS[0]}']
        budgets.append([policy, f"{run['external_events']:,}", f"{run['expert_executions']:,}",
                        f"{mean(audit['runs'][f'{policy}_{s}']['training_seconds'] for s in SEEDS)/60:.2f}"])
    calibration_rows = []
    for seed in SEEDS:
        calibration = json.loads((OUT/f'learned_{seed}'/'calibration.json').read_text())
        s = score('learned',seed,16,'correction','carried',checkpoint='calibrated')
        calibration_rows.append([str(seed),calibration['selected']['offset'],
                                 *[pct(score('learned',seed,16,k,c,checkpoint='calibrated')['accuracy']) for k,c,_ in tasks],
                                 pct(s['write_precision'])+' / '+pct(s['write_recall'])])
    calibration_page = f'''## Validation-only threshold follow-up

Seed 20261111 exposed false writes on distractors despite high training accuracy. Before viewing any test results, a separate protocol locked a small threshold grid: offsets 0, 0.5, 1, 1.5, 2, 2.5 and 3 subtracted from the trained gate bias. All expert, recurrent, ingress and decoder weights remain unchanged.

Validation ranks whether the declared task and decision thresholds pass, then minimum task accuracy, minimum decision precision/recall, mean task accuracy, lower NLL and smaller offset. The original fixed-budget results on the preceding pages are unchanged.

{table(['Seed','Bias offset','Recall','Correction','Withheld: old','Write precision / recall'],calibration_rows)}

These are test scores after validation-only selection. **{sum(calibrated_targets.values())}/3 calibrated runs** meet all three task thresholds and all three conditions' decision thresholds. The full nine-condition evaluations are included in the raw results. Calibration does not add parameters or an inference pass.

The selected offset is checked against the full engine. Because this isolated producer has no feedback edges, candidate latents and write logits are independent of retained publications. Validation computes those sequences once and checks the seven offsets by applying ordinary publication replacements. The audit independently regenerates the selection and exported tensors.

This follow-up distinguishes a misplaced decision threshold from an incorrect learned write classification. Highly confident false writes can survive a threshold change; improving that distinction requires further producer learning.
'''
    page3 = f'''## Training, verification and scope

Each update uses a fresh batch. Loss combines candidate-value decoding at writes, retained-value decoding after observation, balanced write/hold classification, and the existing router regularizers. All retained labels follow the actual observed stream. Gradients detach every 16 events; held publication values persist. There is no straight-through gradient through the hard write decision.

{table(['Arm','Events per run','Expert calls per run','Mean training minutes'],budgets)}

Training time is wall time during concurrent CPU runs, excluding validation and audits. Gate computation and parameter counts are matched; expert counts do not imply equal wall time.

All **{tests} regression tests pass**. The audit reconstructs all 1,440 training batches, verifies matched initial weights and exposure in each pair, recomputes validation selection, regenerates **{audit['prediction_groups']} test prediction groups**, and compares inference exports against their selected checkpoints. Resume tests reproduce the next optimizer update and moments exactly.

The original seven-node checkpoint still achieves **100% two-round composition on 592 held-out cases**. Running it through the old and new engine gives identical logits, carried state tensors and execution counts. Its weights are unchanged, and selective publication remains disabled by default.

This phase implements and tests one producer's write/retain interface, including seed sensitivity. It does not combine the trained producer with the seven-node composition model. **M7 remains open.** The next work is to stabilize write discrimination across seeds and then integrate retained fact publications with current query signals. A single held publication cannot simultaneously serve as an unchanged fact and a newly arriving query signal; that interface distinction needs to be explicit in the integration experiment.

## Reproduce

Extract both ZIPs into the same directory; each uses the SHoEbox/ root. Install requirements.txt, then run:

```bash
python scripts/audit_selective_publication.py --evaluations
python scripts/check_all.py
```

For fresh runs, use the locked protocol and scripts/run_selective_publication.py. The artifact ZIP includes the parent, six complete optimizer checkpoints and selected models, inference exports, histories, predictions, audits, compatibility evidence and the separate development pilot. The source ZIP contains the exact published Git tree.
'''
    report = ROOT/'docs/SHoEbox_Selective_Publication_Report.md'
    report.write_text('\n---\n\n'.join((page1,page2,calibration_page,page3)))
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(report)


if __name__ == '__main__': main()
