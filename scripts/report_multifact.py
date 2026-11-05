"""Generate the report and compact public results from completed independent audits."""
import json
from pathlib import Path
import re
import shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/multifact'
SEEDS=(20261220,20261221,20261222)
KINDS=('recall','shared_left','shared_right','left_private','right_private')
FIELDS=('Shared','Left private','Right private')


def main():
    data={s:json.loads((OUT/str(s)/'evaluation.json').read_text()) for s in SEEDS}
    audits={s:json.loads((OUT/str(s)/'audit.json').read_text()) for s in SEEDS}
    assert all(a['all_passed'] for a in audits.values())
    protocol=json.loads((OUT/'protocol_lock.json').read_text())
    regression=(OUT/'regression.log').read_text()
    recovery=OUT/'regression_recovery.json'
    if recovery.exists():
        ledger=json.loads(recovery.read_text());assert ledger['all_passed']
        assert set(ledger['completed_before_interruption'])|set(ledger['resumed_tests'])==set(ledger['expected_text_region_tests'])
        tests=ledger['unique_total']
    else:
        assert 'All five experiment suites passed.' in regression
        tests=sum(int(n) for n in re.findall(r'Ran (\d+) tests? in',regression))
    supplement=OUT/'supplemental_regression.log'
    if supplement.exists():
        extra=supplement.read_text();assert re.search(r'^OK$',extra,re.MULTILINE)
        pattern=r'^(test_\w+ \([^)]+\)) \.\.\.'
        tests+=len(set(re.findall(pattern,extra,re.MULTILINE))-set(re.findall(pattern,regression,re.MULTILINE)))
    def scores(seed,name,key='accuracy'):return [r[key] for r in data[seed][name]['fields']]
    def mean(name,key='accuracy'):return [sum(scores(s,name,key)[f] for s in SEEDS)/3 for f in range(3)]
    primary=list(KINDS)+['withheld_'+k for k in KINDS[1:]]
    gates={}
    for s in SEEDS:
        gates[s]=dict(primary=min(v for name in primary for v in scores(s,name))>=.9,
            either_source=min(scores(s,'left_only')[0:2]+[scores(s,'right_only')[0],scores(s,'right_only')[2]])>=.9,
            unsupported_private=max(scores(s,'left_only')[2],scores(s,'right_only')[1])<=.35,
            live=min(v for name in ['live_'+k for k in KINDS[1:]]+['live_withheld'] for v in scores(s,name))>=.9,
            reset_and_block=max(scores(s,'reset')+scores(s,'no_edge'))<=.35,
            recurrence=(sum(scores(s,'shared_right'))-sum(scores(s,'k1')))/3>=.05,
            exact=audits[s]['composer_weights_exact'] and audits[s]['export_exact'])
    holds=sum(row['hold_pairs'] for s in SEEDS for row in data[s]['publication'].values())
    writes=[row for s in SEEDS for row in data[s]['publication'].values()]
    summary=dict(seeds=list(SEEDS),gates=gates,all_targets_passed=all(all(g.values()) for g in gates.values()),
        means={name:mean(name) for name,value in data[SEEDS[0]].items() if 'fields' in value},
        primary_minimum={s:min(v for name in primary for v in scores(s,name)) for s in SEEDS},
        five_task_minimum={s:min(v for name in KINDS for v in scores(s,name)) for s in SEEDS},
        five_task_mean={s:sum(v for name in KINDS for v in scores(s,name))/15 for s in SEEDS},
        initial_minimum={s:min(r['accuracy'] for k in KINDS for r in data[s]['initial'][k]['fields']) for s in SEEDS},
        initial_mean={s:sum(r['accuracy'] for k in KINDS for r in data[s]['initial'][k]['fields'])/15 for s in SEEDS},
        regressions=tests,audited_prediction_groups=sum(a['prediction_groups'] for a in audits.values()),
        training_batches=sum(a['training_batches'] for a in audits.values()),hold_transitions=holds,
        publication_precision_range=[min(r['precision'] for r in writes),max(r['precision'] for r in writes)],
        publication_recall_range=[min(r['recall'] for r in writes),max(r['recall'] for r in writes)],
        parameters=audits[SEEDS[0]]['parameters'],trainable_parameters=audits[SEEDS[0]]['trainable_parameters'],
        interface='Explicit field addresses; learned raw-byte write gates; deterministic latest-valid shared replica')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    pct=lambda x:f'{100*x:.2f}%'
    row=lambda values:'| '+' | '.join(map(str,values))+' |\n'
    md=f'''# SHoEbox: simultaneous shared and private retention

Completed November 5, 2026. Three parent seeds; two source descriptions; three retained logical facts in one eleven-node graph.

## Result

All locked phase targets: **{'passed' if summary['all_targets_passed'] else 'not passed'}**. The primary target passes in **{sum(g['primary'] for g in gates.values())}/3 seeds**. Each primary condition queries all three fields in 512 fresh episodes per seed. Relation-table pairs are held out throughout. Intact final fact triples use held-out combinations; withheld-update controls retain their original triple, which need not be held out.

| Seed | Update | Initial five-task min | Trained five-task min | All primary min |
| --- | --- | --- | --- | --- |
'''
    for s in SEEDS:md+=row([s,audits[s]['selected_step'],pct(summary['initial_minimum'][s]),pct(summary['five_task_minimum'][s]),pct(summary['primary_minimum'][s])])
    md+=f'''
The five-task columns compare the same recall/correction tasks and episodes before and after training. Across those tasks and all three seeds, mean accuracy rises from **{pct(sum(summary['initial_mean'].values())/3)} to {pct(sum(summary['five_task_mean'].values())/3)}**. The all-primary column additionally includes four withheld-correction conditions. The initial model already inherits the published producer and composer weights. Checkpoint selection uses validation only, with update zero eligible.

| Composed-query condition | Shared | Left private | Right private |
| --- | --- | --- | --- |
'''
    labels={'recall':'Recall all fields','shared_left':'Common correction from left','shared_right':'Common correction from right',
        'left_private':'Left-private correction','right_private':'Right-private correction'}
    for k in KINDS:md+=row([labels[k],*[pct(v) for v in mean(k)]])
    for k in KINDS[1:]:md+=row(['Withheld: '+labels[k].lower(),*[pct(v) for v in mean('withheld_'+k)]])
    md+='''
An answer is `g(f(x))` for the requested retained fact x and newly supplied relation tables. The model never decodes x into an intermediate byte. Correcting one field must preserve the other two answers.

## What changed

F and L read the complete left-source byte stream; G and R read the complete right-source stream. Each source has its own common-field and private-field producer. Every owner has independent contextual state, sparse expert state, and a retained publication. Their write gates distinguish common `=value` from private `:value` observations among identifiers and distractors.

The field query is an explicit address: shared, left-private, or right-private. A shared read chooses the latest valid F/G publication using occurrence metadata; a private read chooses its dedicated owner. Addressing and freshness arbitration are explicit interface mechanics. The producer experts learn which observed bytes to publish. Natural-language query interpretation remains part of the next integration work.

All four producers begin from the matching joint-retention parent. Private producers exchange the learned input-embedding rows for colon and equals at initialization. Runtime bytes are unchanged. Both fields still reach both local producers, so private/shared separation must survive actual interleaved observations.

The original seven-node composer and 256-byte decoder remain exactly frozen. Only the four producers and their continuous edge value projections learn. Queries run the ordinary two-round composer with the four producers inactive.

## Missing sources and continued updates

| Available source information | Shared | Left private | Right private |
| --- | --- | --- | --- |
'''
    for name,label in (('left_only','Left source only'),('right_only','Right source only'),('erase_left','Left facts erased, equal compute'),('erase_right','Right facts erased, equal compute')):
        md+=row([label,*[pct(v) for v in mean(name)]])
    md+='''
These controls use all 64 independent fact triples. An unsupported private value has a conditional 25% prior. True source absence skips its two producers; erased-source controls keep the same execution count as intact acquisition.

| Update after three live queries | Shared | Left private | Right private |
| --- | --- | --- | --- |
'''
    for k in KINDS[1:]:md+=row([labels[k],*[pct(v) for v in mean('live_'+k)]])
    md+=row(['Common correction withheld',*[pct(v) for v in mean('live_withheld')]])
    md+=f'''
These episodes present the original facts once, ask three live field queries, then append only the update suffix. They do not re-observe the initial facts. Repeated queries preserve all four owners' contextual state, expert state, payload, semantic header, version and provenance exactly. All **{holds:,}** measured hold transitions preserve their publication payload bit-for-bit.

Across the twelve owner/seed pairs, observed write precision ranges from **{pct(summary['publication_precision_range'][0])} to {pct(summary['publication_precision_range'][1])}**, and write recall ranges from **{pct(summary['publication_recall_range'][0])} to {pct(summary['publication_recall_range'][1])}**.

## Controls and longer delay

| Condition | Shared | Left private | Right private |
| --- | --- | --- | --- |
'''
    for name,label in (('long_recall','Recall, gap 64'),('long_correction','Common correction, gap 64'),('k1','One regional round'),
        ('k4','Four regional rounds'),('no_exchange','No delayed exchange'),('no_edge','Memory edges blocked'),('reset','Memory reset'),
        ('shuffle','Shuffled memory: original target'),('always','Unconditional publication'),('repeated','Three repeated query sets')):
        md+=row([label,*[pct(v) for v in mean(name)]])
    md+=f'''
Shuffling the complete memory publications yields counterfactual accuracies of **{', '.join(pct(v) for v in mean('shuffle','counterfactual_accuracy'))}** for shared, left-private and right-private queries. Blocked, reset, shuffle, no-exchange and unconditional-publication conditions match the intact two-round expert count.

Forcing a stale left common replica after a right-source correction follows the old value with **{pct(mean('prefer_stale_left','old_accuracy')[0])}** accuracy; the reverse direction follows the old value with **{pct(mean('prefer_stale_right','old_accuracy')[0])}**. The default latest-publication rule is evaluated separately from the learned write gates.

## Learning and verification

The first development pilot exposed destructive writes to the opposite field. The final gate loss gives those events a separate denominator: 0.5 own-field writes, 0.25 opposite-field writes, and 0.25 ordinary holds. Retained and candidate latents align with this model's frozen A ingress representation; the final objective also includes three byte-query losses and the original routing regularizers. Each 16-event chunk backpropagates before its history is detached. There is no replay store.

The confirmatory protocol uses {protocol['steps']} updates, batch {protocol['batch']}, producer LR 0.0003, edge LR 0.001, AdamW beta=(0.9,0.95), weight decay 0.01 and clipping at 1. Gaps cycle through 0, 4, 8, 16. Validation at 20-update intervals selects the minimum task/field accuracy, then mean accuracy, then lower NLL. Test results never select weights.

| Seed | Expert executions | Encoded source occurrences | Training minutes |
| --- | --- | --- | --- |
'''
    for s,a in audits.items():md+=row([s,f"{a['expert_executions']:,}",f"{a['source_occurrences']:,}",f"{a['training_seconds']/60:.2f}"])
    md+=f'''
Times and training counters sum the completed checkpointed updates. Discarded partial work after an execution-worker interruption, validation and audits are excluded. The graph has **{summary['parameters']['total_unique_parameters']:,} parameters**, **{summary['parameters']['physical_experts']} physical experts**, and **{summary['trainable_parameters']:,} trainable parameters**. One aligned pair of source bytes executes 48 experts across four producers; one table pair and two-round query execute 264. Two raw source streams are encoded four times, and the counters include that duplicated work.

All **{tests} regression tests pass**. Independent audits reconstruct **{summary['training_batches']} training batches**, verify the selected checkpoint and frozen composer tensors, reproduce **{summary['audited_prediction_groups']} field-prediction groups**, and reproduce inference-export logits and complete state exactly. Source isolation, causal labels, held-out factor combinations, unique episode IDs, query preservation and exact optimizer continuation are tested. The pilots are separate from confirmatory results and include their exact source snapshots.

## Next target and reproduction

The next stage is joint natural-text training and a generative byte-stream interface, retaining these memory, source-ownership and composition checks. M7 remains open until that text experiment is measured.

Extract the source and artifact ZIPs into the same directory; both use the SHoEbox/ root. Install requirements.txt, then run:

```bash
python scripts/check_all.py
python scripts/audit_multifact.py artifacts/multifact/20261220
python scripts/report_multifact.py
```

Audit each of the three seed directories to reproduce the full comparison. A fresh run uses `python scripts/run_multifact.py --root artifacts/multifact_fresh`; the runner reuses packaged parent checkpoints and resumes interrupted training when repeated. Initial, selected and final optimizer checkpoints, inference states, per-example predictions, protocols and full audits are in the artifact ZIP. The source ZIP is the exact published Git tree; the companion manifest records its commit and every artifact hash.
'''
    (ROOT/'docs/SHoEbox_Multifact_Report.md').write_text(md)
    public=ROOT/'experiments/text_regions/results/multifact';public.mkdir(parents=True,exist_ok=True)
    for name in ('summary.json','protocol_lock.json','environment.json'):shutil.copyfile(OUT/name,public/name)
    (public/'audit_summary.json').write_text(json.dumps(audits,indent=2)+'\n')
    (public/'README.md').write_text('# Simultaneous shared/private retention\n\nSee the [report](../../../../docs/SHoEbox_Multifact_Report.md) and [locked protocol](../../../../docs/SHoEbox_Multifact_Protocol.md). Full predictions, checkpoints and independent reconstruction records are in the downloadable artifact archive.\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
