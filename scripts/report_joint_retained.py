"""Build the report and compact public evidence from independently audited runs."""
import json
from pathlib import Path
import re
import shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/joint_retained'
SEEDS=(20261130,20261131,20261132)
TASKS=('recall_16','correction_16','withheld_16')


def main():
    records={};audits={}
    for mode in ('edge','joint'):
        for seed in SEEDS:
            path=OUT/f'{mode}_{seed}'
            records[mode,seed]=json.loads((path/'evaluation.json').read_text())
            audits[mode,seed]=json.loads((path/'audit.json').read_text())
            assert audits[mode,seed]['all_passed']
    regression=(OUT/'regression.log').read_text();assert 'All five experiment suites passed.' in regression
    tests=sum(int(n) for n in re.findall(r'Ran (\d+) tests? in',regression))
    supplement=OUT/'supplemental_regression.log'
    if supplement.exists():
        extra=supplement.read_text();assert re.search(r'^OK$',extra,re.MULTILINE)
        pattern=r'^(test_\w+ \([^)]+\)) \.\.\.'
        tests+=len(set(re.findall(pattern,extra,re.MULTILINE))-set(re.findall(pattern,regression,re.MULTILINE)))
    for seed in SEEDS:
        assert records['joint',seed]['initial']==records['edge',seed]['initial']
        for key in ('batch_hashes','expert_executions','external_events','source_occurrences'):
            assert audits['joint',seed][key]==audits['edge',seed][key]
    def mean(mode,name,field='accuracy'):
        return sum(records[mode,s][name][field] for s in SEEDS)/3
    def initial_mean(name):return sum(records['joint',s]['initial'][name]['accuracy'] for s in SEEDS)/3
    gates={str(s):dict(task=all(records['joint',s][n]['accuracy']>=.95 for n in TASKS),
        recurrence=records['joint',s]['correction_16']['accuracy']-records['joint',s]['k1']['accuracy']>=.05,
        retained=records['joint',s]['repeated_queries']['exact_memory_preserved'],
        frozen=audits['joint',s]['composer_weights_exact']) for s in SEEDS}
    weak=20261131
    weakest={mode:min(records[mode,weak][n]['accuracy'] for n in TASKS) for mode in ('edge','joint')}
    weakest['initial']=min(records['joint',weak]['initial'][n]['accuracy'] for n in TASKS)
    improved=weakest['joint']>max(weakest['initial'],weakest['edge'])
    summary=dict(seeds=list(SEEDS),gates=gates,weakest_parent_minimum=weakest,weakest_improved=improved,
        all_targets_passed=improved and all(all(g.values()) for g in gates.values()),
        unchanged_parent_means={n:initial_mean(n) for n in TASKS},
        means={mode:{n:mean(mode,n) for n in records[mode,SEEDS[0]] if 'accuracy' in records[mode,SEEDS[0]][n]}
               for mode in ('edge','joint')},
        regressions=tests,audited_prediction_groups=sum(a['prediction_groups'] for a in audits.values()),
        parameters=audits['joint',SEEDS[0]]['parameters'],
        trainable_parameters={m:audits[m,SEEDS[0]]['trainable_parameters'] for m in ('edge','joint')},
        matched_exposures=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    pct=lambda x:f'{100*x:.2f}%'
    row=lambda items:'| '+' | '.join(map(str,items))+' |\n'
    md=f'''# SHoEbox: joint producer and connection learning

Completed November 5, 2026. Three published parents, six matched continuations, one eight-node architecture.

## Result

Joint local learning reaches **{pct(mean('joint','recall_16'))} recall**, **{pct(mean('joint','correction_16'))} correction**, and **{pct(mean('joint','withheld_16'))} withheld-correction accuracy** at gap 16. The stronger 95% per-task threshold passes in **{sum(g['task'] for g in gates.values())}/3 seeds**. All locked phase targets: **{'passed' if summary['all_targets_passed'] else 'not passed'}**.

The weakest parent's minimum task score moves from **{pct(weakest['initial'])} unchanged** to **{pct(weakest['edge'])} with edge-only continuation** and **{pct(weakest['joint'])} with joint learning**, on the same fresh test episodes.

| Data seed | Arm | Selected update | Recall | Correction | Withheld: old |
| --- | --- | --- | --- | --- | --- |
'''
    for seed in SEEDS:
        initial=records['joint',seed]['initial']
        md+=row([seed,'Unchanged parent',0,*[pct(initial[n]['accuracy']) for n in TASKS]])
        for mode in ('edge','joint'):
            md+=row([seed,mode,audits[mode,seed]['selected_step'],*[pct(records[mode,seed][n]['accuracy']) for n in TASKS]])
    md+='''
Each accuracy uses 512 fresh fact episodes with held-out relation-table pairs. The new data seeds map in order to the three published integration parents, 20261120/21/22. Parent and continuation scores use identical cases; these parent scores therefore differ from the previous report's cohorts. No parent is excluded. Validation selects checkpoints, including update zero; test scores do not select weights.

## What changed

The fact producer F and its F-to-A value projection now learn together. Local supervision reaches F's byte contextualizer, ingress projection, sparse expert banks, recurrent processing and write/hold gate. The existing seven-node composer and common 256-byte output head remain exactly frozen. The architecture, inference algorithm, parameter count and expert execution count are unchanged.

The edge-only control sees the same observations, tables, losses and number of updates. Only its 1,024 edge weights can change. This comparison isolates permission to update the producer; it does not equate backward computation or elapsed time.

| Loss | Learning signal |
| --- | --- |
| Retained-payload alignment | Keep the published fact useful to the receiving region |
| Write-candidate alignment | Encode a newly observed fact before it is published |
| Balanced write/hold classification | Distinguish valid updates from distractors |
| Final byte cross-entropy | Make the composed query answer correct |

Alignment uses this model's own frozen A ingress representation of the observed fact byte. It adds no readout or inference head. The ordinary hard publication decision stays in use. Each chunk backpropagates its local losses before the 16-event detach boundary; the final chunk also receives query loss. Earlier activations are discarded without replay. Training labels never enter forward payloads or carried inference state.

## Delays and controls

| Condition | Edge only | Joint |
| --- | --- | --- |
'''
    for label,name in (('Recall, gap 16','recall_16'),('Correction, gap 16 / K=2','correction_16'),
        ('Withheld correction, gap 16','withheld_16'),('Recall, gap 64','recall_64'),
        ('Correction, gap 64','correction_64'),('Withheld correction, gap 64','withheld_64'),
        ('K=1','k1'),('K=4','k4'),('No delayed exchange','no_exchange'),('Blocked F-to-A edge','no_edge'),
        ('Reset memory','reset_memory'),('Shuffled memory: original target','shuffled_memory'),
        ('Unconditional publication, same weights','forced_always'),('Four consecutive queries','repeated_queries'),
        ('Final endpoint, correction','final_endpoint')):
        md+=row([label,pct(mean('edge',name)),pct(mean('joint',name))])
    md+=f'''
Shuffled memory answers the substituted fact's query with **{pct(mean('joint','shuffled_memory','counterfactual_accuracy'))}** accuracy. The payload and complete semantic header move together. Blocked-edge, reset, shuffle, disabled-exchange and unconditional-publication controls match the intact K=2 expert count. Four-value chance accuracy is 25%.

Repeated queries preserve F's contextual state, expert state, payload, header, version and provenance exactly. The producer executes no experts during queries. Two rounds remain the useful learned schedule; K=4 scores **{pct(mean('joint','k4'))}**.

## Publication and continued acquisition

| Data seed | Arm | Gap 16 write precision | Gap 16 write recall | Gap 64 precision | Gap 64 recall |
| --- | --- | --- | --- | --- | --- |
'''
    for s in SEEDS:
        for mode in ('edge','joint'):
            r=records[mode,s]
            md+=row([s,mode,*[pct(r[f'publication_{gap}'][f]) for gap in (16,64) for f in ('write_precision','write_recall')]])
    holds=sum(records['joint',s][f'publication_{gap}']['hold_pairs'] for s in SEEDS for gap in (16,64))
    md+=f'''
All **{holds:,}** measured joint-model hold transitions retain the payload bit-for-bit. These gate statistics cover correction episodes and are separate from the task acceptance threshold.

| Live sequence | Edge only | Joint |
| --- | --- | --- |
'''
    for name,label in (('before','Three queries before update'),('after','Query after correction'),('withheld','Query with correction withheld')):
        md+=row([label,*[pct(sum(records[m,s]['revision_after_queries'][name]['accuracy'] for s in SEEDS)/3) for m in ('edge','joint')]])
    md+='''
This continued-acquisition test keeps the prior protocol: after three queries, both branches observe the original fact again before an optional correction. It measures continued acquisition following live queries.

## Training and verification

Each continuation receives 80 updates and 2,560 new fact episodes. Recall and correction alternate over gaps 0,4,8,16. The two arms have identical data digests, table choices and exposure/execution counts. AdamW uses producer learning rate 0.0003, edge rate 0.001, beta=(0.9,0.95), weight decay 0.01 and gradient clipping at 1. Validation occurs at updates 0,20,40,60,80.

| Seed | Arm | Expert executions | External events | Training minutes |
| --- | --- | --- | --- | --- |
'''
    for s in SEEDS:
        for m in ('edge','joint'):
            a=audits[m,s];md+=row([s,m,f"{a['expert_executions']:,}",f"{a['external_events']:,}",f"{a['training_seconds']/60:.2f}"])
    md+=f'''
Times are concurrent CPU wall time, excluding validation and audits. Each run has {audits['joint',SEEDS[0]]['source_occurrences']:,} source occurrences. The graph still has **317,098 parameters** and **480 physical experts**. Joint training permits **{summary['trainable_parameters']['joint']:,} parameters** to update. A fact byte executes 12 experts; a table pair and two-round query execute 264.

All **{tests} regression tests pass**. Independent audits reconstruct **480 update batches**, verify checkpoint selection and frozen composer tensors, regenerate **{summary['audited_prediction_groups']} prediction groups**, and reproduce exported logits and complete state exactly. Tests verify local gradient reachability, detach boundaries, label isolation, inference parity, parameter ownership and bit-exact optimizer continuation. Full histories, predictions, optimizer checkpoints and the separate pilot are included in the artifact archive.

## Next target

Extend the retained-state interface to multiple independently owned facts, including a shared fact available from either source and source-private details, while preserving these composition and correction controls. This is the remaining functional integration work toward M7; natural-text evaluation follows that integration.

## Reproduce

Extract source and artifact ZIPs into the same directory. Both use the SHoEbox/ root. Install requirements.txt, then run:

```bash
python scripts/check_all.py
python scripts/audit_joint_retained.py artifacts/joint_retained/joint_20261131
python scripts/report_joint_retained.py
```

Run the audit command for each of the six edge/joint folders to regenerate the full comparison. For a fresh complete run, use `python scripts/run_joint_retained.py --root artifacts/joint_retained_fresh`; the runner reuses the three parent checkpoints in the packaged artifacts/joint_retained/parents directory. The run is resumable by repeating the same command. The source ZIP is the exact published Git tree; artifact hashes and source commit are recorded in its companion manifest.
'''
    (ROOT/'docs/SHoEbox_Joint_Retention_Report.md').write_text(md)
    public=ROOT/'experiments/text_regions/results/joint_retained';public.mkdir(parents=True,exist_ok=True)
    for name in ('summary.json','protocol_lock.json','environment.json','runtime_compatibility.json'):
        shutil.copyfile(OUT/name,public/name)
    compact={f'{mode}_{seed}':{k:v for k,v in a.items() if k not in ('batch_hashes','changed_tensors')}
             for (mode,seed),a in audits.items()}
    (public/'audit_summary.json').write_text(json.dumps(compact,indent=2)+'\n')
    (public/'README.md').write_text('''# Joint producer/edge retention

See the [report](../../../../docs/SHoEbox_Joint_Retention_Report.md) and [locked protocol](../../../../docs/SHoEbox_Joint_Retention_Protocol.md). Compact results are committed here. The downloadable artifact ZIP contains full predictions, histories, three parents, six resumable runs, inference exports and independent audits.
''')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
