"""Generate the integration report directly from completed audited records."""
import json
import re
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/retained_composition'
SEEDS=(20261120,20261121,20261122)


def main():
    train=json.loads((OUT/'training_audit.json').read_text())
    audit=json.loads((OUT/'evaluation_audit.json').read_text())
    compatibility=json.loads((OUT/'compatibility.json').read_text())
    assert train['all_passed'] and audit['all_passed'] and compatibility['all_passed']
    regression=(OUT/'regression.log').read_text()
    assert 'All five experiment suites passed.' in regression
    test_count=sum(int(n) for n in re.findall(r'Ran (\d+) tests? in',regression))
    records={s:json.loads((OUT/str(s)/'evaluation.json').read_text()) for s in SEEDS}
    def mean(name,field='accuracy'):return sum(records[s][name][field] for s in SEEDS)/3
    def pct(x):return f'{x*100:.2f}%'
    def row(items):return '| '+' | '.join(str(x) for x in items)+' |\n'
    gates={str(s):dict(task=all(records[s][name]['accuracy']>=.9 for name in ('recall_16','correction_16','withheld_16')),
        recurrence=records[s]['correction_16']['accuracy']-records[s]['k1']['accuracy']>=.05,
        retained=records[s]['repeated_queries']['exact_memory_preserved']) for s in SEEDS}
    summary=dict(seeds=list(SEEDS),gates=gates,all_targets_passed=all(all(x.values()) for x in gates.values()),
        mean_accuracy={name:mean(name) for name in records[SEEDS[0]] if name!='revision_after_queries'},
        parameters=train['runs'][0]['parameters'],trainable_parameters=1024,regression_tests=test_count,
        audited_prediction_groups=audit['prediction_groups'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    passes=sum(g['task'] for g in gates.values())
    md=f'''# SHoEbox: retained facts drive live composition

Completed November 5, 2026. Three seeds, one eight-node graph, separate retained memory and query workspace.

## Result

The integrated graph answers new two-step relation queries from a retained fact. At gap 16, mean accuracy is **{pct(mean('recall_16'))} for recall**, **{pct(mean('correction_16'))} after correction**, and **{pct(mean('withheld_16'))} when a correction is withheld and the old fact remains authoritative**. The 90% task target passes in **{passes}/3 seeds**.

Two query rounds reach **{pct(mean('correction_16'))}**, compared with **{pct(mean('k1'))}** for one round and **{pct(mean('no_exchange'))}** for two rounds without delayed exchange. The retained fact supplies the query value; the existing regional experts perform both relation lookups.

| Training seed | Producer seed | Selected update | Recall | Correction | Withheld: old |
| --- | --- | --- | --- | --- | --- |
'''
    for s,r in zip(SEEDS,train['runs']):
        md+=row([s,s-10,r['selected_step'],*[pct(records[s][n]['accuracy']) for n in ('recall_16','correction_16','withheld_16')]])
    md+='''
Each cell uses 512 fresh fact episodes with independently selected held-out relation-table pairs. Checkpoint selection uses validation only. The source producer checkpoints are the three previously published, uncalibrated selections; no producer seed is excluded.

## What changed

An eighth sparse node, F, owns its byte contextualizer and expert sharing bank. Its publication connects to A through an ordinary latent edge. The seven existing composition nodes, their relation pathways, and their common 256-byte decoder remain in the same engine. The runtime now supports independently owned source encoders and banks without changing the default shared configuration.

| Input | Information supplied | Role during a query |
| --- | --- | --- |
| F | Earlier fact observations and corrections | Retained latent supplies x; F is inactive |
| B | Current four-entry relation table f | First learned lookup |
| C | Current four-entry relation table g | Second learned lookup |
| P/Q | Regional latent publications | Existing common byte head emits g(f(x)) |

Only the new edge's **1,024 weights** are fitted. All producer and composer weights remain frozen. No fact is decoded to an intermediate byte at inference, and no fact label or task identifier enters the query. A fresh query clears the seven-node workspace while preserving F's contextual and recurrent state and its retained publication.

---

## Delays and causal controls

| Condition | Mean accuracy |
| --- | --- |
'''
    for label,name in (('Recall, gap 64','recall_64'),('Correction, gap 64','correction_64'),
        ('Withheld correction, gap 64','withheld_64'),('Correction, K=1','k1'),('Correction, K=2','correction_16'),
        ('Correction, K=4','k4'),('K=2 without delayed exchange','no_exchange'),('F-to-A edge blocked','no_edge'),
        ('Memory reset before query','reset_memory'),('Memory shuffled: original target','shuffled_memory'),
        ('Same weights, unconditional publication','forced_always'),('Four consecutive live queries','repeated_queries')):
        md+=row([label,pct(mean(name))])
    md+=f'''
Shuffled memory scores **{pct(mean('shuffled_memory','counterfactual_accuracy'))}** against the answer implied by the substituted memory. Its entire latent publication and header are moved together. Blocked-edge, reset, shuffled, no-exchange and unconditional-publication controls have the same expert execution count as the intact K=2 condition. Four-value chance accuracy is 25%.

Four rounds reduce accuracy to **{pct(mean('k4'))}**. The combined model preserves the parent's learned two-round schedule; additional rounds are not yet a reliable refinement mechanism.

Gap 64 is four times the largest training gap. Across four consecutive queries, F's contextualizer state, expert recurrent state, payload, semantic header, version and provenance remain bit-for-bit unchanged. Each query receives fresh B/C tables; labels never enter its forward computation.

## Corrections after live queries

This sequence observes a fact, asks three independently drawn table queries, then continues the same producer state with a correction stream. A matched branch withholds the correction. Both branches receive the original fact again before the optional correction; this tests continued acquisition after queries, rather than a bare single-byte update.

| Sequence stage | Mean accuracy |
| --- | --- |
'''
    for name,label in (('before','Three queries before update'),('after','Query after correction'),('withheld','Query with correction withheld')):
        md+=row([label,pct(sum(records[s]['revision_after_queries'][name]['accuracy'] for s in SEEDS)/3)])
    md+='''
---

## Training and verification

Each seed receives 160 updates and 5,120 fresh fact episodes. Training alternates recall and correction over gaps 0, 4, 8 and 16. The objective combines final byte cross-entropy with local alignment of the edge output to the original query ingress representation of the observed fact byte. The alignment target comes from this project's own procedural observations and frozen ingress; it is used only in training.

| Seed | External events | Source occurrences | Expert executions | Training minutes |
| --- | --- | --- | --- | --- |
'''
    for r in train['runs']:md+=row([r['seed'],f"{r['external_events']:,}",f"{r['source_occurrences']:,}",f"{r['expert_executions']:,}",f"{r['training_seconds']/60:.2f}"])
    params=summary['parameters']
    md+=f'''
The graph has **{params['total_unique_parameters']:,} unique parameters**, **{params['physical_experts']} physical experts**, eight nodes and four regions. A fact byte executes 12 experts; a fresh table pair plus a two-round query executes 264 experts. Producer computation is skipped during queries. Training time is concurrent CPU wall time excluding validation and audits.

All **{test_count} regression tests pass**. The audit reconstructs all 480 training batches and table choices, checks checkpoint selection, verifies that only the new edge changed, and regenerates **{audit['prediction_groups']} held-out prediction groups**. Exported inference weights reproduce logits and complete state exactly. Optimizer continuation is tested bit-for-bit.

The original seven-node model still achieves **100% two-round composition on all 592 held-out cases**, with exact agreement in logits, state tensors and execution counts against the original source checkout.

## Scope and next work

This phase connects retained facts to live regional composition and measures the resulting dependence on memory, learned communication and recurrence. The fixed component weights make this an interface-integration experiment. It does not establish joint learning of all components, shared/private fact recovery, or natural-text learning in this combined graph; M7 remains open. The next step is to improve the weakest producer/connection pair through joint local learning while keeping these memory and composition controls.

## Reproduce

Extract both ZIPs into the same directory; each uses the SHoEbox/ root. Install requirements.txt, then run:

```bash
python scripts/audit_retained_composition.py --evaluations
python scripts/check_all.py
```

The artifact ZIP includes all four parent checkpoints, three resumable runs, selected models, inference exports, predictions, audits and the separate development pilot. The source ZIP is the exact published Git tree. Fresh training uses experiments.text_regions.retained_composition_train with the included parent checkpoints and the locked protocol.
'''
    (ROOT/'docs/SHoEbox_Retained_Composition_Report.md').write_text(md)
    result=ROOT/'experiments/text_regions/results/retained_composition';result.mkdir(parents=True,exist_ok=True)
    for name in ('protocol_lock.json','training_audit.json','evaluation_audit.json','compatibility.json','summary.json','environment.json'):
        shutil.copyfile(OUT/name,result/name)
    for seed in SEEDS:
        folder=result/str(seed);folder.mkdir(exist_ok=True)
        for name in ('protocol.json','history.json','evaluation.json'):shutil.copyfile(OUT/str(seed)/name,folder/name)
    (result/'README.md').write_text('''# Retained facts and live composition

Three frozen-component integration runs connect a retained producer to the existing seven-node recurrent graph through one learned latent edge. See the [report](../../../../docs/SHoEbox_Retained_Composition_Report.md) and [locked protocol](../../../../docs/SHoEbox_Retained_Composition_Protocol.md).

The JSON files preserve every prediction, target, table choice, exposure hash and selection record. The artifact attachment contains parent/checkpoint tensors and exact inference exports. Extract it beside the source ZIP, then run `python scripts/audit_retained_composition.py --evaluations` and `python scripts/check_all.py`.
''')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
