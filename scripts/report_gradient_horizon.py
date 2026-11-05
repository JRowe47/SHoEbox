"""Build the Markdown result tables directly from audited experiment records."""
import argparse
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assessment',required=True)
    p.add_argument('--next-step',required=True)
    args=p.parse_args()
    root=ROOT/'artifacts/gradient_horizon'
    load=lambda path:json.loads((root/path).read_text())
    arms=('truncated','full')
    evaluations={arm:load(arm+'/evaluation.json') for arm in arms}
    selection=load('selection.json');audit=load('evaluation_audit.json')
    assert selection['all_passed'] and audit['all_passed']
    selected=selection['selected']
    histories={arm:load(arm+'/history.json') for arm in arms}
    summaries={arm:load(arm+'/summary.json') for arm in arms}
    diagnostics={arm:load(arm+'/local_diagnostic.json')['records'] for arm in arms}
    tests=sum(map(int,re.findall(r'Ran (\d+) tests?',(root/'regression.log').read_text())))
    assert 'All five experiment suites passed.' in (root/'regression.log').read_text()
    pct=lambda value:f'{100*value:.2f}%'
    def table(headers, rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                          *['| '+' | '.join(map(str,row))+' |' for row in rows]])
    scores={a:evaluations[a]['test_capabilities']['scores'] for a in arms}
    labels={'composition':'Composition, K=2','recall':'Recall, gap 16','correction':'Correction, gap 16',
            'shared':'Shared fact','left':'Left-private fact','right':'Right-private fact',
            'a_only/shared':'A-only shared','a_only/left':'A-only left-private',
            'b_only/shared':'B-only shared','c_only/shared':'C-only shared','c_only/right':'C-only right-private'}
    rows=[[label,pct(scores['truncated'][key]),pct(scores['full'][key]),
           f"{100*(scores['full'][key]-scores['truncated'][key]):+.2f} pp"] for key,label in labels.items()]
    pages=[f'''# SHoEbox: gradient horizon

Completed November 5, 2026. Matched complete-episode training from the regional-facts control checkpoint.

## Result

{args.assessment}

{table(['Held-out task','Truncated','Full episode','Change'],rows)}

Both arms complete 32 optimizer updates, with 2,048 composition cases and 512 complete episodes of each stream task. The sole paired intervention is whether gradients cross the 16-event chunk boundary. Each held-out stream condition/query uses 512 fresh episodes; composition uses all 592 held-out cases. Predictions use the common 256-byte output head. Four-value chance accuracy is 25%.

Validation selects **{selected['checkpoint']} at phase update {selected['step']}**. Endpoint scores above are reported regardless of checkpoint selection. Test scores never select a checkpoint.
''']
    valrows=[]
    for arm in arms:
        for row in histories[arm]:
            if 'validation' not in row:continue
            v=row['validation']
            valrows.append([arm,row['step'],pct(v['scores']['composition']),
                            'pass' if v['composition_preserved'] else 'fail',pct(v['minimum_accuracy'])])
    controls={'k1':'One round','k2':'Two rounds','k4':'Four rounds',
              'k2_no_exchange':'Exchange disabled','k2_shuffled_bridge':'Changed table, original target',
              'k2_counterfactual':'Changed table, changed target','k2_reset':'State reset before query'}
    comprows=[[label,*[pct(evaluations[a]['composition_controls'][key]['accuracy']) for a in arms]]
              for key,label in controls.items()]
    pages.append(f'''## Validation and composition

{table(['Arm','Update','Composition','All composition gates','Weakest task'],valrows)}

Selection ranks all capability gates, composition preservation, minimum task accuracy, macro accuracy and lower macro NLL, in that order. Composition preservation includes both local lookups and exact table retention, as well as the final answer. A high answer score alone does not pass all gates.

{table(['Composition control','Truncated','Full episode'],comprows)}

Both arms preserve the seven-node graph, sparse expert banks, recurrent state format and common byte decoder. They inherit identical weights, optimizer moments and composition RNG. New stream episodes start from reset state with seed 20261109; partial parent stream states are intentionally discarded in both arms.

This pair also changes episode loss weighting and optimizer frequency relative to the preceding phase. Those changes are shared by both arms. The paired comparison isolates gradient detachment; comparisons to the earlier phase do not isolate it.
''')
    memrows=[]
    for gap in ('16','32'):
        for task in ('recall','correction'):
            for control in ('carried','reset_query'):
                memrows.append([f'{task}, {gap}, '+('carried' if control=='carried' else 'reset'),
                    *[pct(evaluations[a]['memory_controls'][gap][task]['controls'][control]['exact_accuracy']) for a in arms]])
            if task=='correction':
                memrows.append([f'withheld correction, {gap}, old value',
                    *[pct(evaluations[a]['memory_controls'][gap][task]['controls']['withhold_update']['old_value_accuracy']) for a in arms]])
    localrows=[]
    for task in ('recall','correction'):
        for node in ('P','Q'):
            row=[task+': '+node]
            for arm in arms:
                x=diagnostics[arm][task]['nodes'][node]['current']
                row.append(pct(x['write_accuracy'])+' / '+pct(x['accuracy']))
            localrows.append(row)
    pages.append(f'''## Delayed memory and write retention

{table(['Memory condition','Truncated','Full episode'],memrows)}

Gap 16 is the training setting; gap 32 tests a longer delay. Query reset removes carried state immediately before the answer. Withheld corrections are scored against the old observed value here; raw records also include agreement with the unseen new value.

{table(['Local memory head','Truncated: write / query','Full: write / query'],localrows)}

Local diagnostics use frozen training-only heads on 64 held-out episodes per task after selection. They measure whether regional publications contain readable values at the observed write and the final query. They do not supply inference answers.
''')
    factrows=[]
    for node,factor in (('P','shared'),('Q','shared'),('P','left'),('Q','right')):
        row=[node+': '+factor]
        for arm in arms:
            x=diagnostics[arm]['views/'+factor]['nodes'][node][factor]
            row.append(pct(x['write_accuracy'])+' / '+pct(x['accuracy']))
        factrows.append(row)
    budgetrows=[]
    for key,label in (('expert_executions','Expert executions'),('external_events','External events'),
                      ('source_occurrences','Observed source occurrences'),('composition_episodes','Composition cases'),
                      ('correction_final_queries','Complete correction episodes'),('views_final_queries','Complete fact episodes')):
        budgetrows.append([label,*[f'{summaries[a][key]:,}' for a in arms]])
    budgetrows.append(['Training time, minutes',*[f"{summaries[a]['training_seconds']/60:.2f}" for a in arms]])
    pages.append(f'''## Fact acquisition and verification

{table(['Local fact head','Truncated: write / query','Full: write / query'],factrows)}

{table(['Training budget','Truncated','Full episode'],budgetrows)}

All **{tests} regression tests** pass. Tests cover forward/state/cost equality, full-gradient equivalence to an unsliced episode in FP64, final-query credit reaching an earlier write only with the full horizon, loss-only labels and exact interrupted/resumed updates. FP64 avoids cancellation in the untrained fixture's large normalization gradients; training uses FP32.

The audit reconstructs each training batch from the generators, compares all data hashes and budgets, verifies parent weights and optimizer moments, and recomputes validation selection. It also verifies {audit['prediction_groups_recomputed']} held-out prediction groups and local diagnostic arithmetic. Exports remove all four auxiliary heads and reproduce checked inference logits and states exactly; the inference graph has 252,478 parameters.

## Next development step

{args.next_step}

M7 remains open. The integrated milestone also requires multi-seed replication and the natural-text experiment.

## Reproduce

Extract the source and artifact ZIPs into the same directory. Both use the SHoEbox/ root. Install requirements.txt, then run:

```bash
python scripts/audit_gradient_horizon.py --evaluations
```

The locked protocol contains fresh-run commands. Append --resume to the matching training command to recover a partial run. The artifact archive includes the parent, complete optimizer checkpoints, validation endpoints, inference exports, per-update histories, predictions, diagnostics and a SHA-256 manifest identifying the published source commit and tree.
''')
    path=ROOT/'docs/SHoEbox_Gradient_Horizon_Report.md'
    path.write_text('\n---\n\n'.join(pages))
    print(path)


if __name__=='__main__':main()
