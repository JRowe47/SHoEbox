"""Render the completed EXP00 evidence without rerunning or selecting models."""
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiments.mad_next.tasks import Spec, dataset, fingerprint

HERE = Path(__file__).resolve().parent
NAMES = {'binding':'Two-key binding','initialize':'Initialization',
         'set':'Set','copy':'Copy','swap':'Swap','one_hop':'One-hop graph'}


def pct(x):
    return '—' if x is None else f'{100*x:.2f}%'


def span(values):
    return f'{pct(mean(values))} ({pct(min(values))}–{pct(max(values))})'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                       '| '+' | '.join(['---']*len(headers))+' |']+
                      ['| '+' | '.join(map(str,row))+' |' for row in rows])


def build():
    r = json.loads((HERE/'results.json').read_text())
    pilots = json.loads((HERE/'pilots.json').read_text())
    assert r['status'] == 'completed'
    assert len(r['runs'])*2 == r['plan']['fits']
    assert len(r['audits']) == len(r['calibration']['selected'])*len(r['plan']['seeds'])
    native_bytes = sum(x['native']['bytes'] for x in r['runs'])
    reference_bytes = sum(x['reference']['bytes'] for x in r['runs'])
    assert native_bytes+reference_bytes == r['plan']['training_bytes']
    tasks = [x['task'] for x in r['calibration']['selected']]
    groups = {t:sorted([a for a in r['audits'] if a['task']==t],key=lambda a:a['seed']) for t in tasks}
    for rows in groups.values():
        assert {x['seed'] for x in rows} == set(r['plan']['seeds'])
        assert len({x['test_sha256'] for x in rows}) == 1
    summary = {'tasks':{},'native_training_bytes':native_bytes,
               'reference_training_bytes':reference_bytes,
               'pilot_training_bytes':pilots['training_bytes'],
               'total_training_bytes':native_bytes+reference_bytes+pilots['training_bytes'],
               'formal_fits':r['plan']['fits'],'pilot_fits':pilots['fits'],
               'calibrated_settings':sum(x['eligible_for_architecture_comparison'] for x in r['calibration']['settings'])}
    for task,rows in groups.items():
        cell = next(c for c in r['plan']['cells'] if c['id'] == rows[0]['cell'])
        cases = dataset(Spec(**cell['spec']),r['plan']['test_examples'],'test',r['plan']['data_seed'])
        assert fingerprint(cases) == rows[0]['test_sha256']
        shortcut = rows[0]['shortcuts_test']
        strongest = max(shortcut,key=lambda k:shortcut[k]['exact_accuracy'])
        summary['tasks'][task] = {
            'native_exact_mean':mean(x['native']['native_exact_accuracy'] for x in rows),
            'restricted_exact_mean':mean(x['native']['exact_accuracy'] for x in rows),
            'reference_exact_mean':mean(x['reference']['exact_accuracy'] for x in rows),
            'reference_exact_min':min(x['reference']['exact_accuracy'] for x in rows),
            'strongest_shortcut':strongest,'shortcut_exact':shortcut[strongest]['exact_accuracy'],
            'cases_per_seed':rows[0]['test_cases'],
            'test_state_groups':len({e.split_key for e in cases}),
            'diagnostic_only':rows[0]['diagnostic_only']}
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(1,2,figsize=(14,7.5),sharey=True)
    y = np.arange(len(tasks))
    for ax in axes:
        ax.set_xlim(0,105)
        ax.set_xticks([0,25,50,75,100],['0%','25%','50%','75%','100%'])
        ax.set_axisbelow(True)
        ax.grid(axis='x',color='#e4e8ed')
        ax.set_yticks(y,[NAMES[t] for t in tasks])
    axes[0].barh(y,[100*summary['tasks'][t]['restricted_exact_mean'] for t in tasks],
                 height=.5,color='#315b7c',label='Corrected C: restricted answers')
    for i,t in enumerate(tasks):
        score=100*summary['tasks'][t]['shortcut_exact']
        axes[0].plot([score,score],[i-.32,i+.32],color='#bb7529',lw=3,
                     label='Strongest registered shortcut' if i==0 else None)
        values=[100*x['native']['exact_accuracy'] for x in groups[t]]
        axes[0].scatter(values,[i]*3,s=22,c='#162c40',zorder=3)
        label_x = max(values)+2
        if 0 <= score-label_x <= 8:
            label_x = score+2
        axes[0].text(label_x,i,f"{mean(values):.1f}%",va='center',fontsize=10)
    axes[1].barh(y,[100*summary['tasks'][t]['reference_exact_mean'] for t in tasks],
                 height=.5,color='#378579')
    for i,t in enumerate(tasks):
        values=[100*x['reference']['exact_accuracy'] for x in groups[t]]
        axes[1].scatter(values,[i]*3,s=22,c='#153f37',zorder=3)
        axes[1].text(2,i,f"{mean(values):.1f}%",va='center',color='white',fontweight='bold',fontsize=10)
    axes[1].axvline(95,color='#153f37',ls='--',lw=1,alpha=.7)
    axes[0].invert_yaxis()
    axes[0].set_title('SHoEbox vs shortcut\n250K training bytes per fit',loc='left',fontsize=13,pad=18)
    axes[1].set_title('Independent learned controls\n0.25M–128M training bytes per fit',loc='left',fontsize=13,pad=18)
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,-.12),frameon=False,fontsize=10)
    axes[1].text(0,-.14,'Dots: three seeds  •  Dashed line: 95%',transform=axes[1].transAxes,fontsize=10)
    fig.suptitle('EXP00 · held-out whole-episode accuracy',x=.05,ha='left',fontsize=21,fontweight='bold',y=.98)
    fig.text(.05,.91,'Balanced binding and atomic state tasks | Locked development selection | Fresh corrected-C models',fontsize=11,color='#52606b')
    fig.text(.05,.03,'Controls receive parsed records and have different training/state budgets. This is a learnability audit, not an efficiency ranking.\nC autoregression covers answer bytes; query and framing bytes are supplied. Family variants share an initial-state group.',fontsize=10,color='#52606b')
    fig.subplots_adjust(left=.15,right=.97,top=.79,bottom=.2,wspace=.13)
    fig.savefig(HERE/'SHoEbox_EXP00_Results.png',dpi=180,facecolor='white')
    plt.close(fig)

    metric_rows=[]
    for t,rows in groups.items():
        s=summary['tasks'][t]
        metric_rows.append([NAMES[t],pct(s['restricted_exact_mean']),pct(s['native_exact_mean']),
                            pct(s['shortcut_exact']),span([x['reference']['exact_accuracy'] for x in rows])])
    calibration_rows=[]
    for x in r['calibration']['settings']:
        calibration_rows.append([x['cell'],pct(x['native_mean_exact']),pct(x['strongest_shortcut_exact']),
                                 f"{x['shortcut_normalized_exact']:.4f}" if x['shortcut_normalized_exact'] is not None else 'undefined',
                                 pct(x['positive_mean_exact']),x['status']])
    category_rows=[]
    cf_rows=[]
    offset_rows=[]
    counts=[]
    for t,rows in groups.items():
        if t in ('set','copy','swap'):
            category_rows.append([NAMES[t],pct(mean(x['native']['changed_accuracy'] for x in rows)),
                                  pct(mean(x['native']['unchanged_accuracy'] for x in rows)),
                                  pct(mean(x['reference']['changed_accuracy'] for x in rows)),
                                  pct(mean(x['reference']['unchanged_accuracy'] for x in rows)),
                                  pct(rows[0]['shortcuts_test']['ignore_updates']['token_accuracy'])])
        cf_rows.append([NAMES[t],pct(mean(x['native_counterfactuals']['query_reversed']['exact_accuracy'] for x in rows)),
                        pct(mean(x['reference_counterfactuals']['query_reversed']['exact_accuracy'] for x in rows)),
                        pct(mean(x['native_counterfactuals']['operation_changed']['exact_accuracy'] for x in rows)) if t in ('set','copy','swap') else '—',
                        pct(mean(x['reference_counterfactuals']['operation_changed']['exact_accuracy'] for x in rows)) if t in ('set','copy','swap') else '—'])
        c=[mean(x['native_counterfactuals']['offsets'][i]['exact_accuracy'] for x in rows) for i in range(64)]
        ref=[mean(x['reference_counterfactuals']['offsets'][i]['exact_accuracy'] for x in rows) for i in range(64)]
        offset_rows.append([NAMES[t],pct(c[0]),f'{pct(min(c))}–{pct(max(c))}',
                            f'{pct(min(ref))}–{pct(max(ref))}'])
        run=next(x for x in r['runs'] if x['cell']==rows[0]['cell'])
        cell=next(x for x in r['plan']['cells'] if x['id']==rows[0]['cell'])
        counts.append([NAMES[t],f"{run['datasets']['train_cases']:,} / {run['datasets']['train_worlds']:,}",
                       f"{run['datasets']['dev_cases']:,} / {run['datasets']['dev_worlds']:,}",
                       f"{rows[0]['test_cases']:,} / {summary['tasks'][t]['test_state_groups']:,}",f"{cell['reference_budget']/1e6:g}M",
                       f"{run['reference']['inventory']['parameters']:,}"])
    lesion_rows=[]
    for t,rows in groups.items():
        lesion_rows.append([NAMES[t],pct(mean(x['reference_lesions']['no_query']['exact_accuracy'] for x in rows)),
                            pct(mean(x['reference_lesions']['zero_state']['exact_accuracy'] for x in rows)),
                            pct(mean(x['reference_lesions']['ignore_operation']['exact_accuracy'] for x in rows)) if t in ('set','copy','swap') else '—'])
    pilot_rows=[]
    for p in pilots['phases']:
        pilot_rows.append([p['name'],len(p['runs']),f"{sum(x['curve'][-1]['bytes'] for x in p['runs']):,}",
                           '; '.join(f"{x['task']} {pct(x['curve'][-1]['exact_accuracy'])}" for x in p['runs'])])
    native_cpu=sum(x['native']['cpu_seconds'] for x in r['runs'])
    ref_cpu=sum(x['reference']['cpu_seconds'] for x in r['runs'])
    inventory=r['runs'][0]['native']['inventory']
    binding=summary['tasks']['binding']
    text=f'''# SHoEbox EXP00 — calibration and learned positive controls

**Executed 16 November 2026 UTC · Plan published first as [`bd30bfc`](https://github.com/JRowe47/SHoEbox/commit/bd30bfc96f647155a44da2b98f413de8537cdb27).**

EXP00 is implemented and the formal training/audit is complete. **{summary['calibrated_settings']} of eight settings pass the comparative calibration gate.** On held-out balanced two-key binding, corrected C scores **{pct(binding['restricted_exact_mean'])} exact**, the input-order shortcut scores **50.00%**, and the learned attention control scores **{pct(binding['reference_exact_mean'])}**. The next step is EXP01's query/readout diagnosis; a broad architecture comparison has not passed its entry gate.

The six new fixture families now have independent label checks, grouped splits, shortcut scoring, trained controls and frozen counterfactual audits. Production model behavior and the original nine MAD tasks are unchanged. These are fresh small-model mechanism tests; no generalist capability label is promoted.

![Held-out exact episode results](https://raw.githubusercontent.com/JRowe47/SHoEbox/main/reports/exp00/SHoEbox_EXP00_Results.png)

## Held-out results

Scores are means over seeds 1, 7 and 19. Parentheses give the observed seed range, not a confidence interval. Each selected setting has at least 1,024 test episodes. Presentation siblings share an initial-state group and are not independent trials.

{table(['Task','C exact, restricted','C exact, 256-byte','Best shortcut exact','Learned control exact (seed range)'],metric_rows)}

Both C decode modes generate the scored answer bytes autoregressively; query/framing bytes remain observed. The learned controls receive parsed records and predict in the task alphabet. They establish learnability through that adapter, not native-byte or matched-compute equivalence.

On binding, C's teacher-forced answer-token score is {pct(mean(x['native']['token_accuracy'] for x in groups['binding']))}, while its exact autoregressive score is {pct(binding['restricted_exact_mean'])}. That gap is why whole-episode decoding and the input-order shortcut determine the gate.

## Development selection, locked before the audit

The gate uses `G = (C exact − strongest shortcut exact) / (1 − strongest shortcut exact)`. A comparative setting requires `0.1 ≤ G ≤ 0.9`, learned-control mean >=95%, every control seed >=90%, three seeds and at least 512 development episodes. Final C selection uses autoregressive scores; its saved learning curves are explicitly teacher forced. A below-band setting remains a diagnostic.

{table(['Setting','C exact','Shortcut exact','G','Control exact','Decision'],calibration_rows)}

The first/easiest declared setting was retained for each family when none qualified. Test results did not select settings, checkpoints, training budgets or hyperparameters. Binding uses vocabulary 2 for its diagnostic audit; vocabulary 4 and 8 remain in the development evidence.

## Atomic updates: partial correctness does not establish an operation

C category scores below are teacher forced; the control predicts queries independently. Changed targets and untouched registers are reported separately. Ignoring the operation scores 0% exact on every atomic task, even when its answer-token score looks high.

{table(['Task','C changed','C preserved','Control changed','Control preserved','Ignore-update token score'],category_rows)}

## Counterfactual and boundary checks

Query reversal changes only query order. Operation counterfactuals stay within the trained operator family: change the set value, reverse copy direction, or change the swap partner. All operations on a starting state remain in the same split. Scores here use one representative per initial-state group, up to 128 groups, so their denominator differs from the full balanced audit.

{table(['Task','C reversed queries','Control reversed queries','C changed operation','Control changed operation'],cf_rows)}

The 64-offset sweep uses up to 32 distinct held-out starting states per seed. Changing padding varies alignment and delay together; it is not a pure boundary-phase intervention. References omit padding in their syntax adapter, so their invariance is expected by construction.

{table(['Task','C at offset 0','C range over offsets','Control range over offsets'],offset_rows)}

Learned-control lesions, exact episodes:

{table(['Task','Query removed','State zeroed','Operation record erased'],lesion_rows)}

## What was built

- Balanced two-key binding; four-register initialization and individual set/copy/swap tasks; arbitrary three-node non-self one-hop mappings. Record and query positions are independently balanced using Cartesian cyclic rotations.
- Independent array-based interpreter alongside dictionary-based generation, including exhaustive repeated-value/no-op tiny cases and every offset 0..63.
- A learned attention compatibility table with value embeddings and a decoder; a 128-unit GRU with learned state transitions and register-addressed output heads. Syntax adapters never execute an instruction or read an answer label. Explicit register output addressing is a declared inductive bias.
- Protocol v2 folds grouped by task and concrete initial mapping. Operation changes and explicit presentation variants stay together; symmetric swap arguments share one semantic identity. This holds out initial mappings, not every isomorphic renaming or unseen program structure.
- Separate train/audit phases, exact charged-byte accounting, input fingerprints, source hashes, checkpoint restoration checks, source provenance and temporary checkpoint cleanup.

Implementation and reproduction commands: [`experiments/mad_next/README.md`](https://github.com/JRowe47/SHoEbox/blob/main/experiments/mad_next/README.md).

## Budgets and reproducibility

C is the unmodified corrected-C runtime: **{inventory['parameters']:,} fitted parameters**, four scales, width 64, no extra refinement passes. Requested capacity is 500K; bank granularity accounts for the difference. Its measured inventory includes {inventory['recurrent_vector_bytes']:,} recurrent-vector bytes, {inventory['minkron_matrix_bytes']:,} MiKron-matrix bytes, {inventory['address_matrix_bytes']:,} address-matrix bytes, {inventory['retained_record_bytes']:,} retained-record bytes and {inventory['child_buffer_bytes']:,} child-buffer bytes. The complete inventory is recorded for every fit.

{table(['Selected family','Train episodes / state groups','Dev episodes / state groups','Test episodes / state groups','Control bytes per seed','Control parameters'],counts)}

Each C fit receives 250K encoded input bytes. References use Adam at 0.05 for attention and 0.003 for the GRU, batch 32, norm clipping 1. Their budgets count the original encoded bytes, including padding, although their parser omits padding from neural inputs. Partial final episodes are charged without revealing their labels. State and optimizer budgets differ; no efficiency ranking is inferred.

- **Formal run:** {r['plan']['fits']} fits; **{native_bytes+reference_bytes:,} bytes**, comprising {native_bytes:,} for C and {reference_bytes:,} for learned controls.
- **Development-only pilots:** {pilots['fits']} additional fits; **{pilots['training_bytes']:,} bytes**.
- **Total including exploration:** {r['plan']['fits']+pilots['fits']} fits; **{summary['total_training_bytes']:,} charged bytes**.
- Formal training CPU time: C {native_cpu:.1f}s; controls {ref_cpu:.1f}s. Overall train-phase elapsed time: {r['training_elapsed_wall_seconds']:.1f}s with two pinned workers. Evaluation/initialization is outside training timers; model timing scopes differ.
- Environment: Python {r['provenance']['python']}, NumPy {r['provenance']['numpy']}, PyTorch {r['provenance']['torch']}. Data seed 20261115; finite-epoch order seed 1067. Complete fingerprints and source hashes are in the raw result.

The original 31M planning estimate covered a later binding tranche. This run separately registered expanded learned-control budgets and additional atomic calibration tasks; it did not execute that four-arm tranche. The first small-data pilots exposed control-training limitations, so larger datasets and longer budgets were registered before the formal test audit. Earlier v1 results use a superseded, narrower split and are not combined with v2 scores:

{table(['Pilot phase','Fits','Charged bytes','Final development exact scores'],pilot_rows)}

Pilot recipes and their exact fixture source are preserved under `reports/exp00/pilot_source_v1/`; their manifest is `pilots.json`. No weights, generated datasets or compiled artifacts are committed.

## Verification

**29 Python tests passed**: 15 existing MAD tests and 14 EXP00 tests. **23 native checks passed under AddressSanitizer/UndefinedBehaviorSanitizer.** The formal fixture checks passed **14,848 exhaustive tiny cases** and **864 oracle cases covering the original nine tasks**. Tests cover label exclusion, balanced positions, grouped split isolation, counterfactual semantics, all offsets, exact partial-byte accounting, native category-mask equivalence and optimizer/checkpoint continuation. Each formal fit also checks checkpoint prediction restoration.

## Next development action

Run **EXP01 as a diagnostic repair** using this fixed balanced binding fixture. First check a single binding, then trace values through write, retained/addressed state, query-conditioned reads and the native answer. Compare identically fitted raw/conditioned frozen probes and a matched-capacity generic head before adding working slots. C's failure on initialization makes broad state-update or deeper-loop searches premature. Re-enter comparative architecture testing only after a setting clears the shortcut-aware calibration gate.

Evidence: [raw results](https://github.com/JRowe47/SHoEbox/blob/main/reports/exp00/results.json), [summary](https://github.com/JRowe47/SHoEbox/blob/main/reports/exp00/summary.json), [pilot ledger](https://github.com/JRowe47/SHoEbox/blob/main/reports/exp00/pilots.json), and [published development plan](https://github.com/JRowe47/SHoEbox/blob/main/docs/capabilities/SHoEbox_Capability_Research.md).
'''
    (HERE/'README.md').write_text(text)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    build()
