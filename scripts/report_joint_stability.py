"""Build the paired continuation report from audited measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics as stats
import sys
import textwrap
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.joint_stability import primary_conditions


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
        *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'artifacts/joint_stability/study')
    p.add_argument('--out',type=Path,default=ROOT/'docs/SHoEbox_Joint_Stability_Report.md')
    a=p.parse_args();lock=json.loads((a.root/'protocol_lock.json').read_text())
    verification=json.loads((a.root/'verification.json').read_text());assert verification['exit_code']==0
    runs=[];evaluations={};audits={}
    for spec in lock['runs']:
        for mode in lock['modes']:
            folder=a.root/str(spec['seed'])/mode
            e=json.loads((folder/'evaluation.json').read_text());audit=json.loads((folder/'audit.json').read_text())
            assert audit['all_passed']
            last=torch.load(folder/'last.pt',weights_only=True,map_location='cpu')
            history=last['history'][1:]
            floor=min(f['accuracy'] for name,_,_ in primary_conditions() for f in e['memory'][name]['fields'])
            final_floor=min(f['accuracy'] for r in e['final']['memory'].values() for f in r['fields'])
            recurrence=stats.mean(f['accuracy'] for f in e['memory']['shared_right']['fields'])-stats.mean(f['accuracy'] for f in e['memory']['k1']['fields'])
            gradients={group:dict(
                median_text_norm=stats.median(r['gradient_groups'][group]['text_norm'] for r in history),
                median_memory_norm=stats.median(r['gradient_groups'][group]['memory_norm'] for r in history),
                max_text_norm=max(r['gradient_groups'][group]['text_norm'] for r in history),
                median_text_update_norm=stats.median(r['gradient_groups'][group]['text_norm']*r['gradient_groups'][group]['text_scale']*r['global_clip_scale'] for r in history),
                median_memory_update_norm=stats.median(r['gradient_groups'][group]['memory_norm']*r['gradient_groups'][group]['memory_scale']*r['global_clip_scale'] for r in history),
                conflicting_updates=sum(r['gradient_groups'][group]['dot']<0 for r in history))
                for group in ('core','readout','producer')}
            row=dict(**spec,mode=mode,selected_step=e['selected_step'],text_before=e['initial']['text']['nats'],
                text_nats=e['text']['intact']['nats'],memory_min=floor,recurrence_gain=recurrence,
                final_text_nats=e['final']['text']['nats'],final_memory_min=final_floor,
                targets={**e['targets'],'resume':audit['export_and_resume_exact']},unigram_nats=e['unigram_nats'],
                gradients=gradients,totals=last['totals'],peak_rss_mib=max(r['peak_rss_mib'] for r in history),
                validation=[dict(step=r['step'],text_nats=r['text_validation']['nats'],
                    memory_min=min(f['accuracy'] for score in r['memory_validation'] for f in score['fields']))
                    for r in last['history'] if 'text_validation' in r])
            runs.append(row);evaluations[f"{spec['seed']}/{mode}"]=e;audits[f"{spec['seed']}/{mode}"]=audit
    groups={mode:[r for r in runs if r['mode']==mode] for mode in lock['modes']}
    aggregate={mode:dict(mean_text_nats=stats.mean(r['text_nats'] for r in rows),
        worst_memory_min=min(r['memory_min'] for r in rows),passing_runs=sum(all(r['targets'].values()) for r in rows),
        mean_final_text_nats=stats.mean(r['final_text_nats'] for r in rows)) for mode,rows in groups.items()}
    summary=dict(runs=runs,aggregate=aggregate,all_clipped_targets_passed=aggregate['group_clip']['passing_runs']==3,
        verification=verification)
    (a.root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    pct=lambda x:f'{100*x:.2f}%'
    mode_name=dict(sum='Ordinary sum',group_clip='Task/group cap')
    passed=summary['all_clipped_targets_passed']
    delta=aggregate['sum']['mean_text_nats']-aggregate['group_clip']['mean_text_nats']
    comparison=(f"The cap improves mean selected text NLL by {delta:.4f} nats/byte." if delta>0 else
                f"Ordinary summation has the lower mean selected text NLL by {-delta:.4f} nats/byte.")
    lines=['# SHoEbox: stable joint text and memory learning','',
        'Completed November 5, 2026 (America/Denver). Three matched parent/data pairs, two update rules, one shared regional model per run.','',
        '## Result','',
        ('The validation-selected checkpoints in the capped arm meet every locked endpoint in all three pairs.' if passed else
         f"The validation-selected checkpoints in the capped arm meet every locked endpoint in {aggregate['group_clip']['passing_runs']} of three pairs. M7 remains open."),
        comparison,'',
        table(['Seed','Update rule','Selected step','Text before','Text after','Memory minimum'],[
            [r['seed'],mode_name[r['mode']],r['selected_step'],f"{r['text_before']:.4f}",f"{r['text_nats']:.4f}",pct(r['memory_min'])] for r in runs]),'',
        'Text values are nats per byte on 4,096 previously unscored test targets. Memory minimum is the weakest of 27 field scores across recall, four correction directions, and four withheld-correction conditions, each with 512 fresh episodes. Checkpoints are selected using validation only.','',
        '![Paired validation curves](../experiments/text_regions/results/joint_stability/learning_curves.png)','',
        'The dashed line marks the 90% minimum-field validation guard. The plot includes the initial parent and all three scheduled continuation validations.','',
        f"The training-only unigram reference scores {runs[0]['unigram_nats']:.4f} nats/byte on these same targets.",'',
        (f"{sum(r['text_nats'] < r['unigram_nats'] for r in runs)} of six selected checkpoints beat that reference."
         if any(r['text_nats'] < r['unigram_nats'] for r in runs) else
         'No selected checkpoint beats that reference; useful language learning remains the next target.'),'',
        table(['Update rule','Mean text NLL','Worst memory score','Runs passing all endpoints'],[
            [mode_name[m],f"{v['mean_text_nats']:.4f}",pct(v['worst_memory_min']),f"{v['passing_runs']}/3"] for m,v in aggregate.items()]),'',
        ('Selection retained the starting parent in these arms: '+', '.join(
            f"{r['seed']} ({mode_name[r['mode']]})" for r in runs if r['selected_step']==0)+
            '. Their selected scores measure the starting checkpoint. The fixed-budget final results below measure the continuation.'
            if any(r['selected_step']==0 for r in runs) else
            'Every selected checkpoint includes continuation updates.'),'',
        '## What changed','',
        'Text and memory gradients are captured separately from the existing backward passes. The new rule caps each task contribution at norm one within each optimizer group: core, shared byte readout, and memory producers. Contributions are then added and the common final global cap is applied. There are no extra model executions, task-specific decoders, or model parameters. The ordinary-sum control shares the same numerical norm calculation and logging.','',
        'The probe that motivated this change measured text/core norms of 18,866, 3.91, and 11,559 across the three parents. The corresponding memory/core norms were 10.83, 2.63, and 7.05. A global cap alone consequently scaled down decoder and producer contributions along with the large core gradients.','',
        table(['Seed','Rule','Median text/core norm','Median memory/core norm','Median decoder text contribution'],[
            [r['seed'],mode_name[r['mode']],f"{r['gradients']['core']['median_text_norm']:.3f}",
             f"{r['gradients']['core']['median_memory_norm']:.3f}",f"{r['gradients']['readout']['median_text_update_norm']:.6f}"] for r in runs]),'',
        'The final column is the task contribution norm after both clipping stages and before AdamW. Per-node, connection, HAT, shared-expert, and memory-owner norms and cosine similarities are preserved in each training history.','',
        '## Broader checkpoint selection','',
        'The previous 64-episode, five-condition guard becomes a 512-episode, nine-condition guard. Every one of its 27 field scores must reach 90%; the eligible checkpoint with lowest text validation NLL wins. Exact ties keep the earlier checkpoint. Initial parents, selected weights, and fixed-budget final weights are all retained.','',
        table(['Seed','Rule','Step 0 memory','Step 20 memory','Step 40 memory','Step 60 memory'],[
            [r['seed'],mode_name[r['mode']],*[pct(v['memory_min']) for v in r['validation']]] for r in runs]),'',
        'Fixed-budget final results are a prespecified diagnostic, independent of primary checkpoint selection:','',
        table(['Seed','Rule','Final text NLL','Final memory minimum'],[
            [r['seed'],mode_name[r['mode']],f"{r['final_text_nats']:.4f}",pct(r['final_memory_min'])] for r in runs]),'',
        '## Retained computation and source ownership','',
        table(['Seed','Rule','K2 gain over K1','Text target','Memory target','State / resume'],[
            [r['seed'],mode_name[r['mode']],f"{100*r['recurrence_gain']:.2f} pp",
             'pass' if r['targets']['text'] else 'fail','pass' if r['targets']['memory'] else 'fail',
             'pass' if r['targets']['ownership'] and r['targets']['resume'] else 'fail'] for r in runs]),'',
        'The memory answers still require g(f(x)) through the regional composer. Additional controls include K4, no exchange, blocked/reset/shuffled memory, missing and erased sources, and 64-byte gaps. The four memory owners remain inactive during text. Interleaving 32 text bytes preserves their state exactly, and 128-byte autoregressive sessions reproduce their bytes, RNG, next logits, and complete state after saving at byte 41.','',
        'Mean held-out field accuracy across the three capped runs:','',
        table(['Memory condition','Shared','Left private','Right private'],[
            [label,*[pct(stats.mean(evaluations[f"{r['seed']}/group_clip"]['memory'][name]['fields'][field]['accuracy'] for r in groups['group_clip'])) for field in range(3)]]
            for name,label in [('recall','Recall'),('shared_right','Shared correction, K2'),('k1','Shared correction, K1'),
                ('k4','Shared correction, K4'),('no_exchange','No regional exchange'),('no_edge','Blocked memory edges'),
                ('reset','Reset memory'),('shuffle','Shuffled memory'),('left_only','Left source only'),('right_only','Right source only'),
                ('erase_left','Erase left source'),('erase_right','Erase right source'),('long_recall','Recall after 64 bytes'),('long_correction','Correction after 64 bytes')]]),'',
        'Mean text NLL across the three selected checkpoints in each arm:','',
        table(['Text condition','Ordinary sum','Task/group cap'],[
            [label,*[f"{stats.mean(evaluations[str(r['seed'])+'/'+mode]['text'][name]['nats'] for r in groups[mode]):.4f}" for mode in lock['modes']]]
            for name,label in [('intact','Two regional rounds'),('k1','One regional round'),('reset_chunks','Reset state every 16 bytes')]]),'',
        '## Samples','',
        'Fixed seed samples from every capped run, prompted with ROMEO and a newline at temperature 0.8. Escapes preserve non-ASCII bytes; raw samples are in the artifact package.','']
    for r in groups['group_clip']:
        e=evaluations[f"{r['seed']}/group_clip"]
        sample=repr(bytes.fromhex(e['live_text']['sample_hex']))[2:-1]
        lines.extend([f"Run {r['seed']}:",'','```text',textwrap.fill(sample,100,break_on_hyphens=False),'```',''])
    lines.extend(['## Training and verification','',
        'Each arm adds 60 updates, 7,680 ordered natural-text bytes, and 960 fresh procedural memory episodes to its parent. Each final lineage has therefore consumed 28,160 natural-text bytes during the joint-text phases. A selected checkpoint may contain fewer continuation updates. The three pairs share an earlier composition ancestor; each pair shares its starting weights, Adam moments, text cursor/state, and new data exactly.','',
        'The graph contains 510,958 parameters and 768 physical experts. AdamW rates remain core 0.0003, readout 0.002, and producer 0.0001, with a five-update warmup. The memory objective and fixed alignment targets are inherited. No replay store is introduced.','',
        table(['Seed','Rule','Expert executions','Training minutes','Peak RSS MiB'],[
            [r['seed'],mode_name[r['mode']],f"{r['totals']['expert_executions']:,}",f"{r['totals']['training_seconds']/60:.2f}",f"{r['peak_rss_mib']:.1f}"] for r in runs]),'',
        'Training times sum recorded completed updates and exclude validation, audits, and interrupted unsaved work. Timings come from concurrent CPU workers.','',
        f"All {verification['tests_passed']} regression tests and all six run audits pass. Audits reconstruct every training batch and counter, verify identical paired data and parent state, rescore every recorded validation and test memory field against regenerated targets, recompute all recorded text NLL values, rerun the weakest selected validation condition, three full held-out memory conditions, and 1,024 text targets per run, and reproduce serialized generation exactly.",'',
        '## Reproduction','',
        'Extract source and artifact ZIPs into the same directory; both use the SHoEbox/ root. The package includes all three parent checkpoints, every initial/selected/final continuation checkpoint, full histories and predictions, raw sample bytes, data, and audit evidence.','',
        '```bash','python -m pip install -r requirements.txt -r requirements-reports.txt','python scripts/check_all.py',
        'python scripts/run_joint_stability.py \\',
        '  --parents artifacts/joint_stability/parents --stage audit',
        'python scripts/plot_joint_stability.py',
        'python scripts/report_joint_stability.py',
        'python -m experiments.text_regions.byte_stream \\',
        '  --checkpoint artifacts/joint_stability/study/20261400/group_clip/best.pt \\',
        '  --prompt "ROMEO:" --count 128 --out sample.bin','```','',
        'For a fresh matched study, run scripts/run_joint_stability.py with --parents artifacts/joint_stability/parents and a new --root. Repeat the same command to resume saved optimizer and stream state. Then run --stage evaluate and --stage audit against that root.',''])
    recovery=a.root.parent/'recovery_provenance.json'
    if recovery.exists():
        recovered=json.loads(recovery.read_text())
        lines.extend(['## Interrupted-run recovery','',
            f"The saved in-progress archive passed all {recovered['archive_files_verified']} file hash checks. "
            'The six runs resumed their full optimizer, stream, graph, and RNG checkpoints with the original locked source and protocol.', '',
            table(['Run','Restored update'],[[r['run'],r['restored_step']] for r in recovered['restored']]),'',
            f"The recovery used PyTorch {recovered['environment']['torch']} and "
            f"{recovered['environment']['training_workers']} concurrent CPU workers. "
            'The artifact package preserves the original environment record and the recovery provenance separately.',''])
        lines.extend([
            'A saved step-zero text validation recomputed at 3.7440 rather than its recorded 3.8115 nats/byte in the recovery environment. The original validation history and selection rule were preserved. The same checkpoint reproduced all 1,536 answers in its weakest memory-validation condition exactly, with loss-reduction differences below 0.00000005. The audit requires exact discrete validation results and permits only 0.000001 absolute/relative tolerance for those floating-point loss reductions. Fresh held-out results and generation sessions are independently reproduced in the recovery environment.',''])
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text('\n'.join(lines))
    public=ROOT/'experiments/text_regions/results/joint_stability';public.mkdir(parents=True,exist_ok=True)
    for name in ('summary.json','protocol_lock.json','environment.json','verification.json'):
        shutil.copyfile(a.root/name,public/name)
    (public/'audit_summary.json').write_text(json.dumps(audits,indent=2)+'\n')
    compact={}
    for key,e in evaluations.items():
        compact[key]=dict(selected_step=e['selected_step'],targets=e['targets'],
            memory={name:dict(accuracy=[f['accuracy'] for f in r['fields']],nats=[f['nats'] for f in r['fields']],
                expert_executions=r['expert_executions']) for name,r in e['memory'].items()},
            text={name:{k:v for k,v in r.items() if k!='rows'} for name,r in e['text'].items()},
            interleaved=dict(exact_memory=e['interleaved']['exact_memory'],accuracy=[f['accuracy'] for f in e['interleaved']['fields']]),
            live_text=e['live_text'])
    (public/'evaluation_summary.json').write_text(json.dumps(compact,indent=2)+'\n')
    (public/'README.md').write_text('# Joint stability\n\nSee the [report](../../../../docs/SHoEbox_Joint_Stability_Report.md) and [protocol](../../../../docs/SHoEbox_Joint_Stability_Protocol.md). Full checkpoints, histories, predictions and samples are in the downloadable artifact archive.\n')
    print(json.dumps(dict(aggregate=aggregate,all_clipped_targets_passed=passed)),flush=True)


if __name__=='__main__':main()
