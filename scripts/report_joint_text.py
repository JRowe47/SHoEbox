"""Summarize the locked three-pair joint-text study without selecting on test."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import torch
ROOT=Path(__file__).resolve().parents[1]
KINDS=('recall','shared_left','shared_right','left_private','right_private')
PRIMARY=KINDS+tuple('withheld_'+k for k in KINDS[1:])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'artifacts/joint_text')
    p.add_argument('--out',type=Path,default=ROOT/'docs/SHoEbox_Joint_Text_Report.md')
    a=p.parse_args();lock=json.loads((a.root/'protocol_lock.json').read_text());runs=[]
    verification=json.loads((a.root/'verification.json').read_text());assert verification['exit_code']==0
    for spec in lock['runs']:
        folder=a.root/str(spec['seed']);e=json.loads((folder/'evaluation.json').read_text());audit=json.loads((folder/'audit.json').read_text())
        assert audit['all_passed'],'Complete the reconstruction audit before reporting.'
        best=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        last=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        primary=min(f['accuracy'] for name in PRIMARY for f in e['memory'][name]['fields'])
        recurrence=statistics.mean(f['accuracy'] for f in e['memory']['shared_right']['fields'])-statistics.mean(f['accuracy'] for f in e['memory']['k1']['fields'])
        text=e['text']['intact']['nats'];initial=e['text']['initial']['nats']
        targets=dict(text=text<4 and text<initial*.5,memory=primary>=.9,recurrence=recurrence>=.05,
            ownership=e['live_text']['exact_memory'] and e['interleaved']['exact_memory'],resume=audit['export_and_resume_exact'])
        runs.append(dict(**spec,selected_step=best['step'],initial_text_nats=initial,text_nats=text,
            unigram_nats=e['unigram_nats'],primary_min=primary,recurrence_gain=recurrence,targets=targets,
            selected_five_task_min=min(f['accuracy'] for name in KINDS for f in e['memory'][name]['fields']),
            final_text_nats=e['final']['text']['nats'],
            final_five_task_min=min(f['accuracy'] for v in e['final']['memory'].values() for f in v['fields']),
            totals=last['totals'],audit=audit,evaluation=e,
            validation_curve=[dict(step=r['step'],text_nats=r['text_validation']['nats'],
                memory_min=min(f['accuracy'] for v in r['memory_validation'] for f in v['fields']))
                for r in last['history'] if 'text_validation' in r],
            max_rss_mib=max(r.get('peak_rss_mib',0) for r in last['history']),
            optimizer_bytes=max(r.get('optimizer_bytes',0) for r in last['history'])))
    passed=all(all(r['targets'].values()) for r in runs)
    coverage=[]
    for path in sorted((a.root/'diagnostics').glob('validation_*.json')):
        d=json.loads(path.read_text())
        assert all(d[k] for k in ('saved_validation_exact','validation_batch_predictions_exact','test_batch_predictions_exact'))
        assert d['checkpoint_sha256']==hashlib.sha256((a.root/str(d['seed'])/'last.pt').read_bytes()).hexdigest()
        coverage.append({**{k:v for k,v in d.items() if k not in ('original','expanded','test_subset')},
            'original_count':d['original']['count'],'expanded_count':d['expanded']['count'],
            'original_accuracy':[f['accuracy'] for f in d['original']['fields']],
            'expanded_accuracy':[f['accuracy'] for f in d['expanded']['fields']]})
    summary=dict(all_targets_passed=passed,validation_coverage=coverage,
        runs=[{k:v for k,v in r.items() if k not in ('evaluation','audit')} for r in runs])
    (a.root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    mean=lambda values:statistics.mean(values)
    fmt=lambda x:f'{100*x:.2f}%'
    field_means=lambda name:[mean(r['evaluation']['memory'][name]['fields'][i]['accuracy'] for r in runs) for i in range(3)]
    table=lambda headers,rows:'\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,row))+' |' for row in rows]])
    sample=repr(bytes.fromhex(runs[0]['evaluation']['live_text']['sample_hex']))[2:-1]
    candidates=[r for r in runs if r['final_text_nats']<4 and r['final_five_task_min']>=.9]
    diagnostic_note=('Final diagnostics meeting both the text NLL and five-task memory thresholds: '+
        '; '.join(f"run {r['seed']}: {r['final_text_nats']:.4f} nats/byte, {fmt(r['final_five_task_min'])} memory minimum, versus {fmt(r['validation_curve'][-1]['memory_min'])} memory on validation" for r in candidates)+'. '
        if candidates else 'No final checkpoint meets both diagnostic thresholds. ')
    diagnostic_note+='These final memory diagnostics cover five recall/correction tasks; they do not include the four withheld-correction conditions in the selected primary evaluation.'
    status='passed' if passed else 'not all passed'
    lines=['# SHoEbox: joint text and retained-memory learning','',
        f'Completed November 5, 2026 (America/Denver). Locked phase targets: **{status}**. Three parent/data pairs, one eleven-node graph and one shared 256-byte decoder per pair.','',
        '## Result','',
        table(['Run seed','Selected update','Text NLL before','Text NLL after','Memory minimum'],
            [[r['seed'],r['selected_step'],f"{r['initial_text_nats']:.4f}",f"{r['text_nats']:.4f}",fmt(r['primary_min'])] for r in runs]),'',
        f"Mean held-out text loss is **{mean(r['text_nats'] for r in runs):.4f} nats/byte** ({mean(r['evaluation']['text']['intact']['bits'] for r in runs):.4f} bits/byte). The unigram reference scores **{mean(r['unigram_nats'] for r in runs):.4f} nats/byte** on the same bytes. The initial parent was specialized for four-value composition, so its large text loss is a transfer starting point, not a language-model baseline.",'',
        'Each test evaluates 4,096 natural-text target bytes and 512 fresh episodes for every structured condition. Selection uses validation only. The three parents share the same frozen composition ancestor; their producer weights and new synthetic training episodes differ.','',
        'The validation trajectory records the full training run, including checkpoints rejected by the memory guard.','',
        table(['Update','Mean text validation NLL','Worst memory validation score'],
            [[rows[0]['step'],f"{mean(r['text_nats'] for r in rows):.4f}",fmt(min(r['memory_min'] for r in rows))]
             for rows in zip(*(r['validation_curve'] for r in runs))]),'',
        'The fixed-budget final weights are evaluated separately to show what continued joint learning did. These diagnostics were declared during training after observing validation interference, and do not replace the locked primary checkpoint selection. The memory columns compare the same five recall/correction tasks on the same episodes.','',
        table(['Run seed','Final text NLL','Selected five-task minimum','Final five-task minimum'],
            [[r['seed'],f"{r['final_text_nats']:.4f}",fmt(r['selected_five_task_min']),fmt(r['final_five_task_min'])] for r in runs]),'',
        diagnostic_note,'',
        *[line for c in coverage for line in (
            f"A post-test coverage diagnostic on run {c['seed']} replayed its weakest final validation condition ({c['kind'].replace('_',' ')}). The original {c['original_count']}-episode predictions reproduce exactly. Expanding that condition to {c['expanded_count']} episodes changes its minimum field score from **{fmt(min(c['original_accuracy']))} to {fmt(min(c['expanded_accuracy']))}**. The original validation and test-subset predictions remain identical across batch sizes. This diagnostic does not change checkpoint selection or the locked primary result.",'')],
        table(['Run seed','Text target','Memory target','K2 gain over K1','Ownership / resume'],
            [[r['seed'],'pass' if r['targets']['text'] else 'fail',
                'pass' if r['targets']['memory'] else 'fail',f"{100*r['recurrence_gain']:.2f} pp",
                'pass' if r['targets']['ownership'] and r['targets']['resume'] else 'fail'] for r in runs]),'',
        ('The integrated targets pass across all three parent/data pairs. The next interface step is learned natural-language fact ingestion and query addressing through the existing memory ports.'
         if passed else
         'The byte-stream interface is implemented and verified, but the locked joint-learning acceptance target is not met; M7 remains open. The next stage should broaden retention validation, extend text learning, and measure language/memory gradients by subsystem to explain the variation between runs. The final-checkpoint diagnostics expose useful progress that the current guard rejects; they remain separate from the selected primary results. Natural-language fact addressing follows reliable joint learning.'),'',
        '## What changed','',
        'Natural-text prediction and fresh memory episodes now update the same composer, expert banks, connections and byte decoder in one optimizer step. All four memory producers and their edge value projections continue learning; the inherited memory-edge context and gate weights remain fixed. The graph still has 510,958 parameters and 768 physical experts; no additional decoder or language model was added.','',
        'The byte-stream API carries graph state across chunks and saves the next-byte logits plus a private sampling RNG. Resuming generation consumes only new bytes. Text uses the A/B/C views; the four memory owners stay inactive and their outgoing edges are masked. The three source encodings are counted separately. Structured queries retain an explicit field address. Natural-language fact parsing and query addressing remain the next interface work.','',
        'Fixed 128-byte sample from the first run, prompted with ROMEO: and a newline, temperature 0.8. The literal below escapes line breaks and non-ASCII bytes; the artifact package also preserves the raw output.','',
        '```text',sample,'```','',
        '## Retained capabilities','',
        table(['Condition','Shared','Left private','Right private'],[[name.replace('_',' '),*map(fmt,field_means(name))] for name in PRIMARY]),'',
        'The primary minimum includes every field, correction direction and withheld update separately. Answers remain composed results g(f(x)); the retained value x is never decoded into an intermediate byte.','',
        table(['Available sources','Shared','Left private','Right private'],[[name.replace('_',' '),*map(fmt,field_means(name))] for name in ('left_only','right_only','erase_left','erase_right')]),'',
        'Unavailable private facts use a conditional 25% prior. Erasure controls preserve intact source execution counts; source absence skips the absent producers.','',
        '## Interventions','',
        table(['Memory condition','Shared','Left private','Right private'],[[name.replace('_',' '),*map(fmt,field_means(name))] for name in
            ('shared_right','k1','k4','no_exchange','no_edge','reset','shuffle','always','long_recall','long_correction','repeated')]),'',
        'The one-, two- and four-round comparisons use identical retained observations and weights. No-exchange, blocked-memory, reset-memory, shuffled-memory and unconditional-publication controls match the intact two-round execution count. Longer gaps are reported separately from the primary target.','',
        table(['Text condition','Mean nats/byte','Expert calls/byte','Seconds / 4,096 bytes'],[[name.replace('_',' '),
            f"{mean(r['evaluation']['text'][name]['nats'] for r in runs):.4f}",
            f"{runs[0]['evaluation']['text'][name]['expert_executions']/runs[0]['evaluation']['text'][name]['bytes']:.0f}",
            f"{mean(r['evaluation']['text'][name]['seconds'] for r in runs):.2f}"] for name in ('intact','k1','k4','no_exchange','reset_chunks')]),'',
        'Text state-reset controls reset at each 16-byte chunk. Each test starts with an independent state and cursor. Natural-text K=1/2/4 comparisons are measurements of this checkpoint, not separately trained models. Timings are means from the CPU reference implementation while the three study workers run concurrently.','',
        '## Text interleaved with memory','',
        table(['Run seed','Shared','Left private','Right private'],[[r['seed'],*[fmt(f['accuracy']) for f in r['evaluation']['interleaved']['fields']]] for r in runs]),'',
        'These episodes insert 32 natural-text bytes after fact acquisition and before querying, across 512 episodes per seed. Every retained owner tensor is preserved exactly. Each seed also exports a 128-byte autoregressive session with facts resident, followed by all three field queries. Split sampling at byte 41 reproduces the remaining 87 bytes, RNG, logits and complete state exactly.','',
        '## Learning, cost and verification','',
        'Each run performs 160 joint updates, consuming 20,480 natural-text bytes and 2,560 fresh memory episodes. Eight ordered language rows carry state across 16-byte backpropagation chunks. The memory objective cycles five tasks and gaps 0, 4, 8 and 16. Four alignment targets are fixed from each parent before training; no replay store is introduced.','',
        'AdamW uses betas (0.9, 0.95), weight decay 0.01, clipping at 1 and five warmup updates. Core, decoder and producer learning rates are 0.0003, 0.002 and 0.0001. Validation every 20 updates requires every memory task/field to remain at least 90%, then selects the lowest language loss.','',
        table(['Run seed','Expert executions','Training minutes','Peak RSS MiB'],[[r['seed'],f"{r['totals']['expert_executions']:,}",f"{r['totals']['training_seconds']/60:.2f}",f"{r['max_rss_mib']:.1f}"] for r in runs]),'',
        f"Training time sums completed updates and excludes validation and audits. Forward source encodings and repeated expert executions are counted. All **{verification['tests_passed']} regression tests pass**. Audit outputs reconstruct training bytes, synthetic batches, tables, source hashes, cursor positions, guarded selection, held-out predictions and the inference export.",'',
        '## Reproduction','',
        'Extract the source and artifact ZIPs into the same directory; both use the SHoEbox/ root. The artifact package includes the three parent checkpoints, initial/selected/final joint checkpoints, the separate development pilot with its exact source, predictions, samples, audits and the hash-verified Tiny Shakespeare data for offline reproduction.','',
        '```bash','python scripts/prepare_data.py','python scripts/check_all.py',
        'python scripts/audit_joint_text.py artifacts/joint_text/20261310',
        'python scripts/diagnose_joint_text_validation.py artifacts/joint_text/20261311 \\',
        '  --out artifacts/joint_text/diagnostics/validation_20261311.json',
        'python scripts/report_joint_text.py',
        'python -m experiments.text_regions.byte_stream \\',
        '  --checkpoint artifacts/joint_text/20261310/best.pt \\',
        '  --prompt "ROMEO:" --count 128 --out sample.bin','```','',
        'Repeat the audit for the other two run directories. A fresh three-pair study uses `python scripts/run_joint_text.py --root artifacts/joint_text_fresh`; repeating that command resumes complete optimizer and stream state. A generation session can continue with `--resume-session sample.session.pt` and the same checkpoint.','']
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text('\n'.join(lines))
    public=ROOT/'experiments/text_regions/results/joint_text';public.mkdir(parents=True,exist_ok=True)
    for name in ('summary.json','protocol_lock.json','environment.json','verification.json'):
        shutil.copyfile(a.root/name,public/name)
    diagnostic=a.root/'final_diagnostic_protocol.json'
    if diagnostic.exists():shutil.copyfile(diagnostic,public/diagnostic.name)
    else:(public/diagnostic.name).unlink(missing_ok=True)
    (public/'audit_summary.json').write_text(json.dumps({r['seed']:r['audit'] for r in runs},indent=2)+'\n')
    def memory_summary(value):
        return {**{k:v for k,v in value.items() if k not in ('fields','before')},
            'fields':[{k:v for k,v in field.items() if not isinstance(v,list)} for field in value['fields']]}
    compact={}
    for r in runs:
        e=r['evaluation']
        compact[r['seed']]=dict(memory={k:memory_summary(v) for k,v in e['memory'].items()},
            text={k:{key:value for key,value in v.items() if key!='rows'} for k,v in e['text'].items()},
            final=dict(step=e['final']['step'],memory={k:memory_summary(v) for k,v in e['final']['memory'].items()},
                text={k:v for k,v in e['final']['text'].items() if k!='rows'}),
            unigram_nats=e['unigram_nats'],publication=e['publication'],
            interleaved=dict(exact_memory=e['interleaved']['exact_memory'],
                fields=[dict(accuracy=f['accuracy']) for f in e['interleaved']['fields']]),
            live_text=e['live_text'])
    (public/'evaluation_summary.json').write_text(json.dumps(compact,indent=2)+'\n')
    (public/'README.md').write_text('# Joint natural text and retained memory\n\n'
        'See the [report](../../../../docs/SHoEbox_Joint_Text_Report.md) and '
        '[protocol](../../../../docs/SHoEbox_Joint_Text_Protocol.md). '
        'These files preserve per-pair metrics, validation trajectories, exact sample bytes, '
        'locked settings and verification summaries. Full predictions, checkpoints, '
        'raw data and reconstruction records are in the downloadable artifact archive.\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
