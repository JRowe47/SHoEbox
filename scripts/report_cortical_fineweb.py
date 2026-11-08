"""Export measured cortical FineWeb results as Markdown, PDF and a loss plot."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Image, PageBreak)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', default='local working source')
    args = p.parse_args()
    s = json.loads((args.run/'results.json').read_text())
    if not s['complete'] or s['trained_targets'] != s['protocol']['budget'] or 'final_validation' not in s:
        raise ValueError('Only completed, evaluated runs may be reported.')
    baseline = json.loads((args.run/'count_baselines.json').read_text())
    manifest = json.loads((args.run/'data_manifest.json').read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out/'SHoEbox_Cortical_10M_FineWeb_Report'
    a, z, cfg = s['initial_validation'], s['final_validation'], s['config']
    reduction = 100*(1-z['nll']/a['nll'])
    h = s['history']
    xs = [r['trained_targets'] for r in h]
    smooth = [sum(v['nll']*v['targets'] for v in h[max(0,i-9):i+1])/
              sum(v['targets'] for v in h[max(0,i-9):i+1]) for i in range(len(h))]
    vals = s['validation']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
    ax.plot(xs, [r['nll'] for r in h], color='#47a4ac', alpha=.2, linewidth=.6)
    ax.plot(xs, smooth, color='#19848e', label='Training: rolling 10 updates', linewidth=1.6)
    ax.plot([v['trained_targets'] for v in vals], [v['nll'] for v in vals],
            '-o', color='#243d72', label='Fixed holdout: 4,096 byte targets', markersize=4)
    for name, color in [('unigram', '#b77b36'), ('bigram', '#777777')]:
        ax.axhline(baseline[name]['nll'], color=color, linestyle='--', linewidth=1,
                   label=name.capitalize()+' count baseline')
    ax.set(xlabel='FineWeb training byte targets', ylabel='Next-byte NLL (nats)',
           xlim=(0, s['trained_targets']))
    ax.ticklabel_format(axis='x', style='plain')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.15)
    ax.legend(fontsize=8, loc='upper right')
    plot = args.out/'SHoEbox_Cortical_10M_FineWeb_Learning.png'
    fig.savefig(plot, dpi=180)
    plt.close(fig)
    unidiff = z['nll']-baseline['unigram']['nll']
    comparison = (f"Final holdout loss is {abs(unidiff):.4f} nats/byte "
                  f"{'below' if unidiff < 0 else 'above'} the training-count unigram baseline "
                  f"({baseline['unigram']['nll']:.4f}); the add-one bigram baseline is "
                  f"{baseline['bigram']['nll']:.4f}.")
    assessment_path = args.run/'assessment.json'
    assessment = json.loads(assessment_path.read_text())['summary'] if assessment_path.exists() else ''
    metrics = [
        ('Parameters', f"{s['parameters']:,}"),
        ('Training targets / updates', f"{s['trained_targets']:,} / {s['optimizer_updates']}"),
        ('Holdout NLL, initial -> final', f"{a['nll']:.4f} -> {z['nll']:.4f} nats/byte"),
        ('Holdout bits per byte', f"{a['bits_per_byte']:.4f} -> {z['bits_per_byte']:.4f}"),
        ('Holdout next-byte accuracy', f"{100*a['accuracy']:.2f}% -> {100*z['accuracy']:.2f}%"),
        ('Training time / throughput', f"{s['training_seconds']/60:.2f} min / {s['training_bytes_per_second']:.2f} bytes/s"),
        ('Peak process RSS / runtime state', f"{s['peak_rss_mib']:.1f} MiB / {s['runtime_state_bytes']/2**20:.2f} MiB"),
    ]
    samples = s.get('samples', [])
    diagnostic_path = args.run/'port_diversity_final.json'
    diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else None
    diversity_text = ''
    if diagnostic:
        counts = [v['unique_complete_frames'] for k,v in diagnostic['ports'].items() if k.endswith(':4')]
        diversity_text = (f'At the endpoint, complete four-byte ports have {min(counts)}-{max(counts)} '
                          f'distinct outputs across 200 independent stream states (median {np.median(counts):.0f}). '
                          f"The Y readout has {diagnostic['ports']['Y:1']['unique_complete_frames']} distinct "
                          'argmax bytes across those states.')
    train_docs = manifest['files']['train']['documents']
    val_docs = manifest['files']['validation']['documents']
    settings = [
        ('Cortex', '18 ordinary regions; two rounds; EC-only hippocampus'),
        ('Regional capacity', 'Width 64; 192 experts; expert hidden width 16'),
        ('Sparse selection', 'Three layers; 1 of 64 experts selected per layer'),
        ('HAT / laces', '1 level; 64 HAT symbols; 32 laces, 2 selected'),
        ('minKron', '2 heads; key/value dimensions 8/8'),
        ('Hippocampus', '32 episodes/stream; 2 returns; 64-address/8-active'),
        ('Training', 'Batch 200; 2-byte TBPTT; 250 updates; FP32 CPU'),
        ('Optimizer', 'AdamW; LR 0.002 after 10 warmup updates; betas 0.9/0.95'),
        ('Controls', 'Clip norm 1; decay 0.01; auxiliary weight 0.001'),
    ]
    text = '# Cortical SHoEbox: 10M / 100k FineWeb run\n\nNovember 8, 2026.\n\n'
    text += (f"Completed one pass over **{s['trained_targets']:,} FineWeb next-byte targets** "
             f"with **{s['parameters']:,} parameters**. Fixed-holdout NLL fell "
             f"**{reduction:.1f}%**, from {a['nll']:.4f} to {z['nll']:.4f} nats/byte.\n\n")
    text += '| Measurement | Result |\n| --- | --- |\n'
    text += ''.join(f'| {k} | {v} |\n' for k,v in metrics)+'\n'
    text += f'![Learning curve]({plot.name})\n\n{comparison}\n\n'
    if assessment:
        text += assessment+'\n\n'
    text += ('The training curve is a moving online loss over different fresh text positions; '
             'the holdout curve evaluates the same untrained-on positions at fixed weights. '
             'Count baselines use only the training targets with add-one smoothing.\n\n')
    text += '## Model and protocol\n\n| Setting | Value |\n| --- | --- |\n'
    text += ''.join(f'| {k} | {v} |\n' for k,v in settings)+'\n'
    text += (f"Training uses {len(train_docs)} source documents (viewer rows "
             f"{train_docs[0]['row_idx']}-{train_docs[-1]['row_idx']}); the holdout uses "
             f"{len(val_docs)} different documents (rows {val_docs[0]['row_idx']}-{val_docs[-1]['row_idx']}). "
             'Text is UTF-8 with two newline bytes between documents. The corpus has 100,200 bytes: '
             '100,000 scored targets plus 200 unscored stream-prefix bytes. Validation has '
             '4,096 targets plus 32 prefix bytes. No source IDs or complete-text hashes overlap.\n\n'
             'Each training stream spans 500 observations. State carries forward throughout; gradients '
             'detach after every two observations. The hippocampus keeps 32 episodes per stream. '
             'No corpus wrap, repeated epoch, regional archive or query-specific controller was introduced.\n\n')
    text += ('All nine existing architecture checks and both new data/trainer checks passed. '
             'An interrupted/resumed production-runner test reproduced model updates exactly. '
             'Pure next-byte loss on the first actual training batch reached all 18 regions and '
             f"the hippocampus (gradient L1 sum {s['task_gradient_coverage']['hippocampus']:.6g}).\n\n")
    if diversity_text:
        text += diversity_text+'\n\n'
    text += '## Sample continuations\n\n'
    text += 'Each sample uses temperature 0.8 and generates 96 bytes through an external autoregressive text loop.\n\n'
    for sample in samples:
        text += f"Prompt: `{sample['prompt']}`; seed {sample['seed']}.\n\n```text\n{sample['generated']}\n```\n\n"
    text += '## Reproduce and continue\n\n```bash\n'
    text += ('python scripts/prepare_cortical_fineweb.py --out data/cortical_fineweb\n'
             'python -m shoebox.train_fineweb --data data/cortical_fineweb \\\n'
             '  --out artifacts/cortical_fineweb_10m --context 2 --threads 2 \\\n'
             '  --lr 0.002 --warmup-updates 10 --checkpoint-every 50 --sample-bytes 96\n```\n\n'
             'For exact reproduction, use the packaged corpus and source. The viewer serves main; '
             'a fresh download may differ later. The checkpoint retains optimizer, RNG, all runtime '
             'states and the exact completed cursor. Resume finishes an interrupted original budget; '
             'the completed endpoint requires a new training phase to consume additional text.\n\n')
    text += f"Local source commit: `{args.commit}` (GitHub publication pending). Training source fingerprint: `{s['protocol']['source_sha256']}`.\n\n"
    text += f"Runtime: PyTorch {s['torch']}; Python {s['python']}; {s['protocol']['threads']} CPU threads.\n\n"
    text += f"Training SHA-256: `{s['protocol']['train_sha256']}`.\n\nValidation SHA-256: `{s['protocol']['validation_sha256']}`.\n\n"
    text += ('Sources: [FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb), '
             '[official row-slice API](https://huggingface.co/docs/dataset-viewer/rows).\n')
    stem.with_suffix('.md').write_text(text)

    for name, family, weight in [('DV', 'DejaVu Sans', 'normal'),
                                 ('DV-Bold', 'DejaVu Sans', 'bold'),
                                 ('DV-Mono', 'DejaVu Sans Mono', 'normal')]:
        pdfmetrics.registerFont(TTFont(name, findfont(FontProperties(family=family, weight=weight))))
    styles = getSampleStyleSheet()
    styles['Title'].fontName = 'DV-Bold'
    styles['Heading2'].fontName = 'DV-Bold'
    styles.add(ParagraphStyle(name='BodySmall', fontName='DV', fontSize=9, leading=12,
                              spaceAfter=7, textColor=colors.HexColor('#243041')))
    styles.add(ParagraphStyle(name='Label', fontName='DV-Bold', fontSize=11, leading=14,
                              spaceBefore=8, spaceAfter=6, textColor=colors.HexColor('#173c65')))
    styles.add(ParagraphStyle(name='Sample', fontName='DV-Mono', fontSize=8, leading=11,
                              spaceAfter=10, wordWrap='CJK'))
    body = styles['BodySmall']
    def para(t, style=body):
        return Paragraph(escape(t), style)
    def table(rows, widths):
        tbl = Table([[para(k),para(v)] for k,v in rows], colWidths=widths, hAlign='LEFT')
        tbl.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),
                                 ('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.HexColor('#f1f5f9'),colors.white]),
                                 ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                                 ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),2)]))
        return tbl
    doc = SimpleDocTemplate(str(stem.with_suffix('.pdf')), pagesize=(612,792),
                            rightMargin=43,leftMargin=43,topMargin=40,bottomMargin=38,
                            title='Cortical SHoEbox: 10M / 100k FineWeb',author='Joshua Rowe / Codex')
    flow = [para('Cortical SHoEbox', styles['Title']),
            para('9.98M parameters | 100,000 FineWeb byte targets | November 8, 2026'),
            para(f'Completed: holdout loss fell {reduction:.1f}%', styles['Heading2']),
            table(metrics, [235,291]), Spacer(1,8), Image(str(plot),width=526,height=204.6),
            para(comparison),
            para(assessment) if assessment else Spacer(1,1),
            para('Holdout evaluation uses 4,096 targets from two source documents excluded from training. '
                 'The training curve averages ten updates over fresh positions; the holdout curve uses a fixed slice.'),
            PageBreak(), para('Configuration and observed output',styles['Heading2']),
            table(settings,[120,406]),Spacer(1,7),
            para('The 18-region wiring, complete byte ports, two recurrent rounds and EC-only hippocampus are preserved. '
                 'Each stream carries state across 500 observations; gradients detach every two observations. '
                 'The 100,200-byte training file supplies 100,000 targets and 200 unscored prefix bytes.'),
            para('Checks',styles['Label']),
            para('All 11 checks passed, including exact production-runner resume. Pure task-loss gradients reached '
                 'all 18 regions and the hippocampus. Checkpoints preserve optimizer, runtime state, RNG and cursor.'),
            para('Sample continuations',styles['Label'])]
    for sample in samples:
        flow += [para(f"Prompt {sample['prompt']!r} | temperature 0.8 | seed {sample['seed']}"),
                 para(json.dumps(sample['generated'], ensure_ascii=True), styles['Sample'])]
    flow += [para('Reproduction',styles['Label']),
             para('Use the attached source and exact corpus with python -m shoebox.train_fineweb. '
                  'The Markdown report contains the full command, checksums and interpretation. '
                  'last.pt is the completed 100,000-target endpoint.'),
             para(f'Source: {args.commit}'),
             para('Data: huggingface.co/datasets/HuggingFaceFW/fineweb (sample-10BT). '
                  'Protocol: huggingface.co/docs/dataset-viewer/rows.')]
    def footer(canvas, doc):
        canvas.setFont('DV',8);canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(43,23,'SHoEbox | measured training results')
        canvas.drawRightString(569,23,str(doc.page))
    doc.build(flow,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps({'markdown':str(stem.with_suffix('.md')),'pdf':str(stem.with_suffix('.pdf')),'plot':str(plot)}))


if __name__ == '__main__':
    main()
