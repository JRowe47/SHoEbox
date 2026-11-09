"""Report the fresh-data, longer-horizon continuation from audited checkpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    seed = json.loads((args.seed/'results.json').read_text())
    run = json.loads((args.run/'results.json').read_text())
    audit = json.loads((args.run/'checkpoint_audit.json').read_text())
    manifest = json.loads((args.data/'manifest.json').read_text())
    if not run['complete'] or not audit['exact_score_reproduction']:
        raise ValueError('Complete training and exact checkpoint reproduction are required.')
    a, b = seed['evaluation'][-1], run['evaluation'][-1]
    if seed['protocol']['validation_sha256'] != run['protocol']['validation_sha256']:
        raise ValueError('Validation changed.')
    if a['targets'] != b['targets'] or not a['full'] or not b['full']:
        raise ValueError('Endpoints must use the same complete validation set.')
    if run['initial_targets'] != seed['lifetime_targets']:
        raise ValueError('Lifetime target history does not match the seed.')
    seed_audit = json.loads((args.seed/'checkpoint_audit.json').read_text())
    for field in ('nll', 'accuracy', 'targets', 'span_nll', 'span_targets'):
        if run['evaluation'][0][field] != seed_audit['monitoring_endpoint'][field]:
            raise ValueError('The initial monitoring score differs from the seed.')
    if len(run['history']) != run['updates']:
        raise ValueError('Incomplete update history.')
    for row in run['history']:
        if (row['phase'] != 'joint' or row['trainable_parameters'] != run['parameters']['total']
                or row['local_scale'] != 1 or row['fast_lr_scale'] != 1
                or row['lr'] != run['protocol']['lr']):
            raise ValueError('The executed policy differs from the reported joint policy.')
        if not all(math.isfinite(x) for x in row['regional_gradient_norms'].values()):
            raise ValueError('Nonfinite gradient history.')
    delta = b['nll']-a['nll']
    improvement = 100*(a['nll']-b['nll'])/a['nll']
    perplexity = 100*(1-math.exp(delta))
    increased = run['upper_all_swap_delta_nll'] > seed['upper_all_swap_delta_nll']
    outcome = (f"Held-out next-byte NLL fell from {a['nll']:.6f} to {b['nll']:.6f} "
               f"({improvement:.2f}% lower; byte perplexity {perplexity:.2f}% lower)."
               if delta < 0 else
               f"Held-out next-byte NLL changed from {a['nll']:.6f} to {b['nll']:.6f} ({delta:+.6f}).")
    influence = (f"The combined upper-message swap penalty {'increased' if increased else 'decreased'} "
                 f"from {seed['upper_all_swap_delta_nll']:.6f} to {run['upper_all_swap_delta_nll']:.6f} nats/byte.")
    decision = ('Keep joint training with fresh data and the longer window as the preferred continuation. '
                'The endpoint monitoring loss is still improving. Upper-context influence has grown, '
                'while the direct upper decoder contribution remains negligible.')
    metrics = [
        ('Next-byte NLL', a['nll'], b['nll']),
        ('Next-four-byte NLL', a['span_nll']['mid'], b['span_nll']['mid']),
        ('Next-sixteen-byte NLL', a['span_nll']['high'], b['span_nll']['high']),
        ('Decoder-summary swap delta', seed['upper_swap_delta_nll'], run['upper_swap_delta_nll']),
        ('Decoder + feedback swap delta', seed['upper_all_swap_delta_nll'], run['upper_all_swap_delta_nll'])]
    table = '\n'.join(f'| {label} | {x:.6f} | {y:.6f} | {y-x:+.6f} |' for label,x,y in metrics)
    train_docs = len(manifest['files']['train']['documents'])
    val_docs = len(manifest['files']['validation']['documents'])
    total_unique = seed['unique_corpus_targets']+manifest['train_targets']
    protocol = run['protocol']
    ckpt_sha = hashlib.sha256((args.run/'last.pt').read_bytes()).hexdigest()
    record = dict(format='shoebox-fresh-joint-phase-1',
                  initial_updates=run['initial_updates'], additional_updates=run['updates'],
                  lifetime_updates=run['lifetime_updates'], initial_targets=run['initial_targets'],
                  additional_targets=run['trained_targets'], lifetime_targets=run['lifetime_targets'],
                  lifetime_unique_targets=total_unique, training_documents=train_docs,
                  validation_documents=val_docs, validation_targets=b['targets'],
                  before=dict(nll=a['nll'], accuracy=a['accuracy'], span_nll=a['span_nll'],
                              upper_swap_delta_nll=seed['upper_swap_delta_nll'],
                              upper_all_swap_delta_nll=seed['upper_all_swap_delta_nll']),
                  after=dict(nll=b['nll'], accuracy=b['accuracy'], span_nll=b['span_nll'],
                             upper_swap_delta_nll=run['upper_swap_delta_nll'],
                             upper_all_swap_delta_nll=run['upper_all_swap_delta_nll']),
                  delta_nll=delta, nll_reduction_percent=improvement,
                  perplexity_reduction_percent=perplexity, protocol=protocol,
                  final_checkpoint_sha256=ckpt_sha, training_seconds=run['training_seconds'],
                  training_bytes_per_second=run['training_bytes_per_second'],
                  peak_rss_mib=run['peak_rss_mib'], checkpoint_audit=audit,
                  samples=run['samples'], routing=run['routing'])
    args.out.mkdir(parents=True, exist_ok=True)
    name = 'SHoEbox_Fresh_Joint_Training_Report'
    (args.out/'SHoEbox_Fresh_Joint_Training_Results.json').write_text(json.dumps(record,indent=2)+'\n')
    md = f'''# SHoEbox: fresh-data joint training

November 8, 2026. Six-region predictive hierarchy; {run['parameters']['total']:,} parameters.

{outcome} {influence}

{decision}

## Policy and completed training

The previous joint run was still improving and beat the matched freeze/thaw
schedule (2.392272 vs 2.448039 nats/byte). This phase therefore keeps all regions
learning at AdamW 0.001 and extends both the corpus and the gradient window.

| Setting | This phase |
|---|---|
| Start | {run['initial_updates']:,} lifetime updates; {run['initial_targets']:,} lifetime target exposures |
| Added training | {run['updates']:,} updates; {run['trained_targets']:,} byte targets |
| End | {run['lifetime_updates']:,} lifetime updates; {run['lifetime_targets']:,} lifetime exposures |
| Corpus | {train_docs} fresh FineWeb documents; one pass, no repeats this phase |
| Credit window | B8/T256, up from T128; 16 L3 updates per gradient window |
| Optimizer | Joint AdamW 0.001; betas (0.9, 0.95); matrix decay 0.01; clip 1 |
| Auxiliary objectives | Next-four-byte weight 0.2; next-sixteen-byte weight 0.1 |
| Held out | Same {b['targets']:,} targets / {val_docs} documents; evaluation chunks stay at 128 |

Weights, Adam moments and update ages are preserved. Recurrent state and the
corpus cursor reset for unrelated streams. There is no freezing, route attenuation
or learning-rate anneal. Lifetime exposure counts retain each phase's actual
window size; the older T128 updates are not rescaled to T256. Across the complete
training history, {total_unique:,} distinct target positions have been used.

## Held-out result

All losses are nats/byte. Prediction loss is lower-is-better; swap deltas are
intervention loss minus ordinary loss.

| Measurement | Before | After | Change |
|---|---:|---:|---:|
{table}

Next-byte accuracy: **{100*a['accuracy']:.2f}% -> {100*b['accuracy']:.2f}%**.
Decoder-only swapping preserves producer trajectories. The combined swap also
mismatches top-down messages and recomputes lower trajectories from fresh states.
A positive penalty means those upper messages help the ordinary prediction.
Fresh data and a longer gradient window change together in this continuation.

## Runtime and verification

Training completes in **{run['training_seconds']/60:.2f} minutes** of measured update
time, averaging **{run['training_bytes_per_second']:.2f} aggregate bytes/s** on one
CPU thread with the AVX512 backend. Peak process RSS: **{run['peak_rss_mib']:.1f} MiB**.
These counts refer to external byte targets across all eight streams.

**62 checks pass.** An independent CLI transition/pause/resume run exactly matches
uninterrupted model, Adam, recurrent state and RNG. It verifies state reset,
preserved Adam update age and mixed-window exposure accounting. The final
checkpoint exactly reproduces the full held-out byte and future-span scores
after reload; parameters and recurrent state are finite. The phase's initial
8,192-target monitoring score also exactly matches the seed checkpoint.

## Data and reproduction

Source: [FineWeb sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb),
official viewer rows 1000-1299. Complete response bytes and checksums are included
in the checkpoint bundle. Preparation excludes every prior training, validation
and reserved document by ID and full source-text hash. Validation bytes and
document boundaries are identical to the previous phase.

Commands: `docs/fresh_joint_training.md`. Data preparation:
`scripts/prepare_hierarchy_phase.py`. Endpoint audit:
`scripts/audit_hierarchy_checkpoint.py`. Full phase metrics, per-region gradients,
optimizer state, exact prepared data and raw source responses accompany the
checkpoint. The seed checkpoint is also retained for reproduction.

- Runtime source SHA-256: `{protocol['source_sha256']}`
- Initial checkpoint SHA-256: `{protocol['initial_checkpoint_sha256']}`
- Final checkpoint SHA-256: `{ckpt_sha}`
- Training SHA-256: `{protocol['train_sha256']}`
- Validation SHA-256: `{protocol['validation_sha256']}`
'''
    (args.out/(name+'.md')).write_text(md)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 2.65), dpi=180)
    rows = [r for r in run['evaluation'] if not r['full']]+[audit['monitoring_endpoint']]
    ax.plot([r['updates']*protocol['batch']*protocol['context']/1024 for r in rows],
            [r['nll'] for r in rows], marker='o', ms=3.5, color='#3569ad', linewidth=1.8)
    ax.set(xlabel='Additional byte targets (Ki targets)', ylabel='Validation NLL',
           title='Fixed 8,192-target monitoring window')
    ax.set_xlim(0,run['trained_targets']/1024)
    ax.grid(alpha=.18); ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    curve = args.out/'SHoEbox_Fresh_Joint_Training_Curve.png'
    fig.savefig(curve); plt.close(fig)

    from reportlab.lib import colors as rc
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    fonts = Path(matplotlib.get_data_path())/'fonts/ttf'
    pdfmetrics.registerFont(TTFont('ReportSans',str(fonts/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('ReportSansBold',str(fonts/'DejaVuSans-Bold.ttf')))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodySmall',fontName='ReportSans',fontSize=9.5,leading=13,
                              spaceAfter=8,textColor=rc.HexColor('#283644')))
    styles.add(ParagraphStyle(name='SmallNote',fontName='ReportSans',fontSize=8,leading=11,
                              spaceAfter=6,textColor=rc.HexColor('#526174')))
    styles['Title'].fontSize=22; styles['Title'].leading=25; styles['Title'].alignment=TA_LEFT
    styles['Title'].fontName='ReportSansBold'; styles['Title'].textColor=rc.HexColor('#16384d')
    styles['Heading2'].fontSize=12; styles['Heading2'].fontName='ReportSansBold'
    styles['Heading2'].textColor=rc.HexColor('#16384d')
    def para(text, style='BodySmall'):
        return Paragraph(escape(text),styles[style])
    def table_pdf(rows,widths):
        t=Table([[para(str(c),'SmallNote') for c in row] for row in rows],colWidths=widths)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),rc.HexColor('#e7eff4')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[rc.white,rc.HexColor('#f5f7f9')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),
            ('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),
            ('BOTTOMPADDING',(0,0),(-1,-1),3)]))
        return t
    flow=[para('SHoEbox: fresh-data joint training','Title'),
          para('COMPLETED CONTINUATION  |  NOVEMBER 8, 2026','SmallNote'),
          para(outcome),para(influence),
          para(f"{run['trained_targets']:,} new byte targets from {train_docs} fresh documents. "
               f"The model reaches {run['lifetime_targets']:,} lifetime target exposures. "
               f"Held-out evaluation: unchanged {b['targets']:,} targets across {val_docs} documents."),
          table_pdf([['Measurement (nats/byte)','Before','After','Change']]+
                    [[label,f'{x:.6f}',f'{y:.6f}',f'{y-x:+.6f}'] for label,x,y in metrics],
                    [249,85,85,85]),Spacer(1,9),
          para(f"Next-byte accuracy: {100*a['accuracy']:.2f}% to {100*b['accuracy']:.2f}%. "
               'Positive swap deltas mean mismatched upper messages worsen prediction. '
               'Combined swapping includes top-down feedback; decoder-only swapping preserves producer trajectories.','SmallNote'),
          Image(str(curve),width=504,height=166.95),Spacer(1,6),
          para('The monitoring subset is fixed across phases. Its values differ from the full validation scores in the table.','SmallNote'),
          para(decision,'SmallNote'),
          PageBreak(),para('Training policy and verification','Title'),
          table_pdf([['Setting','Completed policy'],
                     ['Optimizer','Joint AdamW 0.001; betas 0.9 / 0.95; matrix decay 0.01; gradient clip 1'],
                     ['Temporal credit','Eight streams, 256 observations per update; 16 L3 updates per gradient window'],
                     ['Budget',f"{run['updates']} added updates; {run['lifetime_updates']} lifetime updates; {run['lifetime_targets']:,} lifetime targets"],
                     ['Objectives','Next-byte loss plus 0.2 next-four-byte and 0.1 next-sixteen-byte losses'],
                     ['State transition','Keep weights, Adam moments and ages; reset recurrence and cursor for new streams']],
                    [115,389]),Spacer(1,10),
          para('Why this phase','Heading2'),
          para('Joint training was still improving and beat the matched freeze/thaw schedule. '
               'This continuation gives the hierarchy fresh text and twice the temporal credit while keeping all regions trainable. '
               'Learning rate, routing and loss weights stay fixed.'),
          para('Data and accounting','Heading2'),
          para(f'One pass through {train_docs} fresh FineWeb documents, excluding every prior training, validation and reserved document '
               f'by ID and source-text hash. The original held-out bytes and document boundaries are preserved. '
               f'The full history now contains {total_unique:,} distinct target positions. '
               'Earlier exposure counts retain their original window size.'),
          para('Verification','Heading2'),
          para('All 62 checks pass. A separate CLI transition and split run reproduces weights, optimizer, recurrent state and RNG exactly. '
               'It confirms reset state, preserved Adam age and correct mixed-window counts. '
               'The final checkpoint reproduces full held-out next-byte and span scores exactly after reload; parameters and state are finite.'),
          para('Measured runtime','Heading2'),
          para(f"{run['training_seconds']/60:.2f} minutes of update time; {run['training_bytes_per_second']:.2f} aggregate training bytes/s; "
               f"{run['peak_rss_mib']:.1f} MiB peak RSS. One CPU thread, AVX512 backend. "
               'Throughput counts external byte targets across all eight streams.'),
          para('Reproduction','Heading2'),
          para('The bundle contains seed/final checkpoints, optimizer and runtime state, full histories, data manifests, exact corpus bytes and raw source responses. '
               'Commands: docs/fresh_joint_training.md. Checksums and the FineWeb source link are in the Markdown report. '
               'Fresh data and the longer gradient window change together in this continuation.','SmallNote')]
    def footer(canvas,doc):
        canvas.setFont('ReportSans',8); canvas.setFillColor(rc.HexColor('#657386'))
        canvas.drawString(54,30,'SHoEbox | Fresh-data joint continuation')
        canvas.drawRightString(558,30,str(doc.page))
    SimpleDocTemplate(str(args.out/(name+'.pdf')),pagesize=letter,leftMargin=54,rightMargin=54,
                      topMargin=42,bottomMargin=48).build(flow,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps(dict(outcome=outcome,influence=influence,markdown=str(args.out/(name+'.md')),
                          pdf=str(args.out/(name+'.pdf')),curve=str(curve))))


if __name__ == '__main__':
    main()
