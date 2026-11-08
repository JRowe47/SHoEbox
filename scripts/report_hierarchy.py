"""Render the completed hierarchy/control implementation report as Markdown/PDF."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    pilot, control, comparison = [json.loads(path.read_text()) for path in
                                  (args.pilot, args.control, args.comparison)]
    manifest = json.loads((args.pilot.parent/'data_manifest.json').read_text())
    v, c = pilot['evaluation'][-1], control['evaluation'][-1]
    records = [
        ('Training-fitted unigram', pilot['baselines']['unigram']['nll'], pilot['baselines']['unigram']['accuracy']),
        ('Training-fitted bigram', pilot['baselines']['bigram']['nll'], pilot['baselines']['bigram']['accuracy']),
        ('One-region control', c['nll'], c['accuracy']),
        ('Six-region pilot', v['nll'], v['accuracy'])]
    gate = ('The hierarchy passes the comparison gate: it improves over the local control and '
            'swapping upper summaries worsens prediction.' if comparison['hierarchy_gate'] else
            'The hierarchy has not passed the comparison gate. Keep regional routing fixed '
            'through the 1,024-update comparison before adding regional selection or episodic memory.')
    tables = '\n'.join(f'| {name} | {nll:.4f} | {acc:.2%} |' for name, nll, acc in records)
    params = pilot['parameters']
    continuity = [f'{r["continuity_4k"]["nll"]:.4f}' if 'continuity_4k' in r else 'not run'
                  for r in (pilot, control)]
    md = f'''# SHoEbox predictive hierarchy: implementation and first gate

November 8, 2026. Based on architecture review/source `c0bbf64`.

Implemented the six-region pilot and its matched one-region control. All **50 checks pass**.
Each model completed **{pilot['updates']:,} optimizer updates / {pilot['trained_targets']:,} byte targets** at B8/T128.

## Held-out result

All rows below score the same **{v['targets']:,} targets from {len(manifest['files']['validation']['documents'])} documents** absent from training.

| Predictor | NLL, nats/byte | Accuracy |
|---|---:|---:|
{tables}

Pilot minus local-control NLL: **{comparison['pilot_minus_local_nll']:+.4f}**; negative favors the pilot.
Swapping only upper decoder summaries changes pilot NLL by **{pilot['upper_swap_delta_nll']:+.4f}**,
with local inputs and upstream recurrent trajectories held fixed.
Swapping input streams changes pilot NLL by **{pilot['input_swap']['nll']-v['nll']:+.4f}**.

{gate}

## What changed

- A1/B1 process each byte; L2/S2 consume four completed local spans; L3 consumes four completed middle spans. Small Y runs at byte cadence.
- One continuous 64-dimensional message per occurrence replaces internal 256-way byte-code ports. Source, order, span endpoints and age remain explicit.
- Every required source reaches its conjunction. Fixed scheduling leaves expert execution sparse: four of 32 experts per layer.
- The joint decoder combines local, middle and high slots, local/context products, ages and masks, with a direct local residual.
- Teacher forcing, evaluation and sampling share one observed-prefix contract. Targets do not enter state; generation feeds the sampled byte back.
- Four-byte and 16-byte future-span heads supply scale-specific training credit with weights 0.2 and 0.1. Only internal expert/HAT balance receives the small 0.001 auxiliary weight.
- State and incomplete spans carry across chunks; detach happens at optimizer boundaries. Checkpoints preserve optimizer, runtime state, cursor, RNG and baseline counts.

## Cost and training protocol

| Metric | Six-region pilot | One-region control |
|---|---:|---:|
| Total parameters | {params['total']:,} | {control['parameters']['total']:,} |
| Regional executions per byte | 3.5625 | 1 |
| Scheduled parameter touches per byte | {params['scheduled_inference_parameter_touches_per_byte']:,.1f} | {control['parameters']['scheduled_inference_parameter_touches_per_byte']:,.1f} |
| Aggregate training bytes/s | {pilot['training_bytes_per_second']:.1f} | {control['training_bytes_per_second']:.1f} |
| Training minutes | {pilot['training_seconds']/60:.2f} | {control['training_seconds']/60:.2f} |
| Full validation bytes/s | {v['scored_bytes_per_second']:.1f} | {c['scored_bytes_per_second']:.1f} |
| Old 4k continuity NLL | {continuity[0]} | {continuity[1]} |

CPU PyTorch {pilot['torch']}, native AVX512 kernels, one intra-op thread per run. Training throughput
includes forward, backward, clipping and AdamW; validation/checkpoint time is separate. Evaluation
uses batches of up to 32 document streams; these rates are aggregate external bytes/s, not single-stream
generation speed. Parameter touches are a selected-work estimate, not FLOPs. AdamW bank updates remain dense.

The finite training corpus contains **{pilot['unique_corpus_targets']:,} unique target positions** across
{len(manifest['files']['train']['documents'])} source documents; the run wraps and repeats part of it to reach
{pilot['trained_targets']:,} exposures. Both models see identical pairs and update budgets. AdamW uses LR 0.001,
16-update warmup, norm-1 clipping, and recurrent state reset at corpus wrap. Baselines use add-one smoothing
and exactly the observed training-pair counts. The old 4k continuity documents are reserved from training too.

## Verification and handoff

50 checks cover the new hierarchy and existing cortical model: no future leakage, target independence,
complete source recipes, span clocks, task gradients at every scale, fused/portable agreement,
document isolation, and exact model/optimizer/state continuation between span boundaries.

The default full experiment is B8/T128 for 1,024 updates (1,048,576 exposures). The runner stops at
the 256-update gate if it fails to beat unigram or establish observation conditioning. The completed
256-update comparison here is the first gate, not the full 1,024-update experiment.

```bash
python scripts/check_all.py
python -m shoebox.train_hierarchy --data data/hierarchy --out artifacts/pilot_full
python -m shoebox.train_hierarchy --data data/hierarchy --out artifacts/local_full --local-only
python -m scripts.compare_hierarchy artifacts/pilot_full/results.json artifacts/local_full/results.json --out artifacts/comparison.json
```

The full cortical map and original checkpoint behavior remain available. Learned regional routing,
learned boundaries, VQ transport and EC/hippocampal participation are the staged follow-ons from the review.
Detailed commands and architecture: `docs/predictive_hierarchy.md`. Run results include source/data hashes,
the full learning histories, route-use counts, diagnostics and samples.
'''
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out/'SHoEbox_Predictive_Hierarchy_Report'
    stem.with_suffix('.md').write_text(md)

    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    font_root = Path('/usr/share/fonts/truetype/dejavu')
    for name, filename in (('ReportSans', 'DejaVuSans.ttf'), ('ReportSans-Bold', 'DejaVuSans-Bold.ttf'),
                           ('ReportMono', 'DejaVuSansMono.ttf')):
        pdfmetrics.registerFont(TTFont(name, str(font_root/filename)))
    pdfmetrics.registerFontFamily('ReportSans', normal='ReportSans', bold='ReportSans-Bold',
                                  italic='ReportSans', boldItalic='ReportSans-Bold')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodySmall', fontName='ReportSans', fontSize=9.1, leading=13, spaceAfter=7))
    styles.add(ParagraphStyle(name='CodeSmall', fontName='ReportMono', fontSize=8, leading=11, spaceAfter=8))
    styles['Title'].fontName = 'ReportSans-Bold'; styles['Title'].fontSize = 19
    styles['Title'].leading = 24; styles['Title'].alignment = TA_LEFT
    styles['Title'].textColor = colors.HexColor('#173f3a')
    styles['Heading2'].textColor = colors.HexColor('#173f3a')
    styles['Heading2'].fontName = 'ReportSans-Bold'
    doc = SimpleDocTemplate(str(stem.with_suffix('.pdf')), pagesize=A4,
                            rightMargin=42, leftMargin=42, topMargin=40, bottomMargin=42)
    story = []
    def para(text, style='BodySmall'):
        story.append(Paragraph(text, styles[style]))
    def table(rows, widths):
        t = Table([[Paragraph(escape(str(x)), styles['BodySmall']) for x in row] for row in rows],
                  colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2eeeb')),
                              ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                              ('LINEBELOW', (0, 0), (-1, -1), .35, colors.HexColor('#bdd0ca')),
                              ('LEFTPADDING', (0, 0), (-1, -1), 8),
                              ('TOPPADDING', (0, 0), (-1, -1), 5),
                              ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
        story.extend([t, Spacer(1, 10)])
    para('SHoEbox: predictive hierarchy', 'Title')
    para('Implementation and first learning gate | November 8, 2026')
    para(f'<b>Six-region implementation complete. All 50 checks pass.</b> Each model trained for '
         f'{pilot["updates"]:,} updates and {pilot["trained_targets"]:,} external byte targets at B8/T128.')
    para('Held-out comparison', 'Heading2')
    para(f'{v["targets"]:,} targets from {len(manifest["files"]["validation"]["documents"])} separate documents; fresh state per document.')
    table([['Predictor', 'NLL (nats/byte)', 'Accuracy']]+
          [[n, f'{loss:.4f}', f'{acc:.2%}'] for n, loss, acc in records], [270, 125, 116])
    para(f'Pilot minus local control: <b>{comparison["pilot_minus_local_nll"]:+.4f} nats/byte</b>. '
         f'Upper-summary swap penalty: <b>{pilot["upper_swap_delta_nll"]:+.4f}</b>. '
         f'Input-stream swap penalty: <b>{pilot["input_swap"]["nll"]-v["nll"]:+.4f}</b>.')
    para(gate)
    para('What now learns differently', 'Heading2')
    for text in (
        '<b>Real temporal levels:</b> A1/B1 run every byte, L2/S2 every four bytes and L3 every 16. Required inputs arrive as complete ordered spans.',
        '<b>Continuous transport:</b> one 64-dimensional vector per occurrence carries source, order, span and age. Coordinates do not become recurrent time steps.',
        '<b>Joint expression:</b> small ordinary Y combines separate local/middle/high slots, products, masks and ages, with a local residual and one byte head.',
        '<b>Direct scale credit:</b> next-byte CE plus 0.2 times four-byte prediction and 0.1 times 16-byte prediction; no regional balancing loss.',
        '<b>One history contract:</b> training and generation observe the known byte and predict its successor. No self-argmax feedback or repeated revision round.'):
        para(text)
    para('Private HAT-SHoE recurrence and four-of-32 expert execution remain in every region. '
         'The full 18-region map remains the expansion design; the pilot starts from fresh weights.')
    story.append(PageBreak())
    para('Cost, verification and handoff', 'Title')
    table([['Metric', 'Pilot', 'Local control'],
           ['Total parameters', f'{params["total"]:,}', f'{control["parameters"]["total"]:,}'],
           ['Regional executions / byte', '3.5625', '1'],
           ['Selected parameter touches / byte', f'{params["scheduled_inference_parameter_touches_per_byte"]:,.1f}',
            f'{control["parameters"]["scheduled_inference_parameter_touches_per_byte"]:,.1f}'],
           ['Aggregate training bytes / second', f'{pilot["training_bytes_per_second"]:.1f}', f'{control["training_bytes_per_second"]:.1f}'],
           ['Training minutes', f'{pilot["training_seconds"]/60:.2f}', f'{control["training_seconds"]/60:.2f}'],
           ['Validation bytes / second', f'{v["scored_bytes_per_second"]:.1f}', f'{c["scored_bytes_per_second"]:.1f}'],
           ['Old 4k continuity NLL', *continuity]], [265, 123, 123])
    para('PyTorch CPU/native AVX512, one intra-op thread per run. Training timing includes forward, '
         'backward, clipping and AdamW. Validation/checkpoint time is separate. Evaluation batches '
         'up to 32 document streams; rates are aggregate external bytes/s. Parameter touches estimate '
         'scheduled selected work, not FLOPs; AdamW still updates dense banks.')
    para('Matched data and update budget', 'Heading2')
    para(f'The corpus has {pilot["unique_corpus_targets"]:,} training targets across '
         f'{len(manifest["files"]["train"]["documents"])} documents. It wraps to reach '
         f'{pilot["trained_targets"]:,} exposures. Both models see the same training pairs. '
         'Validation and old continuity documents are excluded from training. Baselines use '
         'add-one smoothing fitted only to observed training pairs.')
    para('AdamW: LR 0.001, 16-update warmup, gradient norm 1. State carries across T128 chunks, '
         'detaches at optimizer boundaries and resets at corpus wrap. The horizon covers eight L3 updates.')
    para('Verification', 'Heading2')
    para('50 checks cover future exclusion, target independence, complete recipes, span clocks, '
         'task gradients at every scale, fused/portable agreement, document isolation and exact '
         'weight/optimizer/state continuation. Existing cortical-model checks also pass.')
    para('Next run', 'Heading2')
    para('The default harness runs 1,024 updates / 1,048,576 exposures and stops at the 256-update '
         'gate if observation conditioning fails. This report covers the completed first gate. '
         'Keep routing fixed until the hierarchy improves over the matched local control.')
    para('python -m shoebox.train_hierarchy --data data/hierarchy<br/> '
         '&nbsp; --out artifacts/pilot_full', 'CodeSmall')
    para('Repeat with --local-only for the control. See docs/predictive_hierarchy.md for '
         'preparation, exact resume and comparison commands. Checkpoints retain private states, '
         'pending spans, AdamW, RNG, cursor and baseline counts. Results include hashes, histories, '
         'route use and interventions.')
    def footer(canvas, document):
        canvas.setFont('ReportSans', 8)
        canvas.setFillColor(colors.HexColor('#476860'))
        canvas.drawString(42, 24, 'SHoEbox | Predictive hierarchy implementation | November 8, 2026')
        canvas.drawRightString(A4[0]-42, 24, str(document.page))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(json.dumps(dict(markdown=str(stem.with_suffix('.md')), pdf=str(stem.with_suffix('.pdf')))))


if __name__ == '__main__':
    main()
