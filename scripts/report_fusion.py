"""Render the measured runtime update as Markdown and a compact PDF."""
import json
from pathlib import Path
import statistics
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = json.loads((ROOT/'experiments/text_regions/results/fusion_runtime/measurements.json').read_text())
    validation = json.loads((ROOT/'experiments/text_regions/results/fusion_runtime/validation.json').read_text())
    names = {'default':'2.56M', '25m':'24.91M'}
    workload_names = {'train':'Forward + backward', 'inference':'Inference segment', 'generation':'Generation'}
    rows = []
    for result in results['measurements']:
        for name, row in result['workloads'].items():
            rows.append([names[result['size']], workload_names[name],
                         f"{statistics.median(row['seconds']['previous']):.3f}",
                         f"{statistics.median(row['seconds']['current']):.3f}", f"{row['speedup']:.2f}x"])
    scaled = next(r for r in results['measurements'] if r['size']=='25m')
    allocation = scaled['backward_allocations']
    previous, current = (allocation[mode]['expert_dense_bank_buffers'] for mode in ('previous','current'))
    saving = 100*(1-current['allocated_bytes']/previous['allocated_bytes'])
    changes = [
        ('Expert gradients', 'Large banks combine selected-expert backward operations, accumulate compact row derivatives and allocate one dense gradient per bank per byte event. A native row accumulator avoids sparse-to-dense overhead. Small banks keep the cheaper existing path. AdamW still receives dense gradients.'),
        ('Recurrent backward', 'Large-bank models retain MinKron forward arithmetic and reuse saved decay, norm, score and update terms in a fused CPU backward. A differentiable PyTorch path handles higher derivatives. Small models retain the existing cheaper path; systems without the extension have a scripted fallback.'),
        ('Routing projections', 'Each forward computes the prototype/message matrix product once and reuses it across bytes and rounds. The trainable projection remains in the gradient graph.'),
        ('Generation caching', 'Both generation interfaces retain normalized routing tables, symbol keys, lace templates and decay constants across bytes. Parameter and buffer versions, replacement, dtype/device changes, checkpoint loads and autocast mode invalidate the cache.'),
        ('Independent stages', 'Large-bank training batches compatible full-activity contextualizers and pilot projections. Nodes keep separate weights and owned state storage. Inference, partial masks and registered hooks use the individual path; stage snapshots and commit order are preserved. The batching flag can also be enabled explicitly on smaller models.'),
        ('Repeated calculations', 'Gradient norms and task dot products are reduced once per parameter and aggregated for both reporting views. The observational lace pass reuses probabilities already produced by routing.'),
    ]
    protocol = results['protocol']
    paragraphs = [
        'The main measured gain is lower expert-gradient allocation; throughput changes are mixed. Comparison is against the immediately preceding optimized release, bd0842e, with identical initialized weights and inputs. Model capacity and feedback rounds are preserved.',
        f"For the 24.91M configuration, one byte position across eight streams now creates {current['calls']} whole-bank gradient buffers instead of {previous['calls']}. Cumulative bank allocations fall from {previous['allocated_bytes']/2**20:.2f} MiB to {current['allocated_bytes']/2**20:.2f} MiB ({saving:.1f}% less). This measures allocated bytes, not peak process RAM.",
        f"Validation: {validation['unique_tests']} unique regression checks pass after the two batching fixes. Checks include FP64 outputs and derivatives, finite differences and higher derivatives, repeated expert selections, AdamW moments, independent backwards, cache invalidation, mixed masks, resets, hooks, state storage, checkpoint continuation and generation resume.",
        'Projection reassociation and batched operations change FP32 rounding. The regression checks compare that drift against an FP64 reference while checking discrete routes in the tested cases. The exact FP64 derivative checks remain strict.',
        f"Protocol: PyTorch {results['torch']}, {results['cpu']}, one CPU thread, {protocol['batch']} streams x {protocol['context']} byte positions, two feedback rounds, {protocol['repeats']} interleaved measured pairs after warmup. Generation uses one stream and {protocol['generate']} new bytes. The 24.91M graph uses the existing eleven-node configuration and the byte-stream policy, which leaves four memory producers inactive. These are synthetic throughput checks, not new language-training results.",
        'The native CPU extension builds locally on Linux when a C++20 compiler is available; it requires no downloads or Ninja. Initial compilation is excluded from timings. SHOEBOX_NATIVE=0 selects the portable fallback. model.batch_independent=False disables stage batching; model.set_execution_backend("reference") selects the preserved reference implementation.',
    ]
    md = ['# SHoEbox Runtime Optimization Update', '', '2026-11-06', '', paragraphs[0], '',
          '| Model | Workload | Previous seconds | Current seconds | Speedup |',
          '|---|---|---:|---:|---:|']
    md += ['| '+' | '.join(row)+' |' for row in rows]
    md += ['', paragraphs[1], '', '## Implemented changes', '']
    md += [f'- **{title}:** {body}' for title, body in changes]
    md += ['', '## Validation and reproduction', '', *[p+'\n' for p in paragraphs[2:]],
           '```bash', 'python scripts/check_all.py',
           'python scripts/benchmark_fusion.py --baseline-root ../SHoEbox_previous \\',
           '  --sizes default 25m --batch 8 --context 16 --repeats 5 --generate 32 \\',
           '  --profile --out experiments/text_regions/results/fusion_runtime/measurements.json', '```', '',
           'Use a clean bd0842e checkout for SHoEbox_previous. Raw timings, configurations and allocation counters are included beside this report in the repository.', '']
    folder = ROOT/'docs';folder.mkdir(exist_ok=True)
    (folder/'SHoEbox_Runtime_Optimization_Update.md').write_text('\n'.join(md))

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import reportlab
    fonts = Path(reportlab.__file__).parent/'fonts'
    pdfmetrics.registerFont(TTFont('RuntimeSans', str(fonts/'Vera.ttf')))
    pdfmetrics.registerFont(TTFont('RuntimeSans-Bold', str(fonts/'VeraBd.ttf')))
    pdfmetrics.registerFontFamily('RuntimeSans', normal='RuntimeSans', bold='RuntimeSans-Bold',
                                 italic='RuntimeSans', boldItalic='RuntimeSans-Bold')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodySmall',fontName='RuntimeSans',fontSize=9.5,leading=13,spaceAfter=8))
    styles['Title'].fontName='RuntimeSans-Bold'
    styles['Heading2'].fontName='RuntimeSans-Bold'
    styles['Title'].fontSize=23;styles['Title'].leading=27;styles['Title'].textColor=colors.HexColor('#214D44')
    styles['Heading2'].fontSize=13;styles['Heading2'].textColor=colors.HexColor('#214D44')
    body=styles['BodySmall']
    def para(text): return Paragraph(escape(text),body)
    story=[Paragraph('SHoEbox Runtime<br/>Optimization Update',styles['Title']),
           para('6 November 2026 | Compared with bd0842e'),Spacer(1,8),para(paragraphs[0])]
    table=Table([['Model','Workload','Before (s)','After (s)','Speedup'],*rows],colWidths=[53,148,75,75,65])
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#214D44')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,-1),'RuntimeSans'),
        ('FONTNAME',(0,0),(-1,0),'RuntimeSans-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),9),('LEADING',(0,0),(-1,-1),12),
        ('BOTTOMPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F0F5F3'),colors.white]),
        ('ALIGN',(2,1),(-1,-1),'RIGHT')]))
    story += [Spacer(1,7),table,Spacer(1,14),para(paragraphs[1]),Paragraph('Implemented changes',styles['Heading2'])]
    for title,text in changes[:3]:
        story.append(Paragraph(f'<b>{escape(title)}.</b> {escape(text)}',body))
    story.append(PageBreak())
    story.append(Paragraph('Caching, batching and verification',styles['Heading2']))
    for title,text in changes[3:]:
        story.append(Paragraph(f'<b>{escape(title)}.</b> {escape(text)}',body))
    story.append(Paragraph('Validation and measurement',styles['Heading2']))
    story += [para(text) for text in paragraphs[2:]]
    story.append(para('Reproduction commands and full measurements are supplied in the Markdown report and source archive.'))
    def footer(canvas,doc):
        canvas.setFont('RuntimeSans',8);canvas.setFillColor(colors.HexColor('#687A74'))
        canvas.drawString(48,30,'SHoEbox | Runtime optimization');canvas.drawRightString(564,30,str(doc.page))
    SimpleDocTemplate(str(folder/'SHoEbox_Runtime_Optimization_Update.pdf'),pagesize=(612,792),
        leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=48,
        title='SHoEbox Runtime Optimization Update',author='SHoEbox project').build(story,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':
    main()
