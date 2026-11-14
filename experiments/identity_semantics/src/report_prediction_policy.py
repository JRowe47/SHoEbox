"""Render Action-4 results as downloadable Markdown and PDF reports."""
import argparse,json
from pathlib import Path
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[1]
LABELS=dict(baseline='Action-3 event',predictor='Compact predictor',y='Fast/slow Y',combined='Combined')
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--revision',required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    s=json.loads((ROOT/'results/action4_summary.json').read_text());env=json.loads((ROOT/'results/action4_environment.json').read_text())
    url='https://github.com/JRowe47/SHoEbox/commit/'+a.revision;selected=s['selection']
    rows=[]
    for name,row in s['natural'].items():
        rows.append([LABELS[name],f"{row['bytes_per_second']:,.1f}",f"{row['heldout_nll']:.5f}",
                     '-' if row['time_to_target'] is None else f"{row['time_to_target']:.2f} ({row['reached']}/{row['runs']})",f"{row['parameters']:,}",f"{row['rss_mib']:.1f}"])
    repeat=[]
    for name,row in s.get('repeated',{}).items():
        repeat.append([LABELS[name],f"{row['bytes_per_second']:,.1f}",f"{row['heldout_nll']:.5f}",'-' if row['time_to_target'] is None else f"{row['time_to_target']:.2f} ({row['reached']}/{row['runs']})"])
    qualified=[[LABELS[name],f"{row['bytes_per_second']:,.1f}",f"{row['heldout_nll']:.5f}",'-' if row['time_to_target'] is None else f"{row['time_to_target']:.2f} ({row['reached']}/{row['runs']})"] for name,row in s['qualified'].items()]
    tuning=[[row['variant'],str(row['lr']),str(row['rank']),str(row['stride']),str(row['slow_scale']),f"{row['heldout_nll']:.5f}",f"{row['bytes_per_second']:,.0f}"] for row in s['tuning']]
    implementation=[
      ('Compact feature view','An immutable value snapshot records publication and semantic identity, execution tick, response-parameter version, forecasting-head version and content-bank epoch. It holds a fixed signed projection of the published response, a separate content projection and the MinKron read. The response contains completed-content and contextual contributions; content identity remains separate.'),
      ('Next-event forecast','Fourteen upper forecasting heads use 64 x rank weights (rank 8, 16, 32 or 64) instead of 64 x 64. Fixed O(64) pooling has an exact transpose for credit. The next actual completed event, including a relay, supplies the detached teacher exactly once. Idle ticks never supply targets. This head is independent of the directional credit predictor. Rank 64 reproduces the dense forecasting arithmetic.'),
      ('Fast/slow Y','At slow boundaries, Y samples the current higher-level sources, performs its normal content/context computation, and expands the same compact response view used by forecasting. Between boundaries the slow contribution is immutable. Every byte still gets a learned diagonal correction from the observed prefix, A1 and any declared previous-byte feedback, followed by tanh and the existing decoder.'),
      ('Learning schedule','The first byte starts a slow interval. Subsequent refreshes occur at the configured 2/4/8/16-byte age limit, independent of API chunks. Before refresh, the mean accumulated byte gradient teaches the saved slow publication at the saved rate and declared slow scale. Fast correction weights and byte readout learn online. Mode/rate changes close the old training interval; document drain/reset also flush it. The next slow forward reads the resulting parameters. Frozen evaluation never applies pending training work.'),
      ('Continuation and inspection','Checkpoint version 44 saves the exact prediction policy, compact weights, parameter versions, immutable feature views, slow clock, fast weights and partial gradient interval. Existing checkpoints retain their old policies. Snapshots and serial/threaded execution agree. Inspection copies current memory at API boundaries; no timers, observers or historical performance counters were added to production.')]
    validation=[
      f"{s['validation']['full_suite']} tests passed in the complete 1M regression suite. The dedicated Action-4 suite passes all {s['validation']['prediction_tests']} tests, including the additional source/order and content/context separation check.",
      'Six selected 5M tests passed: numerical gradients, full-rank forecast equivalence, serial/threaded checkpoint continuation, upper-region contributions, explicit transition to frozen evaluation, and the selected eight-byte policy through long continuation.',
      'Actual published Action-3 checkpoints continued exactly for 8,192 bytes each in reference, one-head event, two-head event and pipeline configurations, including chunk boundaries, drain and reset.',
      'Production ASan and UBSan passed compact-only, Y-only, combined, two-head and worker lifecycle cases. Leak detection was disabled because this execution environment does not support LeakSanitizer.',
      'Targeted checks cover changed context after an 8,193-byte continuation, idle gaps, immutable features, one-time relay teaching, finite-difference derivatives, corrupt-state rejection, target-independent evaluation, per-byte generation and frozen parameters.'
    ]
    protocol=(f"Host: {env['cpu']}; {env['compiler']}; {env['flags']}. Nominal 1M means 18 logical regions, width 64 and 12 experts per region. "
      'All four ablations use the same post-Action-3 event policy, matched seeds 7/11/19, fixed natural text, separate 1,024-target validation, LR 0.04 and local LR 0.04/3. '
      'Natural runs train 17,408 bytes. Timed native calls include full logits export; compilation, evaluation, memory inspection and file IO are excluded. '
      'Policy order rotates by seed and timing runs execute alone. Reported rates and losses are medians; time to target is the first observed validation point at or below NLL 3.33, sampled every 2,048 bytes and at the endpoint. '
      'The data are pinned repository README/STATUS text with separate cortical_fineweb documentation as validation, not a FineWeb corpus training run.')
    memory=('Fitted counts exclude the displaced dense forecast weights. Those legacy arrays remain allocated to preserve the existing flat-parameter layout; the compact heads therefore reduce active forecasting parameters, not the full allocation. '
      'RSS includes allocator retention and earlier untimed evaluation copies. Exact active, inactive and allocated parameter counts are saved per run.')
    policy='fresh_model(profile="event", prediction='+repr(selected['policy'])+')'
    backlog=[['5 - next','Lightweight binding frames'],['6','Sparse bank epochs and broader worker overlap'],['7','Dedicated forward-only runtime'],['8','Resident GPU backend and independent streams'],['9 - ongoing','Longer 1M/5M/100M qualification and final profile selection']]
    blocks=['# SHoEbox Action 4: compact prediction and fast/slow Y','14 November 2026',
            f'Published revision: [{a.revision}]({url}). Baseline: `{s["baseline_revision"]}`.',s['outcome'],
            '## Natural-text ablation',table(['Policy','B/s','Held-out NLL','Target time s','Fitted parameters','RSS MiB'],rows),protocol,
            '## Tuning and selection',selected['reason'],table(['Policy','B/s','Held-out NLL','Target time s (reached)'],qualified),table(['Variant','Main LR','Rank','Y stride','Y scale','Held-out NLL','B/s'],tuning),
            'Selected explicit configuration:\n\n```python\nfrom shoebox_reference import fresh_model\nm = '+policy+'\n```',
            'The default full reference and existing event/pipeline policies remain available unchanged. The measured selection advances as an explicit policy; longer qualification remains Action 9.']
    if repeat:blocks+=['## Repeated-text cross-check',table(['Policy','B/s','Held-out NLL','Target time s'],repeat),s['repeated_scope']]
    blocks+=['## Implementation']+[f'{i}. **{name}.** {body}' for i,(name,body) in enumerate(implementation,1)]
    blocks+=['## Validation']+['- '+v for v in validation]+['## Memory and remaining throughput gap',memory,s['gap'],
      '## Checkpoint use','Use `prediction=` only when creating a fresh policy or after an explicit clean reset. Existing checkpoints load their saved policy. '
      'Use `evaluation_copy(memory="cold_frozen")` for independent frozen evaluation. For an in-place transition, call `finish_prediction()` first; this closes the partial slow interval and drains bank work. '
      'Ordinary run/save calls preserve the partial interval. `finish_document()` and reset explicitly finish it.',
      '## Reproduction','Run `src/measure_prediction_policy.py --output results.json` under `experiments/identity_semantics`; its defaults reproduce the natural-text ablation. '
      'Use `--variants`, `--rates`, `--rank`, `--stride` and `--slow-scale` for the saved sweeps. '
      'Run `tests/test_prediction_policy.py`, `src/run_factor_regressions.py --output tests.txt`, and `tests/run_prediction_sanitizer.py` separately from timing.',
      'Runtime source SHA-256: `'+s['runtime_sha256']+'`. Compact raw measurements, source hashes and test logs are in `results/action4_*`. '
      'The code download includes the patch and changed source/evidence; temporary checkpoints, datasets and binaries are excluded.',
      '## Remaining plan',table(['Action','Scope'],backlog)]
    stem=a.output_dir/'SHoEbox_Action4_Compact_Prediction';stem.with_suffix('.md').write_text('\n\n'.join(blocks)+'\n')
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    font_dir=Path('/usr/share/fonts/truetype/dejavu')
    regular,bold='Helvetica','Helvetica-Bold'
    if (font_dir/'DejaVuSans.ttf').exists():
        pdfmetrics.registerFont(TTFont('Action4Sans',str(font_dir/'DejaVuSans.ttf')))
        pdfmetrics.registerFont(TTFont('Action4SansBold',str(font_dir/'DejaVuSans-Bold.ttf')))
        regular,bold='Action4Sans','Action4SansBold'
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyA4',fontName=regular,fontSize=8.6,leading=11.8,spaceAfter=7,textColor=colors.HexColor('#263747')))
    styles.add(ParagraphStyle(name='CellA4',fontName=regular,fontSize=7.5,leading=10.3));styles['Title'].fontName=bold;styles['Title'].fontSize=21;styles['Title'].leading=25;styles['Title'].textColor=colors.HexColor('#173A4A')
    styles['Heading1'].fontName=bold
    styles['Heading1'].fontSize=14;styles['Heading1'].leading=18
    story=[]
    def para(t):story.append(Paragraph(escape(t),styles['BodyA4']))
    def heading(t):story.append(Paragraph(escape(t),styles['Heading1']))
    def grid(h,r,w):
        data=[[Paragraph(escape(str(x)),styles['CellA4']) for x in row] for row in [h]+r]
        t=Table(data,colWidths=[x*516/sum(w) for x in w],repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E2EDF1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F4F7F8')])]))
        story.extend([t,Spacer(1,9)])
    story.append(Paragraph('SHoEbox Action 4<br/>Compact prediction and fast/slow Y',styles['Title']))
    para('14 November 2026 | Nominal 1M model | Completion report')
    story.append(Paragraph('Published: <link href="'+url+'">'+a.revision[:12]+'</link> | Baseline: '+s['baseline_revision'][:12],styles['BodyA4']))
    para(s['outcome']);heading('Natural-text ablation')
    grid(['Policy','B/s','NLL','Target time s','Fitted parameters','RSS MiB'],rows,[130,62,60,70,99,65]);para(protocol);para(memory)
    heading('Selected policy');para(selected['reason']);grid(['Policy','B/s','NLL','Target time s'],qualified,[150,90,90,105]);para(s['gap'])
    story.append(PageBreak());heading('Explicit computation and learning contract')
    for i,(name,body) in enumerate(implementation,1):para(f'{i}. {name}. {body}')
    heading('Checkpoint use');para(blocks[blocks.index('## Checkpoint use')+1])
    story.append(PageBreak());heading('Tuning')
    grid(['Variant','LR','Rank','Stride','Scale','NLL','B/s'],tuning,[112,52,45,45,45,74,70])
    para('Selected configuration: '+policy)
    if repeat:heading('Repeated-text cross-check');grid(['Policy','B/s','NLL','Target time s'],repeat,[150,100,100,100]);para(s['repeated_scope'])
    heading('Validation')
    for v in validation:para(v)
    story.append(PageBreak());heading('Remaining plan');grid(['Action','Scope'],backlog,[90,426])
    heading('Reproduce and inspect');para(blocks[blocks.index('## Reproduction')+1]);para(blocks[blocks.index('## Reproduction')+2])
    para('Default reference policies are preserved. Longer learning/scale qualification is the separate Action-9 track; this action provides the implementation, controlled ablations and selected explicit candidate.')
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#C6D5DD'));canvas.line(48,40,564,40);canvas.setFont(regular,8);canvas.setFillColor(colors.HexColor('#5A6D78'));canvas.drawString(48,27,'SHoEbox | Action 4 | '+a.revision[:12]);canvas.drawRightString(564,27,str(doc.page))
    SimpleDocTemplate(str(stem.with_suffix('.pdf')),pagesize=(612,792),leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=52).build(story,onFirstPage=footer,onLaterPages=footer)
    print(stem.with_suffix('.md'));print(stem.with_suffix('.pdf'))
if __name__=='__main__':main()
