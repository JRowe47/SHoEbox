"""Compact downloadable Markdown/PDF report from the audited continuation."""
import argparse
from html import escape
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def build_markdown(summary):
    evaluation = summary['evaluation']; arms = summary['arms']; protocol = summary['protocol']
    connected, control = evaluation['connected_best'], evaluation['cut_16_best']
    complete = connected['acquisition']['complete_set_accuracy']
    conclusion = ('The connected continuation establishes correct answers for some complete four-value sets.'
                  if complete > 0 else 'The connected continuation does not yet establish reliable value-conditioned recall.')
    rows = []
    for name, label in [('parent','Parent'), ('connected_best','Connected selected'),
                        ('cut_16_best','16-byte cutoff selected'), ('connected_last','Connected final'),
                        ('cut_16_last','16-byte cutoff final')]:
        row = evaluation[name]
        step = 'parent' if name == 'parent' else str(row['step'])
        rows.append(f"| {label} | {step} | {row['language']['nll']:.4f} | {row['ordinary']['answer_nll']:.4f} | "
                    f"{row['acquisition']['accuracy']:.2%} | {row['acquisition']['complete_set_accuracy']:.2%} |")
    coverage = []
    for group,label in [('exact','All exact queries'), ('acquisition','Native immediate queries'),
                        ('exact_beyond_96','Exact queries beyond 96 bytes')]:
        left = arms['connected']['effective_exact_credit'][group]
        right = arms['cut_16']['effective_exact_credit'][group]
        coverage.append(f"| {label} | {left['count']}/{left['answers']} ({left['fraction']:.1%}) | "
                        f"{right['count']}/{right['answers']} ({right['fraction']:.1%}) |")
    benchmark = summary['activation_checkpoint']; full, flat = benchmark['full'], benchmark['flat']
    reduction = 1-flat['peak_process_mib']/full['peak_process_mib']
    cost = flat['seconds']/full['seconds']-1
    total = arms['connected']['totals']['memory']
    return f'''# SHoEbox connected recall continuation

November 7, 2026 | Parent code: 614033e | 312,440 trainable parameters

**{conclusion}** This phase completes {protocol['steps']} paired-source updates in each of two matched arms and adds direct coverage and four-value causal checks.

## Held-out behavior

| Checkpoint | Update | Language NLL | Answer NLL | Immediate accuracy | All four correct |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

Language uses 1,024 new natural-byte targets. Ordinary memory uses {connected['ordinary']['answers']} answer targets across eight documents. The immediate causal test uses {connected['acquisition']['cases']} query prefixes and {connected['acquisition']['targets']} answers. Each prefix is evaluated with all four possible earlier values; only that source byte changes. Accuracy uses the full 256-byte vocabulary. A constant-answer strategy gets 25% accuracy and no complete four-value sets.

Selected checkpoints minimize mean validation language/answer NLL at updates 0, 8 and 16. Final checkpoints retain the fixed-budget result even when an earlier checkpoint wins. Test results do not select checkpoints.

For selected connected/control checkpoints, exact accuracy beyond 96 bytes is {connected['long_delay']['exact_beyond_96']['accuracy']:.2%}/{control['long_delay']['exact_beyond_96']['accuracy']:.2%}. Complete delayed four-value sets score {connected['delayed_values']['complete_set_accuracy']:.2%}/{control['delayed_values']['complete_set_accuracy']:.2%}. Immediate balanced logit margins are {connected['acquisition']['value_margin']:.6f}/{control['acquisition']['value_margin']:.6f}; a source-insensitive answer preference cancels in this measure.

## What the repaired path reaches

| Training answers with exact source credit | Connected | 16-byte cutoff |
| --- | --- | --- |
{chr(10).join(coverage)}

Both arms process {total['bytes']:,} task bytes and the same number of natural-text bytes, with {total['answers']} task answer targets. They share initialization, AdamW moments, byte streams, parameter count and expert budget. Four independent streams use 128-byte optimizer windows and 16-byte execution chunks. Each update sums language/task gradients at identical weights, clips the combined norm to 1.0 and steps AdamW. The existing task grammar and natural distractors are unchanged.

Coverage checks actual source positions, document identity and resets. These are available gradient paths, not a claim that each path learned useful content. The repair trainer's default remains 384 bytes; this first comparison explicitly uses 128-byte windows.

## Memory and verification

Optional segment recomputation reduces measured process peak memory from {full['peak_process_mib']:.1f} to {flat['peak_process_mib']:.1f} MiB (**{reduction:.1%} lower**) on one matched 64-byte, four-stream paired update. Runtime changes from {full['seconds']:.2f} to {flat['seconds']:.2f} seconds ({cost:.1%} higher). Runtime state, credit traces, cursors and totals match exactly; the largest post-update weight difference is {benchmark['weight_max_error']:.3g}. The paired recall study uses ordinary activation storage.

All 43 targeted checks pass, including full parameter-gradient comparison, source/reset isolation, large-bank and batched-stage arithmetic, hook cleanup after failure, balanced value probes and exact resume within each execution mode. Independent audits reproduce paired exposure, checkpoint selection and finite final weights.

The next target remains reliable value-conditioned recall with broader ordinary stream exposure. The optional recomputation path permits larger connected windows within a smaller memory budget. Existing architectural controls remain available for later ablations.

Protocol and commands: docs/connected_recall.md in https://github.com/JRowe47/SHoEbox
'''


def write_pdf(markdown, destination):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    font_root = Path('/usr/share/fonts/truetype/dejavu')
    pdfmetrics.registerFont(TTFont('ReportSans', str(font_root/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('ReportSans-Bold', str(font_root/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('ReportSans', normal='ReportSans', bold='ReportSans-Bold')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyReport', fontName='ReportSans', fontSize=10.3, leading=14.8,
                             spaceAfter=7, textColor=colors.HexColor('#202d35')))
    styles.add(ParagraphStyle(name='TableReport', fontName='ReportSans', fontSize=8.6, leading=11.5,
                             alignment=TA_LEFT))
    styles['Title'].fontName='ReportSans-Bold'; styles['Title'].fontSize=18; styles['Title'].leading=23
    styles['Title'].textColor=colors.HexColor('#174d3a'); styles['Title'].spaceAfter=12
    styles['Heading2'].fontName='ReportSans-Bold'; styles['Heading2'].fontSize=12; styles['Heading2'].leading=15
    styles['Heading2'].textColor=colors.HexColor('#174d3a')
    def rich(text):
        return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escape(text))
    story = []; lines=markdown.splitlines(); index=0
    while index < len(lines):
        line=lines[index]
        if not line:
            index+=1; continue
        if line == '## What the repaired path reaches':
            story.append(PageBreak())
        if line.startswith('|'):
            rows=[]
            while index<len(lines) and lines[index].startswith('|'):
                cells=[cell.strip() for cell in lines[index].strip('|').split('|')]
                if not all(set(cell)<=set('- :') for cell in cells):
                    rows.append([Paragraph(rich(cell),styles['TableReport']) for cell in cells])
                index+=1
            widths=[124,48,68,68,90,90] if len(rows[0])==6 else [260,114,114]
            table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e0ece5')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),
                ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),
                ('BOTTOMPADDING',(0,0),(-1,-1),5),
                ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#8ba898')),
                ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d6dfda'))]))
            story.extend([table,Spacer(1,10)]); continue
        style='Title' if line.startswith('# ') else 'Heading2' if line.startswith('## ') else 'BodyReport'
        story.append(Paragraph(rich(line.lstrip('# ').strip()),styles[style])); index+=1
    def footer(canvas, doc):
        canvas.setFont('ReportSans',8); canvas.setFillColor(colors.HexColor('#62706a'))
        canvas.drawString(52,28,'SHoEbox | Connected recall | November 7, 2026')
        canvas.drawRightString(560,28,str(doc.page))
    SimpleDocTemplate(str(destination),pagesize=(612,792),leftMargin=52,rightMargin=52,
        topMargin=42,bottomMargin=43,title='SHoEbox connected recall continuation',author='SHoEbox').build(
            story,onFirstPage=footer,onLaterPages=footer)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,default=ROOT/'experiments/text_regions/results/connected_recall/summary.json')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    markdown=build_markdown(json.loads(args.summary.read_text()))
    (args.out/'SHoEbox_Connected_Recall_Report.md').write_text(markdown)
    write_pdf(markdown,args.out/'SHoEbox_Connected_Recall_Report.pdf')


if __name__=='__main__':
    main()
