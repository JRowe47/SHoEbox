"""Render the retention report and learning curves (ReportLab + matplotlib)."""
import argparse
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fonts = Path('/usr/share/fonts/truetype/dejavu')
    if fonts.exists():
        for name, filename in [('Body', 'DejaVuSans.ttf'), ('BodyBold', 'DejaVuSans-Bold.ttf'), ('Code', 'DejaVuSansMono.ttf')]:
            pdfmetrics.registerFont(TTFont(name, str(fonts/filename)))
        body, bold, code = 'Body', 'BodyBold', 'Code'
    else:
        body, bold, code = 'Helvetica', 'Helvetica-Bold', 'Courier'
    labels = {'legacy': 'Legacy', 'conditioned': 'Conditioned', 'competitive': 'Competitive',
              'no_credit': 'No delayed credit', 'fresh': 'Trained fresh'}
    colors = {'legacy': '#777777', 'conditioned': '#D9871B', 'competitive': '#1267A8',
              'no_credit': '#8B519B', 'fresh': '#198367'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.5), layout='constrained')
    for column, cohort in enumerate(('pilot', 'continuation')):
        for arm, label in labels.items():
            path = ROOT/'artifacts/task_retention'/cohort/arm/'history.jsonl'
            if not path.exists():
                continue
            history = [json.loads(s) for s in path.read_text().splitlines()]
            x = [r['step'] for r in history]
            for row, values in enumerate((
                [r['validation']['groups']['exact']['nll'] for r in history],
                [100*r['validation']['query_old_read_mass'] for r in history])):
                axes[row, column].plot(x, values, label=label, color=colors[arm], marker='o', markersize=3)
        axes[0, column].set_title('Scratch' if cohort == 'pilot' else 'Common-parent continuation')
        for row in range(2):
            axes[row, column].set_xlabel('Optimizer updates')
            axes[row, column].grid(alpha=.18)
            axes[row, column].legend(frameon=False, fontsize=7.5)
    axes[0, 0].set_ylabel('Exact-answer NLL (nats/byte)')
    axes[1, 0].set_ylabel('Old-publication read mass at answers (%)')
    chart = args.out/'SHoEbox_Task_Retention_Learning_Curves.png'
    fig.savefig(chart, dpi=180)
    plt.close(fig)
    styles = {
        'body': ParagraphStyle('body', fontName=body, fontSize=8.6, leading=11.8, spaceAfter=5, allowWidows=False),
        'title': ParagraphStyle('title', fontName=bold, fontSize=18, leading=23,
                                textColor=HexColor('#15496D'), spaceAfter=10, keepWithNext=True),
        'head': ParagraphStyle('head', fontName=bold, fontSize=11.2, leading=14,
                               textColor=HexColor('#15496D'), spaceBefore=8, spaceAfter=5, keepWithNext=True),
        'cell': ParagraphStyle('cell', fontName=body, fontSize=7.4, leading=10),
        'th': ParagraphStyle('th', fontName=bold, fontSize=7.4, leading=10, textColor=white),
    }
    def inline(value):
        value = html.escape(value)
        value = re.sub(r'`([^`]+)`', lambda m: f'<font name="{code}">{m[1]}</font>', value)
        return re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', value)
    lines = (ROOT/'docs/SHoEbox_Task_Retention_Report.md').read_text().splitlines()
    story = []
    validation = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line == '## Validation and next target':
            validation = [s.strip() for s in lines[index+1:] if s.strip()]
            break
        if line.startswith('|'):
            block = []
            while index < len(lines) and lines[index].strip().startswith('|'):
                cells = [s.strip() for s in lines[index].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    block.append(cells)
                index += 1
            rows = [[Paragraph(inline(c), styles['th' if row == 0 else 'cell']) for c in cells]
                    for row, cells in enumerate(block)]
            weights = [1.5]+[1]*(len(rows[0])-1)
            table = Table(rows, colWidths=[528*w/sum(weights) for w in weights], repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#15496D')), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#EEF4F7'), white]),
                ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
            story += [table, Spacer(1, 7)]
            continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]), styles['title']))
        elif line.startswith('## '):
            story.append(Paragraph(inline(line[3:]), styles['head']))
        elif line:
            story.append(Paragraph(inline(line), styles['body']))
        index += 1
    story += [PageBreak(), Paragraph('Validation and next target', styles['head'])]
    story += [Paragraph(inline(value), styles['body']) for value in validation]
    story += [Spacer(1, 10), Paragraph('Learning curves', styles['title']),
        Paragraph('Both cohorts use paired retention streams and 800 updates per arm. Upper panels show exact-answer validation loss. '
                  'Lower panels show read mass on publications older than 16 bytes at answer events; this is an activity measurement. '
                  'The report uses whole-query interventions to test whether the content helps.', styles['body']),
        Image(str(chart), width=505, height=477)]
    def footer(canvas, document):
        canvas.setFont(body, 7)
        canvas.setFillColor(HexColor('#69747C'))
        canvas.drawString(42, 24, 'SHoEbox | Retention and whole-query retrieval | November 6, 2026')
        canvas.drawRightString(570, 24, str(document.page))
    output = args.out/'SHoEbox_Task_Retention_Report.pdf'
    SimpleDocTemplate(str(output), pagesize=(612, 792), leftMargin=42, rightMargin=42,
                      topMargin=36, bottomMargin=42, title='SHoEbox retention and whole-query retrieval').build(
                          story, onFirstPage=footer, onLaterPages=footer)
    print(json.dumps(dict(pdf=str(output), chart=str(chart))))


if __name__ == '__main__':
    main()
