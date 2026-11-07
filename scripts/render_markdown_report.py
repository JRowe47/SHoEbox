"""Render the repository's simple Markdown research reports (requires reportlab)."""
import argparse
from html import escape
from pathlib import Path
import re


def render(source, destination, *, title='SHoEbox: shared-answer preservation during training',
           footer_text='SHoEbox | Shared-answer preservation | November 4, 2026'):
    import reportlab
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted, Image
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # ReportLab bundles these fonts; embedding them avoids viewer substitutions.
    font_root = Path(reportlab.__file__).parent / 'fonts'
    for suffix, filename in [('', 'Vera.ttf'), ('-Bold', 'VeraBd.ttf'),
                             ('-Italic', 'VeraIt.ttf'), ('-BoldItalic', 'VeraBI.ttf')]:
        pdfmetrics.registerFont(TTFont('ReportSans'+suffix, str(font_root/filename)))
    pdfmetrics.registerFontFamily('ReportSans', normal='ReportSans', bold='ReportSans-Bold',
                                  italic='ReportSans-Italic', boldItalic='ReportSans-BoldItalic')

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body', fontName='ReportSans', fontSize=9.2, leading=12.4,
        textColor=colors.HexColor('#24333c'), spaceAfter=6))
    styles.add(ParagraphStyle(name='BulletReport', parent=styles['Body'], leftIndent=10,
        bulletIndent=0, bulletFontName='ReportSans', bulletFontSize=9.2))
    styles.add(ParagraphStyle(name='TitleReport', fontName='ReportSans-Bold', fontSize=23, leading=26,
        textColor=colors.HexColor('#163f37'), spaceAfter=12))
    styles.add(ParagraphStyle(name='SectionReport', fontName='ReportSans-Bold', fontSize=13, leading=16,
        textColor=colors.HexColor('#163f37'), spaceBefore=10, spaceAfter=8, keepWithNext=True))
    styles.add(ParagraphStyle(name='Cell', fontName='ReportSans', fontSize=8.1, leading=10.8,
        alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='CodeReport', fontName='Courier', fontSize=7.2, leading=10,
        backColor=colors.HexColor('#f1f4f3'), borderPadding=8, spaceAfter=10))

    def inline(value):
        value = escape(value)
        value = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<link href="\2" color="#17654d">\1</link>', value)
        value = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', value)
        value = re.sub(r'`(.+?)`', r'<font name="Courier">\1</font>', value)
        return value

    lines = source.read_text().splitlines(); story = []; i = 0
    width = A4[0] - 86
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line == '---':
            story.append(PageBreak()); i += 1
        elif re.fullmatch(r'!\[.*?\]\((.+)\)', line):
            path = source.parent / re.fullmatch(r'!\[.*?\]\((.+)\)', line).group(1)
            figure = Image(str(path))
            factor = min(1., width / figure.imageWidth)
            figure.drawWidth = figure.imageWidth * factor
            figure.drawHeight = figure.imageHeight * factor
            figure.keepWithNext = True
            after_figure = Spacer(1, 9); after_figure.keepWithNext = True
            story.extend([figure, after_figure]); i += 1
        elif line.startswith('```'):
            i += 1; block = []
            while i < len(lines) and not lines[i].startswith('```'):
                block.append(lines[i]); i += 1
            story.append(Preformatted('\n'.join(block), styles['CodeReport'])); i += 1
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                values = [v.strip() for v in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+', v) for v in values):
                    rows.append(values)
                i += 1
            count = len(rows[0])
            first = width * (.15 if count > 2 and rows[0][0] == 'Seed' else (.32 if count > 2 else .29))
            widths = [first] + [(width - first)/(count - 1)]*(count - 1)
            content = [[Paragraph(('<b>' + inline(v) + '</b>') if n == 0 else inline(v), styles['Cell'])
                        for v in row] for n, row in enumerate(rows)]
            table = Table(content, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#deebe5')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8f7')]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#9eafa7')),
            ]))
            story.extend([table, Spacer(1, 9)])
        elif line.startswith('# '):
            story.append(Paragraph(inline(line[2:]), styles['TitleReport'])); i += 1
        elif line.startswith('## '):
            story.append(Paragraph(inline(line[3:]), styles['SectionReport'])); i += 1
        elif line.startswith('- '):
            paragraph = [line[2:]]; i += 1
            while i < len(lines) and lines[i].startswith('  ') and lines[i].strip():
                paragraph.append(lines[i].strip()); i += 1
            story.append(Paragraph(inline(' '.join(paragraph)), styles['BulletReport'], bulletText='-'))
        else:
            paragraph = [line]; i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|', '```', '---')):
                paragraph.append(lines[i]); i += 1
            story.append(Paragraph(inline(' '.join(paragraph)), styles['Body']))

    def footer(canvas, doc):
        canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#c2d0c9'))
        canvas.line(43, 37, A4[0]-43, 37)
        canvas.setFillColor(colors.HexColor('#52695d')); canvas.setFont('ReportSans', 8)
        canvas.drawString(43, 25, footer_text)
        canvas.drawRightString(A4[0]-43, 25, str(doc.page)); canvas.restoreState()

    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=43, rightMargin=43,
        topMargin=43, bottomMargin=49, title=title,
        author='SHoEbox research project')
    document.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path); parser.add_argument('destination', type=Path)
    parser.add_argument('--title', default='SHoEbox: shared-answer preservation during training')
    parser.add_argument('--footer', default='SHoEbox | Shared-answer preservation | November 4, 2026')
    args = parser.parse_args(); render(args.source, args.destination, title=args.title, footer_text=args.footer)
