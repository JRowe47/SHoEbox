"""Package the tested source, checkpoints and a rendered Markdown/PDF report."""
import argparse
import base64
import hashlib
from html import escape
import json
from pathlib import Path
import re
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def make_pdf(markdown, destination):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('BodySmall', fontName='Helvetica', fontSize=9.5, leading=13,
                              spaceAfter=6, textColor=colors.HexColor('#26313c')))
    styles.add(ParagraphStyle('TableSmall', fontName='Helvetica', fontSize=8, leading=10))
    styles.add(ParagraphStyle('CodeSmall', fontName='Courier', fontSize=7.5, leading=10,
                              spaceBefore=5, spaceAfter=8))
    story, paragraph, rows, code = [], [], [], None

    def body():
        if paragraph:
            story.append(Paragraph(escape(' '.join(paragraph)), styles['BodySmall']))
            paragraph.clear()

    def table():
        if rows:
            columns = len(rows[0])
            widths = [220, 292] if columns == 2 else [48, 260, 204]
            cells = [[Paragraph(escape(c), styles['TableSmall']) for c in row] for row in rows]
            item = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
            item.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbe9f2')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f6f8')]),
                ('LINEBELOW', (0, 0), (-1, 0), .5, colors.HexColor('#9bb3c4')),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.extend([item, Spacer(1, 10)])
            rows.clear()

    for line in markdown.splitlines():
        if line.startswith('~~~'):
            body(); table()
            if code is None:
                code = []
            else:
                story.append(Preformatted('\n'.join(code), styles['CodeSmall']))
                code = None
            continue
        if code is not None:
            code.append(line)
            continue
        if line.startswith('|'):
            body()
            cells = [c.strip() for c in line.strip('|').split('|')]
            if all(re.fullmatch(r'[-: ]+', c) for c in cells):
                continue
            rows.append(cells)
            continue
        table()
        if line.startswith('#'):
            body()
            level = len(line)-len(line.lstrip('#'))
            story.append(Paragraph(escape(line[level:].strip()), styles['Title'] if level == 1 else styles['Heading2']))
        elif not line.strip():
            body()
        else:
            paragraph.append(line.strip())
    body(); table()

    def footer(canvas, doc):
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#687887'))
        canvas.drawString(42, 25, 'SHoEbox | Cortical Input Map v2 skeleton')
        canvas.drawRightString(554, 25, str(doc.page))

    SimpleDocTemplate(str(destination), pagesize=(596, 842), leftMargin=42, rightMargin=42,
                      topMargin=36, bottomMargin=44, title='SHoEbox Cortical Skeleton').build(
                          story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'artifacts/cortical_skeleton')
    parser.add_argument('--out', type=Path, default=ROOT/'delivery')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = json.loads((args.run/'results.json').read_text())
    if not results['all_region_gradients_nonzero'] or results['final']['nll'] >= results['initial']['nll']:
        raise RuntimeError('Shakedown must show live gradients and lower fixture loss before packaging.')
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    measured = ('\n## Reproduced delivery run\n\n'
                f"Source commit: {sha}\n\n"
                f"PyTorch {results['torch']}; Python {results['python']}; {results['threads']} CPU thread.\n\n"
                '| Measurement | Result |\n| --- | ---: |\n'
                f"| Initial NLL | {results['initial']['nll']:.6f} |\n"
                f"| Final NLL | {results['final']['nll']:.6f} |\n"
                f"| Initial / final fixture accuracy | {results['initial']['accuracy']:.1%} / {results['final']['accuracy']:.1%} |\n"
                f"| Training seconds | {results['training_seconds']:.2f} |\n"
                f"| Bytes/second | {results['training_bytes_per_second']:.2f} |\n"
                f"| Runtime state bytes (batch {results['batch']}) | {results['runtime_state_bytes']} |\n"
                f"| Optimization updates | {results['steps']} |\n\n")
    report = (ROOT/'docs/cortical_skeleton.md').read_text()+measured
    md = args.out/'SHoEbox_Cortical_Skeleton_Report.md'
    pdf = args.out/'SHoEbox_Cortical_Skeleton_Report.pdf'
    md.write_text(report)
    make_pdf(report, pdf)
    import fitz
    doc = fitz.open(pdf)
    if len(doc) < 1 or '18 ordinary HAT-SHoEs' not in ''.join(page.get_text() for page in doc):
        raise RuntimeError('PDF content validation failed.')
    # Check no text escapes the page, render every page, and emit a small preview.
    from PIL import Image, ImageOps, ImageDraw
    thumbnails = []
    for i, page in enumerate(doc):
        for block in page.get_text('blocks'):
            x0, y0, x1, y1 = block[:4]
            if min(x0, y0) < 0 or x1 > page.rect.width or y1 > page.rect.height:
                raise RuntimeError(f'PDF text extends outside page {i+1}.')
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        image.thumbnail((450, 640))
        thumbnails.append(image)
    contact = Image.new('RGB', (900, 660*((len(thumbnails)+1)//2)), '#d5dce2')
    for i, image in enumerate(thumbnails):
        contact.paste(image, (450*(i % 2), 660*(i//2)))
    preview = args.out/'report_preview.png'
    contact.save(preview)
    print('PDF_PREVIEW_BASE64:'+base64.b64encode(preview.read_bytes()).decode())
    archive = args.out/'SHoEbox_Cortical_Skeleton.zip'
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in paths:
            if name and (ROOT/name).is_file():
                z.write(ROOT/name, 'SHoEbox/'+name)
        for path in sorted(args.run.iterdir()):
            if path.is_file():
                z.write(path, 'SHoEbox/artifacts/cortical_skeleton/'+path.name)
        for path in (md, pdf):
            z.write(path, 'SHoEbox/'+path.name)
    print(json.dumps({'archive': archive.name, 'bytes': archive.stat().st_size,
                      'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                      'pages': len(doc), 'source': sha, 'results': results['final']}))


if __name__ == '__main__':
    main()
