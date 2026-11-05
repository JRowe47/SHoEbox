"""Render the Markdown report as a PDF (requires reportlab and DejaVu fonts)."""
import argparse
import html
from pathlib import Path
import re
import textwrap
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Preformatted,KeepTogether


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('markdown',type=Path)
    parser.add_argument('pdf',type=Path);args=parser.parse_args()
    fonts=Path('/usr/share/fonts/truetype/dejavu')
    for name,filename in (('ReportSans','DejaVuSans.ttf'),('ReportSans-Bold','DejaVuSans-Bold.ttf'),('ReportMono','DejaVuSansMono.ttf')):
        pdfmetrics.registerFont(TTFont(name,str(fonts/filename)))
    pdfmetrics.registerFontFamily('ReportSans',normal='ReportSans',bold='ReportSans-Bold')
    body=ParagraphStyle('ReportBody',fontName='ReportSans',fontSize=9.3,leading=13.2,spaceAfter=8,textColor=colors.HexColor('#25323c'),allowWidows=0,allowOrphans=0)
    title=ParagraphStyle('ReportTitle',parent=body,fontName='ReportSans-Bold',fontSize=19,leading=23,spaceAfter=14,textColor=colors.HexColor('#163e49'))
    heading=ParagraphStyle('ReportHeading',parent=body,fontName='ReportSans-Bold',fontSize=12.2,leading=16,spaceBefore=13,spaceAfter=7,keepWithNext=True)
    cell=ParagraphStyle('Cell',parent=body,fontSize=8.2,leading=10.6,spaceAfter=0)
    header=ParagraphStyle('TableHead',parent=cell,fontName='ReportSans-Bold',textColor=colors.white)
    code=ParagraphStyle('Code',fontName='ReportMono',fontSize=7,leading=9,backColor=colors.HexColor('#f1f4f5'),spaceBefore=5,spaceAfter=10,borderPadding=7)
    def inline(s):
        s=html.escape(s)
        s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s)
        s=re.sub(r'`(.*?)`',r'<font name="ReportMono">\1</font>',s)
        s=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'<link href="\2" color="#246476">\1</link>',s)
        return s
    lines=args.markdown.read_text().splitlines();story=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('```'):
            block=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                block.extend(textwrap.wrap(lines[i],width=104,replace_whitespace=False,drop_whitespace=False) or ['']);i+=1
            story.append(Preformatted('\n'.join(block),code));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                parts=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',x) for x in parts):rows.append(parts)
                i+=1
            data=[[Paragraph(inline(s),header if j==0 else cell) for s in row] for j,row in enumerate(rows)]
            width=A4[0]-96;n=len(rows[0]);widths=[width/n]*n
            if n==3:widths=[width*.5,width*.25,width*.25]
            table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#234f5c')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f0f4f5')]),
                ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#234f5c')),
                ('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#d3dde0'))]))
            story.extend([KeepTogether([table]) if len(rows)<=5 else table,Spacer(1,10)]);continue
        if line.startswith('# '):story.append(Paragraph(inline(line[2:]),title));i+=1;continue
        if line.startswith('## '):story.append(Paragraph(inline(line[3:]),heading));i+=1;continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```')):
            para.append(lines[i].strip());i+=1
        story.append(KeepTogether([Paragraph(inline(' '.join(para)),body)]))
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#bacbd0'));canvas.setLineWidth(.4)
        canvas.line(48,39,A4[0]-48,39)
        canvas.setFont('ReportSans',7.5);canvas.setFillColor(colors.HexColor('#536a72'))
        canvas.drawString(48,26,'SHoEbox  /  Joint retention  /  November 5, 2026')
        canvas.drawRightString(A4[0]-48,26,str(doc.page))
    args.pdf.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(args.pdf),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=43,bottomMargin=54,
                          title='SHoEbox - Joint producer and connection learning',author='SHoEbox project')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':main()
