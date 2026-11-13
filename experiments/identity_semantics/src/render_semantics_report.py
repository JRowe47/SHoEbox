from pathlib import Path
import re,html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,KeepTogether,Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
md=ROOT/'deliverables/SHoEbox_Identity_Semantics_Report.md'
pdf=md.with_suffix('.pdf')
for name,file in [('Sans','DejaVuSans.ttf'),('Sans-Bold','DejaVuSans-Bold.ttf'),('Sans-Oblique','DejaVuSans-Oblique.ttf')]:
 pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/truetype/dejavu/'+file))
pdfmetrics.registerFontFamily('Sans',normal='Sans',bold='Sans-Bold',italic='Sans-Oblique',boldItalic='Sans-Bold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyAudit',fontName='Sans',fontSize=9.4,leading=14.2,textColor=colors.HexColor('#203244'),spaceAfter=6))
styles.add(ParagraphStyle(name='CodeAudit',fontName='Courier',fontSize=8.4,leading=12,textColor=colors.HexColor('#203244'),spaceAfter=1))
styles.add(ParagraphStyle(name='BodyBeforeTable',parent=styles['BodyAudit'],keepWithNext=True))
styles.add(ParagraphStyle(name='TitleAudit',fontName='Sans-Bold',fontSize=23,leading=29,textColor=colors.HexColor('#113c55'),spaceAfter=13))
styles.add(ParagraphStyle(name='HeadAudit',fontName='Sans-Bold',fontSize=13.4,leading=18,textColor=colors.HexColor('#126078'),spaceBefore=13,spaceAfter=8,keepWithNext=True))
styles.add(ParagraphStyle(name='CellAudit',fontName='Sans',fontSize=8.1,leading=11.5,textColor=colors.HexColor('#233445')))
styles.add(ParagraphStyle(name='CellHeadAudit',parent=styles['CellAudit'],fontName='Sans-Bold',textColor=colors.white))
styles.add(ParagraphStyle(name='ListAudit',parent=styles['BodyAudit'],leftIndent=12,firstLineIndent=-10,spaceAfter=6))
def markup(s):
 s=html.escape(s)
 s=re.sub(r'\[([^]]+)\]\((https?://[^)]+)\)',lambda m:'<a href="'+m.group(2)+'" color="#126078">'+m.group(1)+'</a>',s)
 s=re.sub(r'`([^`]+)`',lambda m:'<font name="Courier" size="8.4">'+m.group(1)+'</font>',s)
 s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
 return s
lines=md.read_text().splitlines();story=[];i=0;width=492
while i<len(lines):
 line=lines[i].strip()
 if not line:i+=1;continue
 if line.startswith('!['):
  filename=line.split('](',1)[1].rstrip(')');story.append(Image(str(md.parent/filename),width=492,height=184));story.append(Spacer(1,10));i+=1;continue
 if line.startswith('```'):
  code=[];i+=1
  while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
  story.append(KeepTogether([Paragraph(html.escape(ln),styles['CodeAudit']) for ln in code]+[Spacer(1,8)]));i+=1;continue
 if line.startswith('# '):story.append(Paragraph(markup(line[2:]),styles['TitleAudit']));i+=1;continue
 if line.startswith('## '):story.append(Paragraph(markup(line[3:]),styles['HeadAudit']));i+=1;continue
 if line.startswith('|'):
  rows=[]
  while i<len(lines) and lines[i].strip().startswith('|'):
   vals=[s.strip() for s in lines[i].strip().strip('|').split('|')]
   if not all(re.fullmatch(r'[-: ]+',v) for v in vals):rows.append(vals)
   i+=1
  n=len(rows[0]);header=rows[0]
  if header[0]=='Region' and n==3:widths=[60,280,152]
  elif header[0]=='Node':widths=[40,135,317]
  elif header[0]=='Controlled change':widths=[258,100,134]
  elif header[0]=='Region/intervention':widths=[250,121,121]
  elif n==5:widths=[52,110,110,110,110]
  elif n==4:widths=[126,122,122,122]
  else:widths=[width/n]*n
  cells=[[Paragraph(markup(v),styles['CellHeadAudit'] if r==0 else styles['CellAudit']) for v in row] for r,row in enumerate(rows)]
  t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
  t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#16465d')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eef4f7'),colors.white]),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#16465d')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d5e0e7'))]))
  story.extend([KeepTogether([t]) if len(rows)<=8 else t,Spacer(1,10)]);continue
 if line.startswith('- ') or re.match(r'^\d+\. ',line):
  s='- '+line[2:] if line.startswith('- ') else line
  story.append(Paragraph(markup(s),styles['ListAudit']));i+=1;continue
 paragraph=[line];i+=1
 while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','- ')) and not re.match(r'^\d+\. ',lines[i]):paragraph.append(lines[i].strip());i+=1
 j=i
 while j<len(lines) and not lines[j].strip():j+=1
 style=styles['BodyBeforeTable'] if j<len(lines) and lines[j].lstrip().startswith('|') else styles['BodyAudit']
 body=Paragraph(markup(' '.join(paragraph)),style)
 story.append(KeepTogether([body]) if paragraph[0].startswith('Finite differences verify') else body)
def page(canvas,doc):
 canvas.saveState();w,h=doc.pagesize
 canvas.setStrokeColor(colors.HexColor('#cfdee5'));canvas.line(52,h-35,w-52,h-35)
 canvas.setFont('Sans',7.2);canvas.setFillColor(colors.HexColor('#536b7c'));canvas.drawString(52,h-27,'SHoEbox / Consistent identity and vector semantics / 1M-model study')
 canvas.line(52,39,w-52,39);canvas.drawString(52,26,'12 November 2026 | Structure, learning and throughput');canvas.drawRightString(w-52,26,str(doc.page));canvas.restoreState()
doc=SimpleDocTemplate(str(pdf),pagesize=(596,842),leftMargin=52,rightMargin=52,topMargin=52,bottomMargin=54,title='SHoEbox: consistent identity and vector semantics',author='OpenAI')
doc.build(story,onFirstPage=page,onLaterPages=page)
print(pdf)
