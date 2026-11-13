"""Render the measured completion repair report as Markdown and PDF."""
from pathlib import Path
import json,statistics,html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont("ReportSans","/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("ReportSansBold","/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"))

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
s=json.loads((ROOT/'results/completion_study.json').read_text());c=json.loads((ROOT/'results/completion_controlled.json').read_text())
groups={k:[r for r in s['runs'] if r['kernel']==k] for k in ('messages','completion')}
means={k:{f:statistics.mean(r[f] for r in rows) for f in ('test_nll','validation_nll','bytes_per_second')} for k,rows in groups.items()}
speed=100*(means['completion']['bytes_per_second']/means['messages']['bytes_per_second']-1)
delta=means['completion']['test_nll']-means['messages']['test_nll']
spans=[[r,f"{c['composition'][r]['span_sum']/c['composition'][r]['published']:.2f}",str(int(c['composition'][r]['published']))] for r in ('L2','L3','S3','F4','G5')]
comparison=[[k,f"{means[k]['test_nll']:.6f}",f"{means[k]['bytes_per_second']:,.0f}"] for k in means]
contribution=[[label]+[f"{statistics.mean(r['contribution_delta_nll'][key] for r in groups[k]):+.8f}" for k in means] for label,key in [('L3 expert','expert_transform:L3'),('F4 expert','expert_transform:F4'),('G5 expert','expert_transform:G5'),('S3 payload','semantic_payload:S3')]]

intro='Implemented the two requested mechanisms in the explicit completion kernel: learned completion above L2 with multi-source roles intact, and recursively learned constituent exports with saved producer credit. Earlier runtimes remain available for exact comparison.'
mechanisms=[
 ('Completed event clock','S2 through G5 admit fresh primary-source blocks with distinct, non-overlapping occurrences. Every declared immediate side role remains in the bound frame, including explicit held occurrences. Side updates alone do not publish a new unit. Transition evidence and recognized frames control completion; novel combinations are provisional. Up to eight frames can form a unit. Partial tails drain without reusing a primary occurrence.'),
 ('Recursive exported content','A normalized role transform of child.code minus child.structure passes through a bounded learned gate into the parent\'s semantic adapter output. The result is exported and used by the next level. IDs, structural descriptors and recurrent neural context stay separate. Sixteen scalar gates add 16 parameters: 1,031,952 total at 12 experts.'),
 ('Saved credit and teaching','Every encoding saves the actual child dependencies, features and derivatives. Duplicate ancestry paths accumulate before updating their producer. A collected pool of 16,384 semantic records retains live histories, pending frames and eligibility; relays preserve the same handle. Neural eligibility is 64 events in this kernel. Bound frames are encoded on admission; regional experts execute on completion, while Y still executes each byte. Forecasts teach the next completed content once.'),
 ('Checkpoint contract','Version 33 / composition ABI 7 saves pending frames, learned gates, semantic dependencies and neural eligibility. The converter accepts version-31 relays and version-32 messages at a fresh stream boundary, preserving fitted weights and dictionaries. Old event state is not reinterpreted.')]
validation='132 existing package tests passed, followed by all 15 completion tests on the final implementation. The native AddressSanitizer/UndefinedBehaviorSanitizer harness passed training, evaluation, draining, reset and state export/import. New tests cover recognition-dependent completion, role/occurrence preservation, odd tails, two-level export sensitivity, repeated-role Jacobians, learned-gate Jacobians, saved ancestry through relays, source freezing, diagnostic parity and exact checkpoint/chunked continuation. Finite-difference tolerances are 2e-5.'
finding=f'The mechanism repair is verified, but this short comparison is not a quality improvement: mean held-out NLL increased by {delta:.6f} nats/byte and training throughput changed by {speed:.1f}%. L3/F4 expert-removal effects are also much smaller. The new event clock needs learning-rate and task-credit tuning before replacing the earlier quality baseline.'
credit='Across each 50,000-target completion run, semantic handles lost, semantic work omissions and semantic capacity prunes were all zero; forecast expiry and duplicate scheduling were also zero. The bounded neural path still recorded 122 missed parent edges per run, with zero expired retained-input edges. Semantic credit and neural credit are reported separately.'
protocol='Two fresh seeds per kernel, 12 experts, 50,000 next-byte training targets from the pinned Tiny Shakespeare stream, identical learning-rate schedule and data. Reserved test: 8,192 targets starting at heldout offset 32,768; validation: the first 8,192 targets. Training-only timing used the same pinned CPU core after warmup. This is a short smoke comparison, not a continuation of the trained 100M model. Raw rows, hashes and timings are in results/completion_study.json.'
usage='Select fresh_model(kernel="completion") from experiments/identity_semantics/runtime. The historical fresh_model() default remains composition. To migrate: python src/convert_to_completion.py OLD.npz NEW.npz. Run python -m unittest discover -s tests -v and python tests/run_completion_sanitizer.py. Reproduce measurements with python src/run_completion_study.py --data /path/to/restored/data. No trained models or corpus bytes are included.'

def mdtable(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(row)+' |' for row in rows])
md='# SHoEbox completion and learned-content repair\n\n13 November 2026 | Base: ea878a33392a36f37963d225b6479663b17504e0\n\n'+intro+'\n\n'
for title,body in mechanisms:md+='## '+title+'\n\n'+body+'\n\n'
md+='## Verification\n\n'+validation+'\n\n## Controlled nested composition\n\nThe 4,200-byte repeated-sentence training test, including final drain, produced:\n\n'+mdtable(['Region','Mean explicit span (bytes)','Publications'],spans)+'\n\nSpans are constituent coverage, including held side sources, not total model memory.\n\n## Matched short training comparison\n\n'+protocol+'\n\n'+mdtable(['Kernel','Mean test NLL','Training bytes/s'],comparison)+'\n\n'+finding+'\n\n'+mdtable(['Removed path','Messages delta NLL','Completion delta NLL'],contribution)+'\n\nPositive intervention delta means removal worsened prediction. These are conditional effects.\n\n'+credit+'\n\n## Use and reproduce\n\n'+usage+'\n'
(OUT/'SHoEbox_Completion_Fix.md').write_text(md)

styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyCompact',fontName='ReportSans',fontSize=9.5,leading=13,spaceAfter=7,textColor=colors.HexColor('#253348')))
styles.add(ParagraphStyle(name='SmallLabel',fontName='ReportSans',fontSize=8.5,leading=12,spaceAfter=10,textColor=colors.HexColor('#58677c')))
styles['Title'].fontName='ReportSansBold';styles['Heading2'].fontName='ReportSansBold';styles['Title'].fontSize=22;styles['Title'].leading=26;styles['Title'].textColor=colors.HexColor('#172f4d');styles['Title'].alignment=TA_LEFT
styles['Heading2'].fontSize=12;styles['Heading2'].leading=15;styles['Heading2'].spaceBefore=9;styles['Heading2'].spaceAfter=5;styles['Heading2'].textColor=colors.HexColor('#176b77')
story=[]
def para(t,style='BodyCompact'):story.append(Paragraph(html.escape(t),styles[style]))
def heading(t):para(t,'Heading2')
def table(headers,rows,widths):
 data=[[Paragraph(html.escape(x),styles['BodyCompact']) for x in row] for row in [headers]+rows]
 t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f1f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#95a8b8')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d8e1e8')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),1)]));story.append(t);story.append(Spacer(1,7))
para('Completed constituents, learned content','Title');para('SHoEbox implementation report | 13 November 2026 | v33 / ABI 7','SmallLabel');para(intro)
for title,body in mechanisms:heading(title);para(body)
heading('Verification');para(validation)
story.append(PageBreak());para('Measured behavior','Title');para('12 experts per region | approximately 1M parameters','SmallLabel')
heading('Controlled nested composition');para('4,200-byte repeated-sentence test, including final drain. Spans measure explicit constituent coverage, including held side sources.')
table(['Region','Mean span (bytes)','Publications'],spans,[125,180,180])
heading('Matched short training comparison');para(protocol)
table(['Kernel','Mean test NLL','Training bytes/s'],comparison,[125,180,180]);para(finding)
table(['Removed path','Messages delta NLL','Completion delta NLL'],contribution,[125,180,180]);para(credit)
heading('Use and reproduce');para(usage)
def footer(canvas,doc):
 canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor('#647488'));canvas.drawString(48,27,'SHoEbox | Completion repair | Base ea878a3');canvas.drawRightString(A4[0]-48,27,str(doc.page))
SimpleDocTemplate(str(OUT/'SHoEbox_Completion_Fix.pdf'),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=40,bottomMargin=43,title='SHoEbox completion and learned-content repair',author='SHoEbox').build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT/'SHoEbox_Completion_Fix.md');print(OUT/'SHoEbox_Completion_Fix.pdf')
