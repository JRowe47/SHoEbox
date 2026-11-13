"""Render the fourth repair, verification evidence and matched measurements."""
from pathlib import Path
import html,json
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
data=json.loads((ROOT/'results/pattern_class_validation.json').read_text())
checks=json.loads((ROOT/'results/pattern_class_checks.json').read_text())
old=data['runs']['exact_bindings'];new=data['runs']['pattern_classes']
gain=100*(new['median_bytes_per_second']/old['median_bytes_per_second']-1)
intro=('Fix 4 is implemented and enabled in fresh reference models. Regions now learn reusable pattern classes separately from exact binding identity. '
       'Completion uses learned class transitions and a next-content forecast. The neural parameter count remains 1,031,952; class prototypes and statistics are additional learned dictionary state.')
sections=[
('Exact provenance remains intact','The occurrence ledger still records every child identity, source role, presence and coverage geometry. Class matching does not replace those IDs or drop held side sources. Class IDs use independent, monotonically increasing serials; expert indices are never semantic labels. Transparent relays retain their original class assignment.'),
('Role-aware learned classes','A descriptor keeps normalized learned child codes in separate role slots, with source/presence schema, order/overlap relations and bounded geometry relative to the primary clock. Primary and repeated-source roles receive full weight; singleton side roles receive 0.1 weight. Matching radius is 0.22 with a 0.35 guard per core role. Online prototype means adapt over at most 64 observations. Recognition begins at min_count, normally three observations.'),
('Predictive completion','Adjacent classes update separate transition counts and a per-class prediction of the next role-aware content vector. The target includes all child roles and a smaller contribution from the newly exported expert content. Categorical and continuous prediction confidence each supply half the completion score. Decisions use evidence accumulated before the current transition is learned. Primary gaps or overlaps still split sequences; L2 atom inputs use byte-class transitions.'),
('Ownership, replacement and checkpoints','Each region has up to 256 classes with least-recently-used replacement. Eviction never reassigns a retired ID. Per-event assignments and forecast targets follow generation-checked semantic ownership and collection. Version 38 saves class policy, prototypes, transitions and live assignments alongside composition ABI 10. Actual version-37 continuation was reproduced exactly. Versions 35-37 keep their historical exact-key recognition policy.'),
('Evaluation and upgrades','Frozen evaluation and frozen regions do not learn class statistics. Stream resets clear assignments and diagnostics while retaining fitted classes. Snapshots, frozen copies and fingerprints include the appropriate state. Upgrade an older checkpoint with reset_context() followed by configure_pattern_classes(True); changing recognition policy midstream is rejected.')]
verification=(f"{checks['regression_tests']} distinct package regression tests passed; {checks['pattern_tests_64_experts']} class tests also passed at 64 experts. "
              'ASan/UBSan passed training, draining, reset, export/import and a live semantic DAG above 16,384 entries. '
              'Current checkpoints, snapshots and chunked training reproduce exact continuation. An actual v37 checkpoint reproduced 269 continuation targets, parameters, composition, neural ownership and content eligibility bit-for-bit.')
acceptance=('Controlled cases verify class reuse under changed side context in all 12 completing regions S2-G5, provisional-to-recognized promotion, distinct primary role/order classes, changed learned content at fixed exact identity, predictive evidence before mutation, multi-frame completion, gap rejection, relay preservation, dictionary turnover and non-aliasing, frozen learning and malformed-state rejection.')
protocol=(f"Fresh 1M models, seed 7, one pinned CPU, {data['warmup_targets']:,} warmup targets followed by {data['repeats']} rounds of {data['targets_per_round']:,} targets; LR 0.04 and local LR 0.04/3. "
          'The policies differ only in class recognition/completion. Compilation, draining and evaluation are excluded from training timing. Training uses repeated sentences; held-out scoring uses separate synthetic prose.')
rows=[['Median bytes/s',f"{old['median_bytes_per_second']:.2f}",f"{new['median_bytes_per_second']:.2f}"],
      ['Median bytes/CPU-second',f"{old['median_cpu_bytes_per_second']:.2f}",f"{new['median_cpu_bytes_per_second']:.2f}"],
      ['Final training NLL',f"{old['rounds'][-1]['nll']:.6f}",f"{new['rounds'][-1]['nll']:.6f}"],
      ['Held-out NLL',f"{old['heldout']['nll']:.6f}",f"{new['heldout']['nll']:.6f}"]]
measured=(f"Throughput increased {gain:.1f}% in this matched smoke; changed completion also changes the amount of neural/credit work. "
          'All missing neural, local, content and feedback producer counters, overwritten credit and expired semantic handles remained zero. '
          'Frozen evaluation/generation preserved fitted weights and dictionary state; production and diagnostic logits matched.')
contribution=(f"On the same trained checkpoint, disabling class recognition increased held-out NLL by {new['class_ablation_delta_nll']:+.6f} nats/byte and familiar-pattern NLL by {new['familiar_delta_nll']:+.6f}. "
              f"Logit RMS differences were {new['class_ablation_logit_rms']:.6f} and {new['familiar_logit_rms']:.6f}, respectively. Positive deltas mean the enabled class policy helped prediction on those probes.")
regions=['L3','S3','R3','F3','E4','F4','V4','M4','G5']
regional=[[r,str(new['patterns'][r]['reused']),str(new['patterns'][r]['promoted']),str(new['patterns'][r]['sequence_publications'])] for r in regions]
limits=('This 3,072-target smoke shows reuse and multi-frame formation in several regions. E4 and G5 did not reach recognized classes in that short natural execution; controlled equivalent-pattern tests pass in both. Counts include bound frames and completed sequences. The report does not infer long-run language quality from this short run.')
commands=['python -m unittest discover -s experiments/identity_semantics/tests -v',
          'SHOEBOX_EXPERTS=64 python -m unittest discover -s experiments/identity_semantics/tests -p test_pattern_classes.py -v',
          'python experiments/identity_semantics/tests/run_completion_sanitizer.py',
          'python experiments/identity_semantics/src/validate_pattern_classes.py']
def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(row)+' |' for row in rows])
md='# SHoEbox: reusable pattern classes\n\n13 November 2026 | Fourth recommended fix | Base: 13aec98\n\n'+intro+'\n\n'
for title,body in sections:md+='## '+title+'\n\n'+body+'\n\n'
md+='## Verification\n\n'+verification+'\n\n'+acceptance+'\n\n'
md+='## Matched measurements\n\n'+protocol+'\n\n'+mdtable(['Measure','Exact bindings','Pattern classes'],rows)+'\n\n'+measured+'\n\n'+contribution+'\n\n'
md+='## Regional observations\n\n'+mdtable(['Region','Class reuses','Promotions','Multi-frame publications'],regional)+'\n\n'+limits+'\n\n'
md+='## Generation checks\n\nPrompt: The river | temperature 0.8 | seed 17 | printable ASCII plus newline | 96 generated bytes.\n\n'
for name,row in [('Exact bindings',old),('Pattern classes',new)]:md+=name+':\n\n```text\n'+row['generation']['text'].rstrip()+'\n```\n\n'
md+='## Reproduce\n\n```bash\n'+'\n'.join(commands)+'\n```\n\nEvidence: results/pattern_class_validation.json, pattern_class_checks.json and pattern_class_legacy37.json. No models, datasets or compiled binaries are included.\n'
(OUT/'SHoEbox_Pattern_Classes.md').write_text(md)
pdfmetrics.registerFont(TTFont('ReportSans','/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'))
pdfmetrics.registerFont(TTFont('ReportBold','/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCompact',fontName='ReportSans',fontSize=9.2,leading=12.2,spaceAfter=7,textColor=colors.HexColor('#253348')))
styles.add(ParagraphStyle(name='Label',fontName='ReportSans',fontSize=8,leading=10.3,spaceAfter=7,textColor=colors.HexColor('#58677c')))
for key in ('Title','Heading2'):
    styles[key].textColor=colors.HexColor('#172f4d');styles[key].fontName='ReportBold'
styles['Title'].fontSize=21;styles['Title'].leading=25;styles['Title'].alignment=0
styles['Heading2'].fontSize=11.5;styles['Heading2'].leading=14;styles['Heading2'].spaceBefore=7;styles['Heading2'].spaceAfter=4
story=[]
def para(text,style='BodyCompact'):story.append(Paragraph(html.escape(text),styles[style]))
def table(headers,rows,widths):
    values=[[Paragraph(html.escape(v),styles['BodyCompact']) for v in row] for row in [headers]+rows]
    t=Table(values,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f1f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#cad6e0')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),0)]))
    story.append(t);story.append(Spacer(1,6))
para('Reusable pattern classes','Title');para('SHoEbox | 13 November 2026 | Recommendation 4 | Base 13aec98','Label');para(intro)
for title,body in sections:para(title,'Heading2');para(body)
para('Verification','Heading2');para(verification);para(acceptance,'Label')
story.append(PageBreak());para('Measured behavior','Title');para(protocol)
table(['Measure','Exact bindings','Pattern classes'],rows,[211,144,144]);para(measured);para(contribution)
table(['Region','Class reuses','Promotions','Multi-frame pubs'],regional,[80,140,135,144]);para(limits,'Label')
para('Generation check','Heading2');para('Temperature 0.8; seed 17; prompt: The river; 96 bytes per policy.','Label')
for name,row in [('Exact bindings',old),('Pattern classes',new)]:para(name+': '+row['generation']['text'].replace('\n',' / '),'Label')
para('Reproduce','Heading2')
for command in commands:para(command,'Label')
def footer(canvas,doc):
    canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor('#647488'));canvas.drawString(48,27,'SHoEbox | Reusable pattern classes | v38 / ABI 10');canvas.drawRightString(A4[0]-48,27,str(doc.page))
SimpleDocTemplate(str(OUT/'SHoEbox_Pattern_Classes.pdf'),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=36,bottomMargin=43,title='SHoEbox reusable pattern classes',author='SHoEbox').build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT/'SHoEbox_Pattern_Classes.pdf')
