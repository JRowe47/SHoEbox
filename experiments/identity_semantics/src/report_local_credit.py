"""Render the local prediction-credit implementation and measured checks."""
from pathlib import Path
import html
import json
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
data=json.loads((ROOT/'results/local_credit_validation.json').read_text())
checks=json.loads((ROOT/'results/local_credit_checks.json').read_text())
old=data['runs']['previous_policy'];new=data['runs']['local_credit']
cost=100*(1-new['median_bytes_per_second']/old['median_bytes_per_second'])
deliveries=sum(v['neural_deliveries'] for v in new['local_credit'].values())
updates=sum(v['neural_updates'] for v in new['local_credit'].values())
feedback=sum(v['feedback_deliveries'] for v in new['local_credit'].values())
old_range=[f(r['bytes_per_second'] for r in old['rounds']) for f in (min,max)]
new_range=[f(r['bytes_per_second'] for r in new['rounds']) for f in (min,max)]
retained=sum(v['input_deliveries'] for v in new['local_credit'].values())

intro='Implemented recommendation 2 on top of neural producer retention: regional local prediction errors now teach their causal child experts, retained-input dependencies and feedback producers. The repair is enabled in fresh reference models and adds no learned parameters: 1,031,952 at 12 experts.'
sections=[
('Causal local credit','The saved forecast supplies the issuing expert, activation, input, routing cue and generation-checked producer handles. A synchronous, depth-bounded reverse pass follows the actual context and feedback branches. All neural input and routing derivatives are formed before any expert or router is mutated. Transparent relays preserve credit without spending a neural hop.'),
('Objective and reuse policy','The pass uses the local learning rate, upper-local gain, issuer and recipient local teaching scales, credit gain and the configured reuse exponent. It never enters language/DFA queues. Context and feedback terms for the same consumed role are combined before counting one use. Repeated roles retain their own uses. Retained-input credit follows the old input rather than inventing a gradient through its expert output.'),
('Semantic and freeze behavior','The semantic branch is evaluated once per computed contribution, separately from neural transport, with local teaching scales. Frozen recipients keep their parameters while transmitting derivatives to causal ancestors. A frozen issuer, zero local rate or zero issuer local scale produces no local update.'),
('Checkpoint policy','Version 36 / composition ABI 9 stores the local-credit policy. Transient local work completes inside one teaching call; no additional queued state is serialized. Version-35 checkpoints retain their previous behavior exactly. Explicitly call model.configure_local_credit(True) to upgrade a loaded older model. Historical kernels and explicit historical profiles retain their prior policy.')]
verification=f"{checks['distinct_1m_tests']} distinct 1M regression tests passed: the 174-test package run plus the additional live regional test. All 10 local-credit tests also passed with 64 experts, and all seven final reference-policy tests passed. ASan/UBSan passed training, evaluation, drain, reset, serialization and semantic-pool growth."
coverage='Independent finite differences check child and feedback derivatives with absolute tolerance 2e-7. Controlled checks cover repeated roles, depth limits, local scales, frozen transmission, untouched language queues, single semantic application, relay transparency, 192 later source executions, and exact save/load. All 12 isolated L3/S3/F3/E4/F4/G5 objectives across two workloads changed causal child experts with language teaching disabled. A real version-35 checkpoint reproduced the previous source version bit-for-bit for parameters, saved composition/neural state and continuation logits.'
protocol=f"Seed 7; one pinned CPU; 2,048 warmup targets followed by {data['repeats']} rounds of {data['targets_per_round']:,} targets. Both policies use the same repeated-sentence input, LR 0.04 and local LR 0.04/3. Training-only timing excludes compilation, draining and evaluation. Separate synthetic prose supplies {len(__import__('validate_local_credit').HELD)-1} held-out targets. These short runs measure mechanism behavior and throughput."
timing=[['Median training bytes/s',f"{old['median_bytes_per_second']:.1f}",f"{new['median_bytes_per_second']:.1f}"],
        ['Round 1 training NLL',f"{old['rounds'][0]['nll']:.6f}",f"{new['rounds'][0]['nll']:.6f}"],
        ['Round 2 training NLL',f"{old['rounds'][-1]['nll']:.6f}",f"{new['rounds'][-1]['nll']:.6f}"],
        ['Synthetic held-out NLL',f"{old['heldout_nll']:.6f}",f"{new['heldout_nll']:.6f}"]]
measured=f"Observed median throughput decreased by {cost:.1f}%. Wall-clock rates varied across the two rounds ({old_range[0]:.0f}-{old_range[1]:.0f} bytes/s previously, {new_range[0]:.0f}-{new_range[1]:.0f} with the repair); all individual timings are retained in the raw results. The new run delivered {deliveries:,} neural uses, {feedback:,} feedback branches and {retained:,} retained-input uses, with {updates:,} descendant neural update calls. All missing local/neural/feedback producers, overwritten credit, expired semantic handles and expired forecasts remained zero."
contribution=[[region,f"{old['contribution_delta_nll'][region]:+.8f}",f"{new['contribution_delta_nll'][region]:+.8f}"] for region in ('L3','F4','G5')]
frozen='Each intervention uses a fresh frozen copy of its own trained endpoint. Positive delta NLL means removing that expert transform worsened prediction. Production/diagnostic logits matched exactly, parameters stayed unchanged during evaluation and generation, and chunked continuation was exact for both policies. This smoke does not establish a language-quality improvement.'
commands=['python -m unittest discover -s experiments/identity_semantics/tests -v',
          'python experiments/identity_semantics/tests/run_completion_sanitizer.py',
          'python experiments/identity_semantics/src/validate_local_credit.py',
          'python experiments/identity_semantics/src/report_local_credit.py']

def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(row)+' |' for row in rows])
md='# SHoEbox: local prediction credit\n\n13 November 2026 | Second recommended fix | Base: b15baf2\n\n'+intro+'\n\n'
for title,body in sections:md+='## '+title+'\n\n'+body+'\n\n'
md+='## Verification\n\n'+verification+'\n\n'+coverage+'\n\n<!-- pagebreak -->\n\n## Matched smoke results\n\n'+protocol+'\n\n'
md+=mdtable(['Measure','Previous policy','Local credit'],timing)+'\n\n'+measured+'\n\n'
md+='## Frozen expert contribution\n\n'+mdtable(['Removed transform','Previous delta NLL','Local-credit delta NLL'],contribution)+'\n\n'+frozen+'\n\n'
md+='## Generation checks\n\nPrintable ASCII plus newline; temperature 0.8; seed 17; prompt "The river "; 96 generated bytes per policy.\n\n'
for name,row in [('Previous policy',old),('Local credit',new)]:md+=name+':\n\n```text\n'+row['generation']['text'].rstrip()+'\n```\n\n'
md+='## Reproduce\n\n```bash\n'+'\n'.join(commands)+'\n```\n\nRaw evidence: results/local_credit_validation.json and results/local_credit_checks.json. No models, corpus files or compiled binaries are included.\n'
(OUT/'SHoEbox_Local_Prediction_Credit.md').write_text(md)

pdfmetrics.registerFont(TTFont('ReportSans','/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'))
pdfmetrics.registerFont(TTFont('ReportBold','/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'))
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyCompact',fontName='ReportSans',fontSize=9.5,leading=13,spaceAfter=7,textColor=colors.HexColor('#253348')))
styles.add(ParagraphStyle(name='Label',fontName='ReportSans',fontSize=8.5,leading=11,spaceAfter=10,textColor=colors.HexColor('#58677c')))
for key in ('Title','Heading2'):styles[key].fontName='ReportBold';styles[key].textColor=colors.HexColor('#172f4d')
styles['Title'].fontSize=22;styles['Title'].leading=26;styles['Title'].alignment=0
styles['Heading2'].fontSize=12;styles['Heading2'].leading=15;styles['Heading2'].spaceBefore=8;styles['Heading2'].spaceAfter=5
story=[]
def para(text,style='BodyCompact'):story.append(Paragraph(html.escape(text),styles[style]))
def table(headers,rows):
    values=[[Paragraph(html.escape(v),styles['BodyCompact']) for v in row] for row in [headers]+rows]
    t=Table(values,colWidths=[207,146,146],repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f1f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#cad6e0')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),1)]));story.append(t);story.append(Spacer(1,7))
para('Local prediction credit','Title');para('SHoEbox | 13 November 2026 | Recommendation 2 | Base b15baf2','Label');para(intro)
for title,body in sections:para(title,'Heading2');para(body)
para('Verification','Heading2');para(verification);para(coverage)
story.append(PageBreak());para('Measured behavior','Title');para(protocol)
table(['Measure','Previous policy','Local credit'],timing);para(measured)
para('Frozen expert contribution','Heading2');table(['Removed transform','Previous delta NLL','Local-credit delta NLL'],contribution);para(frozen)
para('Generation checks','Heading2');para('Temperature 0.8; seed 17; printable ASCII plus newline; prompt: The river')
for name,row in [('Previous policy',old),('Local credit',new)]:para(name+': '+row['generation']['text'].replace('\n',' / '))
para('Reproduce','Heading2')
for command in commands:para(command,'Label')
para('Raw evidence: local_credit_validation.json and local_credit_checks.json. No models, corpus files or binaries are included.','Label')
def footer(canvas,doc):
    canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor('#647488'));canvas.drawString(48,27,'SHoEbox | Local prediction credit | v36 / ABI 9');canvas.drawRightString(A4[0]-48,27,str(doc.page))
SimpleDocTemplate(str(OUT/'SHoEbox_Local_Prediction_Credit.pdf'),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=38,bottomMargin=43,title='SHoEbox local prediction credit',author='SHoEbox').build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT/'SHoEbox_Local_Prediction_Credit.pdf')
