"""Render the constituent-expert implementation and its measured checks."""
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
data=json.loads((ROOT/'results/content_expert_validation.json').read_text())
checks=json.loads((ROOT/'results/content_expert_checks.json').read_text())
old=data['runs']['previous_policy'];new=data['runs']['content_experts']
cost=100*(1-new['median_bytes_per_second']/old['median_bytes_per_second'])
cpu_cost=100*(1-new['median_cpu_bytes_per_second']/old['median_cpu_bytes_per_second'])
intro=('Recommendation 3 is implemented: each encoded constituent now selects a regional expert using only its own structural descriptor and learned child content. '
       'The selected expert transforms that content before export. It shares the existing W/bias/keys banks, adds no learned parameters, and is enabled in fresh reference models. '
       'The 12-expert configuration remains 1,031,952 parameters.')
sections=[
('Constituent-only forward pass','Role-normalized learned child residuals are passed through tanh and the bounded payload gain. The content input is structure plus that payload. A masked key comparison selects the expert and runner; their margin controls the response gate. The exported residual combines the regional structural adapter, direct child payload, expert bias and gated expert matrix response. This runs at each actual encoding, including bound-frame admission before recurrent execution.'),
('Identity and context separation','Content selection and transformation read no recurrent h/u, feedback, usage counts or running means. At fixed weights and constituent inputs, unrelated retained history cannot change the code. Exact IDs, structural descriptors, role provenance, completion clocks and publication decisions keep their existing rules. Byte atoms and transparent relays keep their existing behavior. Expert choice does not introduce reusable pattern-class identities.'),
('Saved causal credit','Each semantic record owns the content expert, runner, input, mask, response and gate derivatives used at encoding. Reverse credit reaches that selected bank and recursively reaches child content, including repeated roles. Adapter and payload-gate derivatives include the new path. Shared-bank content updates are accumulated until all derivatives in the enclosing semantic/local pass have been computed. Task/local scales, semantic learning rate and frozen-node transmission remain in force.'),
('Checkpoint and lifecycle','Version 37 / composition ABI 10 saves content policy and generation-checked content eligibility. The sidecar follows semantic ownership, collection and pool growth. Versions 35/36 import without dropping live events and retain their prior content policy; version 35 also retains its prior local-credit policy. Explicitly call configure_content_experts(True) to upgrade an older loaded model. Previously encoded constituents retain their original credit semantics. Missing or mismatched new eligibility is rejected.')]
verification=(f"All {checks['regression_tests']} package regression tests passed ({checks['initial_passed']} before the runner stopped, followed by {checks['resumed_tests']} resumed checks). All {checks['content_tests_64_experts']} new content-expert tests also passed at 64 experts. "
              'ASan/UBSan passed training, evaluation, drain, reset, export/import and a live semantic DAG larger than 16,384 entries. '
              'The actual version-36 baseline reproduced continuation logits, parameters and saved state bit-for-bit after import.')
coverage=('Controlled interventions verify selected-expert effects on child code and parent input, unchanged structural identity, unrelated-expert isolation, and independence from retained history. '
          'Finite differences check own and two-hop weight/bias derivatives and routing-margin gradients, with fixed hard routing. '
          'Tests also cover shared-bank update ordering, repeated roles, frozen transmission, task/local scales, relays, 192 source replacements, malformed sidecars and exact checkpoint/chunk continuation.')
policy=('This extends online saved-eligibility credit: the saved gate/input derivative is combined with the current pre-update expert matrix. '
        'The content path uses its own input; recurrent context continues through the existing context expert path.')
protocol=(f"Seed 7; one pinned CPU; 2,048 warmup targets, then {data['repeats']} rounds of {data['targets_per_round']:,} targets at LR 0.04 and local LR 0.04/3. "
          'Both policies enable fix 2 and differ only in content-expert encoding. Compilation, draining and evaluation are outside training timing. '
          'The repeated-sentence training smoke and separate synthetic held-out prose measure this mechanism and its cost.')
rows=[['Median wall-clock bytes/s',f"{old['median_bytes_per_second']:.1f}",f"{new['median_bytes_per_second']:.1f}"],
      ['Median bytes/CPU-second',f"{old['median_cpu_bytes_per_second']:.1f}",f"{new['median_cpu_bytes_per_second']:.1f}"],
      ['Final round training NLL',f"{old['rounds'][-1]['nll']:.6f}",f"{new['rounds'][-1]['nll']:.6f}"],
      ['Synthetic held-out NLL',f"{old['heldout_nll']:.6f}",f"{new['heldout_nll']:.6f}"]]
metrics=new['content_experts'];totals={k:sum(v[k] for v in metrics.values()) for k in next(iter(metrics.values()))}
measured=(f"Median wall-clock throughput changed by {-cost:+.1f}%; CPU-time throughput changed by {-cpu_cost:+.1f}%. "
          f"The new run encoded {totals['encodes']:,} constituents and applied {totals['local_updates']:,} local and {totals['task_updates']:,} task content-update calls. "
          'All missing content/local/neural/feedback producer counters, overwritten credit, expired semantic handles and expired forecasts remained zero. '
          'Both policies preserved frozen evaluation/generation weights and exact chunked continuation. Individual round timings are recorded in the raw JSON.')
contributions=[[r,f"{new['contribution_delta_nll'][r]:+.8f}"] for r in ('L3','F4','G5')]
intervention=('The content_expert control removes only the chosen region\'s new constituent transform. '
              'Positive delta NLL means removal worsened prediction. The existing expert_transform control still targets recurrent context. '
              'Production and no-intervention diagnostic logits matched exactly.')
commands=['python -m unittest discover -s experiments/identity_semantics/tests -v',
          'SHOEBOX_EXPERTS=64 python -m unittest discover -s experiments/identity_semantics/tests -p test_content_experts.py -v',
          'python experiments/identity_semantics/tests/run_completion_sanitizer.py',
          'python experiments/identity_semantics/src/validate_content_experts.py',
          'python experiments/identity_semantics/src/report_content_experts.py']
def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(row)+' |' for row in rows])
md='# SHoEbox: constituent experts\n\n13 November 2026 | Third recommended fix | Base: 5d20d40\n\n'+intro+'\n\n'
for title,body in sections:md+='## '+title+'\n\n'+body+'\n\n'
md+='## Verification\n\n'+verification+'\n\n'+coverage+'\n\n'+policy+'\n\n'
md+='## Matched smoke results\n\n'+protocol+'\n\n'+mdtable(['Measure','Previous policy','Content experts'],rows)+'\n\n'+measured+'\n\n'
md+='## Frozen contribution check\n\n'+mdtable(['Removed content transform','Delta NLL'],contributions)+'\n\n'+intervention+'\n\n'
md+='## Generation check\n\nPrintable ASCII plus newline; temperature 0.8; seed 17; prompt "The river "; 96 bytes per policy.\n\n'
for name,row in [('Previous policy',old),('Content experts',new)]:md+=name+':\n\n```text\n'+row['generation']['text'].rstrip()+'\n```\n\n'
md+='## Reproduce\n\n```bash\n'+'\n'.join(commands)+'\n```\n\nRaw results: content_expert_validation.json, content_expert_checks.json and content_expert_legacy36.json. No trained models, corpus files or compiled binaries are included.\n'
(OUT/'SHoEbox_Constituent_Experts.md').write_text(md)

pdfmetrics.registerFont(TTFont('ReportSans','/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'))
pdfmetrics.registerFont(TTFont('ReportBold','/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'))
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyCompact',fontName='ReportSans',fontSize=9.3,leading=12.4,spaceAfter=7,textColor=colors.HexColor('#253348')))
styles.add(ParagraphStyle(name='Label',fontName='ReportSans',fontSize=8.2,leading=10.8,spaceAfter=8,textColor=colors.HexColor('#58677c')))
for key in ('Title','Heading2'):styles[key].fontName='ReportBold';styles[key].textColor=colors.HexColor('#172f4d')
styles['Title'].fontSize=22;styles['Title'].leading=26;styles['Title'].alignment=0
styles['Heading2'].fontSize=12;styles['Heading2'].leading=15;styles['Heading2'].spaceBefore=8;styles['Heading2'].spaceAfter=5
story=[]
def para(text,style='BodyCompact'):story.append(Paragraph(html.escape(text),styles[style]))
def table(headers,rows,widths):
    values=[[Paragraph(html.escape(v),styles['BodyCompact']) for v in row] for row in [headers]+rows]
    t=Table(values,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f1f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#cad6e0')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),1)]));story.append(t);story.append(Spacer(1,7))
para('Constituent experts','Title');para('SHoEbox | 13 November 2026 | Recommendation 3 | Base 5d20d40','Label');para(intro)
for title,body in sections:para(title,'Heading2');para(body)
para('Verification','Heading2');para(verification);para(coverage);para(policy,'Label')
story.append(PageBreak());para('Measured behavior','Title');para(protocol)
table(['Measure','Previous policy','Content experts'],rows,[207,146,146]);para(measured)
para('Frozen contribution check','Heading2');table(['Removed content transform','Delta NLL'],contributions,[300,199]);para(intervention)
para('Generation check','Heading2');para('Temperature 0.8; seed 17; printable ASCII plus newline; prompt: The river')
for name,row in [('Previous policy',old),('Content experts',new)]:para(name+': '+row['generation']['text'].replace('\n',' / '))
para('Reproduce','Heading2')
for command in commands:para(command,'Label')
para('Raw evidence: content_expert_validation.json, content_expert_checks.json and content_expert_legacy36.json.','Label')
def footer(canvas,doc):
    canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor('#647488'));canvas.drawString(48,27,'SHoEbox | Constituent experts | v37 / ABI 10');canvas.drawRightString(A4[0]-48,27,str(doc.page))
SimpleDocTemplate(str(OUT/'SHoEbox_Constituent_Experts.pdf'),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=38,bottomMargin=43,title='SHoEbox constituent experts',author='SHoEbox').build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT/'SHoEbox_Constituent_Experts.pdf')
