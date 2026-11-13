"""Render the sixth repair from completed machine-readable acceptance evidence."""
import html
import json
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'results/reference_acceptance.json').read_text())
timing=json.loads((ROOT/'results/acceptance_runtime_comparison.json').read_text())
checks=json.loads((ROOT/'results/reference_acceptance_checks.json').read_text())
if data['status']!='passed' or not data['complete_acceptance']:
    raise RuntimeError('A complete passing acceptance run is required')
OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)

intro=('The sixth fix makes acceptance enforce the implemented reference contract. '
       'One command now checks live credit, temporal composition, reusable recognition, paired memory, '
       'frozen downstream effects, generation reproducibility, checkpoint continuation, the native regressions and sanitizers. '
       'The change adds observations and tests; it does not alter learning, learned parameters or checkpoint v39 / composition ABI 10.')
sections=[
('Credit and failure detection',
 'Missing neural parents, feedback producers, input dependencies, content producers and semantic handles fail acceptance in every region. '
 'Overwritten credit, omitted semantic work, capacity pruning, expired forecasts and duplicate schedules also fail. '
 'Local prediction and language/DFA deliveries are reported separately, together with eligible expert-update calls and content updates. '
 'The native causal tests check actual weight changes and derivatives. All reference features must be enabled; an inactive path cannot pass merely because its error counters are zero.'),
('Primary clocks and local sequences',
 'Each publication records primary and all-source coverage spans separately, including sums, maxima and logarithmic histograms. '
 'Hierarchy checks follow actual repeated primary-source roles, including E4 to G5. Temporal length counts completed frames in L2-G5, excluding relays; binding three source roles is still one frame. '
 'Recognition of newly formed units is separated from inherited relay recognition. Partial EC/Y startup fusions without a designated primary observation are counted explicitly as unclocked; they do not inflate primary-span means.'),
('Matched contribution and generation checks',
 'Each intervention uses a fresh frozen copy of the same fitted model and identical bytes. Channel controls preserve exact event identity, provenance and primary-clock hashes. '
 'Class removal may change completion. Both positive and negative prediction effects are retained. Cold/warm frozen memory is explicit, and zero held-out recall is not treated as a successful recall intervention. '
 'Two held-out prompts use the same seed and sampling settings in production and diagnostic kernels; outputs must match and fitted state must remain unchanged.'),
('Lifecycle and reproducibility',
 'The new observations are excluded from model fingerprints and serialized state. Reset, composition import and snapshot load begin a new observation window; existing diagnostic counters keep their established scopes. '
 'The default acceptance command runs the full regression suite and ASan/UBSan and writes explicit failure status on error. '
 'The mechanisms-only option is labelled partial and cannot report complete acceptance. The existing reference smoke shares the same credit and hierarchy contracts.')]
verification=(f"{data['regressions']['passed']} package tests passed, including "
              f"{checks['new_acceptance_tests']} new acceptance tests. All {checks['new_acceptance_tests']} also passed at 64 experts. "
              'ASan/UBSan passed, including native observation exports, lifecycle handling and a live semantic DAG above 16,384 records. '
              'Actual baseline-v39 continuation, current chunked continuation, diagnostic logits and generated outputs were exact.')
if data['regressions'].get('recovered'):
    verification+=(f" The execution session ended before the regression summary: {data['regressions']['prefix_passed']} completed passing cases were retained, "
                   f"and the remaining {data['regressions']['recovery_passed']} cases passed in a separate run. The combined test-ID ledger covers the complete suite.")
base=timing['medians']['baseline']['bytes_per_second'];now=timing['medians']['acceptance_observations']['bytes_per_second']
performance=(f"Four alternating measurements restored the same actual fifth-fix v39 checkpoint before each 512-target continuation, on one pinned CPU. "
             f"Median throughput was {base:.2f} -> {now:.2f} bytes/s ({timing['throughput_change_percent']:+.2f}%). "
             'Compilation, initialization and loading were excluded. All measured continuations produced the same logits and final fingerprint. '
             'The final timing ran after the other checks finished; an earlier concurrent timing, including its long outlier, is retained separately.')
hierarchy=[]
for region in ('L2','L3','E4','F4','G5'):
    v=data['metrics']['acceptance'][region];count=v['publications']
    hierarchy.append([region,str(count),f"{v['primary_span_sum']/count:.2f}",f"{v['coverage_span_sum']/count:.2f}",
                      str(sum(v['sequence_length_histogram'][2:])),str(v['recognized'])])
hierarchy_headers=['Region','Publications','Primary mean','Coverage mean','Multi-frame','Recognized']
hierarchy_note=('Means are bytes per publication. Multi-frame counts local temporal completion; recognized counts newly formed publications. '
                'The short 4,096-target probe has no recognized G5 units and no local G5 multi-frame completions. '
                'Its longer primary span comes from lower-level constituents. The acceptance output keeps those facts distinct instead of equating broad coverage with learned upper sequences.')
contributions=[]
for r in data['contributions']:
    name=(f"{r['region']} content" if r['kind']=='content_expert' else 'G5 feedback' if r['kind']=='feedback_channel'
          else 'Pattern classes' if r['label']=='heldout' else 'Paired recall (familiar)')
    contributions.append([name,f"{r['delta_nll']:+.8f}",f"{r['logit_rms']:.6f}",str(r['applications'])])
contribution_headers=['Removed contribution','Delta NLL','Logit RMS','Applications']
contribution_note=(f"Held-out baseline NLL is {data['heldout_nll']:.6f} nats/byte. Delta is removal minus baseline: positive means the intact contribution helped prediction. "
                   f"The held-out memory probe had {data['heldout_memory_hits']} accepted recalls; the separate familiar probe exercises paired returns. "
                   'These are measurements from a small synthetic mechanism run, not a claim that each active region improves prediction.')
commands=[
 'python experiments/identity_semantics/src/run_reference_acceptance.py',
 'SHOEBOX_EXPERTS=64 python -m unittest discover \\\n  -s experiments/identity_semantics/tests \\\n  -p test_reference_acceptance.py -v',
 'python experiments/identity_semantics/src/compare_acceptance_runtime.py \\\n  --baseline-runtime /path/to/base/experiments/identity_semantics/runtime',
]
def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(row)+' |' for row in rows])
md='# SHoEbox: reference acceptance contract\n\n13 November 2026 | Sixth recommended fix | Base: 19f2fc1\n\n'+intro+'\n\n'
for title,body in sections:md+='## '+title+'\n\n'+body+'\n\n'
md+='## Verification and cost\n\n'+verification+'\n\n'+performance+'\n\n'
md+='## Hierarchy measurements\n\n'+mdtable(hierarchy_headers,hierarchy)+'\n\n'+hierarchy_note+'\n\n'
md+='## Frozen downstream contributions\n\n'+mdtable(contribution_headers,contributions)+'\n\n'+contribution_note+'\n\n'
md+='## Held-out prompt generation\n\nSeed 17; temperature 0.8; 64 generated bytes; printable ASCII plus newline. Both outputs reproduce exactly in the diagnostic kernel.\n\n'
for r in data['generation']:md+='Prompt: '+r['prompt']+'\n\n```text\n'+r['text']+'\n```\n\n'
md+='## Reproduce\n\n```bash\n'+'\n'.join(commands)+'\n```\n\n'
md+='Evidence: results/reference_acceptance.json, reference_acceptance_tests.txt, reference_acceptance_recovery_tests.txt, reference_acceptance_recovery.json, reference_acceptance_sanitizer.txt, acceptance_runtime_comparison.json, acceptance_runtime_comparison_concurrent.json, acceptance_smoke.json and reference_acceptance_checks.json. No trained models, datasets or compiled binaries are included.\n'
(OUT/'SHoEbox_Reference_Acceptance.md').write_text(md)

pdfmetrics.registerFont(TTFont('ReportSans','/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'))
pdfmetrics.registerFont(TTFont('ReportBold','/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCompact',fontName='ReportSans',fontSize=9.2,leading=12,spaceAfter=7,textColor=colors.HexColor('#253348')))
styles.add(ParagraphStyle(name='Label',fontName='ReportSans',fontSize=8,leading=10.2,spaceAfter=6,textColor=colors.HexColor('#58677c')))
for key in ('Title','Heading2'):styles[key].textColor=colors.HexColor('#172f4d');styles[key].fontName='ReportBold'
styles['Title'].fontSize=21;styles['Title'].leading=25;styles['Title'].alignment=0
styles['Heading2'].fontSize=11.5;styles['Heading2'].leading=14;styles['Heading2'].spaceBefore=7;styles['Heading2'].spaceAfter=4
story=[]
def para(text,style='BodyCompact'):story.append(Paragraph(html.escape(text),styles[style]))
def table(headers,rows,widths):
    values=[[Paragraph(html.escape(v),styles['Label']) for v in row] for row in [headers]+rows]
    t=Table(values,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f1f5')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#cad6e0')),('LEFTPADDING',(0,0),(-1,-1),5),
        ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),1)]))
    story.append(t);story.append(Spacer(1,8))
para('Reference acceptance contract','Title');para('SHoEbox | 13 November 2026 | Recommendation 6 | Base 19f2fc1','Label');para(intro)
for title,body in sections:para(title,'Heading2');para(body)
para('Verification and cost','Heading2');para(verification);para(performance)
story.append(PageBreak());para('Measured behavior','Title')
para('Primary spans and local sequences','Heading2');table(hierarchy_headers,hierarchy,[44,83,91,91,91,99]);para(hierarchy_note)
para('Frozen downstream contributions','Heading2');table(contribution_headers,contributions,[202,105,100,92]);para(contribution_note)
para('Reproducible generation','Heading2')
for r in data['generation']:
    para('Prompt: '+r['prompt'],'Label');para(r['text'])
para('Reproduce','Heading2')
for command in commands:
    story.append(Paragraph(html.escape(command).replace('\n','<br/>'),styles['Label']))
def footer(canvas,doc):
    canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor('#647488'))
    canvas.drawString(48,27,'SHoEbox | Acceptance contract | v39 / ABI 10');canvas.drawRightString(A4[0]-48,27,str(doc.page))
SimpleDocTemplate(str(OUT/'SHoEbox_Reference_Acceptance.pdf'),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=36,bottomMargin=43,
                  title='SHoEbox reference acceptance contract',author='SHoEbox').build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT/'SHoEbox_Reference_Acceptance.pdf')
