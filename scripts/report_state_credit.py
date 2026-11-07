"""Create matching Markdown and PDF summaries of the completed comparison."""
import argparse
import html
import json
from pathlib import Path
import re
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT=Path(__file__).resolve().parents[1]


def report(summary):
    a,b=[summary['arms'][k]['best'] for k in ('state','publication')]
    parent=summary['parent']; protocol=summary['arms']['state']['protocol']
    totals=summary['arms']['state']['totals']; checks=summary['checks']
    values=[parent,a,b]
    rows=[('Selected continuation update',['0',str(a['step']),str(b['step'])]),
        ('Natural-byte NLL',[f"{x['language']['nll']:.4f}" for x in values]),
        ('All-answer NLL',[f"{x['test']['answer_nll']:.4f}" for x in values]),
        ('Exact-answer NLL',[f"{x['test']['groups']['exact']['nll']:.4f}" for x in values]),
        ('Exact-answer accuracy',[f"{x['test']['groups']['exact']['accuracy']:.1%}" for x in values]),
        ('Exact NLL beyond 96 bytes',[f"{x['long_delay']['exact_beyond_96']['nll']:.4f}" for x in values]),
        ('Exact accuracy beyond 96 bytes',[f"{x['long_delay']['exact_beyond_96']['accuracy']:.1%}" for x in values])]
    table='\n'.join('| '+' | '.join([name]+v)+' |' for name,v in rows)
    d=summary['state_minus_publication']
    causal='\n'.join('| '+name+' | '+' | '.join(f'{fn(x):+.6f}' for x in values)+' |' for name,fn in (
        ('Fact-change directional log-odds',lambda x:x['counterfactual']['mean_directional_log_odds']),
        ('Producer-state swap NLL delta',lambda x:x['counterfactual']['producer_swap_nll_delta'])))
    gradients=summary['arms']['state']['state_gradient_median']
    bench=summary['trace_benchmark']
    return f'''# SHoEbox: credit through producer state

November 6, 2026 | Continuation of `4d919d1`

Delayed learning credit now follows the existing recurrent producer state across
gradient boundaries. A later task loss can train an earlier state-commit decision
after the associated publication has been replaced. Inference and all
{protocol['parameters']:,} trainable parameters are unchanged.

## Completed continuation

Two matched arms completed {protocol['protocol']['steps']} paired updates from the
latest direct-sum checkpoint. Both maintain the new traces; the control omits only
the new producer-state gradient. Every update trains on language and memory at
identical weights, then takes one optimizer step. Each arm processes
{totals['language']['bytes']:,} natural bytes, {totals['memory']['bytes']:,} task-stream
bytes and {totals['memory']['answers']} answer targets, with
{totals['language']['experts']+totals['memory']['experts']:,} expert executions.

| Held-out measurement | Parent | State credit | Publication only |
|---|---:|---:|---:|
{table}

Lower NLL is better. Relative to publication-only credit, the new control changes
natural-byte NLL by {d['language_nll']:+.4f}, answer NLL by
{d['answer_nll']:+.4f}, and delayed exact NLL by {d['delayed_exact_nll']:+.4f}.
These use independently validation-selected checkpoints from the fixed
0/16/32 schedule. The completed final checkpoints are also retained and evaluated.

Testing covers {a['language']['targets']:,} natural-byte targets and
{a['test']['answers']} answers in eight fresh contextual documents. The separate
long-delay set contains {a['long_delay']['exact_beyond_96']['count']} exact targets
more than 96 bytes after their source fact. Four valid exact values give a 25%
uniform-choice baseline. This short continuation evaluates the added learning
path; durable content-sensitive recall remains the development target.

## Causal recall probe

Eight paired histories each change exactly one earlier fact-value byte while
keeping the later context and query identical. Positive directional log-odds
means the answer distribution moves toward the changed fact. A second fork swaps
only paired producer-state contents just before the final query byte; a positive
NLL delta means the swap worsens the answer.

| Query-local measurement | Parent | State credit | Publication only |
|---|---:|---:|---:|
{causal}

Both members of a fact-change pair are answered correctly in
{parent['counterfactual']['paired_correct_fraction']:.1%} of parent pairs,
{a['counterfactual']['paired_correct_fraction']:.1%} with state credit, and
{b['counterfactual']['paired_correct_fraction']:.1%} with publication-only credit.

## Mechanics and verification

Each producer's existing contextualizer and expert-block MinKron matrices carry
bounded-size sensitivities to its admission-policy weights. The actual decay and
erase operations transport those sensitivities. Holds preserve them; replacements
can carry them forward through the recurrent computation; document resets clear
them. At a gradient boundary, the new leaf's loss gradient is contracted with the
stored prefix sensitivity once. Live-window autograd still trains every weight.

The estimator fixes historical module inputs and policy features. Historical
cross-layer, cross-region and already-detached state-to-publication derivatives
remain truncated. The corresponding fixed-input conditional gradients match
untruncated autograd in FP64. This scope is explicit in the implementation.

All {checks['unique_checks']} unique checks pass, including the existing regression
suite, ten producer-credit checks and three causal-probe checks. Coverage includes
local Jacobians, full conditional gate derivatives, replacement survival, disabled
credit equivalence, inactive rows, reset identity, backend agreement, hook cleanup,
source isolation and exact optimizer/state/credit resume. Probe checks verify the
single changed byte, independent-document equivalence and isolated state swaps.

The median delayed state-gradient norm is {gradients['language']:.6f} for language
and {gradients['memory']:.6f} for memory windows. Persistent producer sensitivities
add {protocol['producer_trace_bytes_per_source']:,} bytes
({protocol['producer_trace_bytes_per_source']/1024:.1f} KiB) per four-stream source,
independent of elapsed stream length; live graphs and boundary receipts are
additional. No replay store or new inference state is introduced.

## Execution and next step

The optional recording path recomputes local coefficients and uses per-node
contextualizers while hooks are active. In a small 2-by-8-byte CPU forward/backward
probe, the median was {bench['state']['median_seconds']:.3f} seconds with state credit
versus {bench['publication']['median_seconds']:.3f} seconds with publication credit
alone ({bench['state_over_publication']:.2f}x, three measured repeats). This probe ran
alongside the study and checks. The original trainer retains its cheaper path.

Keep the feature independently ablatable for larger shared-stream runs. Next,
test whether the delayed state responsibility becomes useful content retention
with greater exposure, and reuse recurrent kernel intermediates before scaling
the optional trace path. No separate higher-order-memory curriculum was added.

## Reproduce

Run `python scripts/run_state_credit.py --parent artifacts/state_credit/parent.pt
--out artifacts/reproduction`, then evaluate that directory with
`python scripts/evaluate_state_credit.py --root artifacts/reproduction`.
Run `python scripts/check_all.py` for verification. The archive contains source,
corpus, parent and selected/final checkpoints, optimizer/runtime/credit state,
logs, query traces and file hashes. Detailed protocol: `docs/state_credit.md`.
'''


def render(markdown,destination):
    fonts=Path('/usr/share/fonts/truetype/dejavu')
    for name,filename in [('Body','DejaVuSans.ttf'),('BodyBold','DejaVuSans-Bold.ttf'),('Code','DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/filename)))
    styles={
        'body':ParagraphStyle('body',fontName='Body',fontSize=8.7,leading=11.7,spaceAfter=7),
        'title':ParagraphStyle('title',fontName='BodyBold',fontSize=19,leading=24,textColor=HexColor('#164E48'),spaceAfter=12),
        'head':ParagraphStyle('head',fontName='BodyBold',fontSize=11.2,leading=14,textColor=HexColor('#164E48'),spaceBefore=9,spaceAfter=7,keepWithNext=True),
        'cell':ParagraphStyle('cell',fontName='Body',fontSize=8.0,leading=10.5),
        'th':ParagraphStyle('th',fontName='BodyBold',fontSize=8.0,leading=10.5,textColor=white)}
    def inline(s):
        s=html.escape(s)
        return re.sub(r'`([^`]+)`',r'<font name="Code">\1</font>',s)
    lines=markdown.splitlines();story=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1;continue
        if line=='## Mechanics and verification':story.append(PageBreak())
        if line.startswith('|'):
            data=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[v.strip() for v in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',v) for v in cells):data.append(cells)
                i+=1
            table=Table([[Paragraph(inline(v),styles['th' if j==0 else 'cell']) for v in row] for j,row in enumerate(data)],
                colWidths=[240,96,96,96],repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#164E48')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor('#EDF4F2'),white]),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([table,Spacer(1,8)]);continue
        if line.startswith('# '):story.append(Paragraph(inline(line[2:]),styles['title']))
        elif line.startswith('## '):story.append(Paragraph(inline(line[3:]),styles['head']))
        else:
            paragraph=[line]
            while i+1<len(lines) and lines[i+1].strip() and not lines[i+1].startswith(('#','|')):
                i+=1;paragraph.append(lines[i].strip())
            story.append(Paragraph(inline(' '.join(paragraph)),styles['body']))
        i+=1
    def footer(canvas,doc):
        canvas.setFont('Body',8);canvas.setFillColor(HexColor('#61716C'))
        canvas.drawString(42,23,'SHoEbox | Producer-state credit | November 6, 2026')
        canvas.drawRightString(570,23,str(doc.page))
    SimpleDocTemplate(str(destination),pagesize=(612,792),leftMargin=42,rightMargin=42,
        topMargin=36,bottomMargin=40,title='SHoEbox: credit through producer state').build(story,onFirstPage=footer,onLaterPages=footer)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,default=ROOT/'experiments/text_regions/results/state_credit/summary.json')
    p.add_argument('--out',type=Path,default=ROOT/'docs')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    markdown=report(json.loads(a.summary.read_text()))
    (a.out/'SHoEbox_State_Credit_Report.md').write_text(markdown)
    render(markdown,a.out/'SHoEbox_State_Credit_Report.pdf')
    print('Report written.')


if __name__=='__main__':
    main()
