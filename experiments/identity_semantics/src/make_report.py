import csv,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,Preformatted
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
load=lambda p:json.loads((ROOT/p).read_text())
runs={name:[load(f'results/{prefix}_seed{s}/training.json') for s in (1,2)] for name,prefix in [('Original','native'),('Repaired','delivery_composition')]}
assert all(r['status']=='complete' and r['tokens']==1000000 for values in runs.values() for r in values)
perf=load('results/throughput_final.json');lesions=load('results/lesions.json');credit=load('results/credit_audit.json')
avg={name:statistics.mean(r['validation'][-1]['nll'] for r in values) for name,values in runs.items()}
gain=avg['Original']-avg['Repaired'];slower=100*(1-perf['throughput_ratio'])
speed=perf['median_bytes_s'];mib=runs['Repaired'][0]['extra_runtime_bytes']/2**20
steps=[r['tokens'] for r in runs['Original'][0]['validation']]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9,3.7),layout='constrained')
for name,color in [('Original','#748091'),('Repaired','#007E87')]:
    curves=np.array([[p['nll'] for p in r['validation']] for r in runs[name]])
    x=np.array(steps[1:])/1e6;ax.plot(x,curves.mean(0)[1:],label=name,color=color,lw=2.4,marker='o',ms=3.5)
    ax.fill_between(x,curves.min(0)[1:],curves.max(0)[1:],color=color,alpha=.13)
ax.set(xlabel='Training exposure (million bytes)',ylabel='Held-out loss (nats/byte)',title='Matched fresh starts; line = two-seed mean, shading = seed range')
ax.grid(axis='y',alpha=.18);ax.legend(frameon=False)
fig.savefig(OUT/'SHoEbox_Composition_Repair_Curves.png',dpi=180);plt.close(fig)
with (OUT/'SHoEbox_Composition_Repair_Curves.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['architecture','seed','training_bytes','heldout_nll'])
    for name,values in runs.items():
        for r in values:
            for row in r['validation']:writer.writerow([name,r['seed'],row['tokens'],row['nll']])

endpoints=[[str(s),f'{runs["Original"][s-1]["validation"][-1]["nll"]:.6f}',f'{runs["Repaired"][s-1]["validation"][-1]["nll"]:.6f}'] for s in (1,2)]
endpoints.append(['Mean',f'{avg["Original"]:.6f}',f'{avg["Repaired"]:.6f}'])
components=[
 ['Reusable identity','A separate local dictionary records ordered constituent IDs and relative geometry. IDs survive context changes, byte shifts and table eviction; recognition confidence reflects observed recurrence.'],
 ['Events and boundaries','A1/B1 emit byte atoms. Regions L2 through G5 consume each primary child event once and accumulate up to eight children. Previously learned transition strength determines update, hold and completion.'],
 ['Content and context','Pure constituent codes form the recognition/routing input and future-content teacher. Child neural outputs, feedback and other regional ports remain separately weighted context.'],
 ['Unrecognized input','A singleton transparently carries its identity, context and original neural producer. It does not execute an upper expert or occupy neural eligibility. Unseen combinations of familiar compounds are explicitly provisional.'],
 ['Learning and credit','Dictionary recurrence learning is independent of forecasting. Error-driven teaching multipliers now scale expert, router and relevant state updates together. Retained-input credit follows the input-state Jacobian.'],
 ['Memory and execution','EC stores transitions between stable content cues. Raw A1 clock cues are excluded from EC in this mode. Y still predicts every byte. Local forecasting retains the original one-in-four neural-event schedule.']]
delta={}
for name,kernel in [('Original','native'),('Repaired','composition')]:
    vals={r['condition']:r['nll'] for r in lesions if r['kernel']==kernel}
    delta[name]={k:v-vals['intact'] for k,v in vals.items() if k!='intact'}
lesion_rows=[[label,f'{delta["Original"][key]:+.6f}',f'{delta["Repaired"][key]:+.6f}'] for label,key in [('Replace Y with its fitted mean','mean_Y'),('Remove L3 output','remove_L3'),('Remove E4 output','remove_E4'),('Remove G5 output','remove_G5')]]
received=sum(v['input_credit_received'] for v in credit['composition'].values())
forwarded=sum(v['input_credit_forwarded'] for v in credit['composition'].values())
expired=sum(v['expired_input_edges'] for v in credit['composition'].values())
overwritten=sum(v['overwritten_input_credit'] for v in credit['composition'].values())
source_expired=sum(v['missing_parent'] for v in credit['laces'].values())

def mdtable(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
summary=f'''The repaired runtime implements explicit constituent identities, learned event boundaries, separate content and context, and credit through upper-region retained input. All 19 automated checks pass. Two fresh 1M-parameter runs, each trained on 1M byte targets, reduce mean held-out loss from {avg['Original']:.6f} to {avg['Repaired']:.6f} nats/byte ({100*gain/avg['Original']:.2f}% lower).'''
timing=f'''In six interleaved trials per model on the same CPU core, median training throughput was {speed['native']:,.0f} bytes/s for the original and {speed['composition']:,.0f} bytes/s for the repair: {slower:.1f}% lower throughput, or about {perf['extra_seconds_per_million']:.1f} additional seconds per million bytes at these measured rates. The new dictionary, event and credit state uses {mib:.2f} MiB, independent of expert count. Neural parameter counts are unchanged.'''
mechanism='''A pattern slot identifies an exact ordered composition. Neural experts learn how that content and its context predict subsequent observations. A bounded four-way associative dictionary provides 2,048 pattern entries per region, independently of compute experts; two additional bounded tables hold transition numerators and denominators. The default recognition threshold is three observations and the transition continuation threshold is 0.35. Content IDs are 128-bit structural fingerprints; timestamps and absolute byte offsets are excluded, while child order and relative geometry are included.'''
credit_text=f'''The retained state already affected forward predictions. The missing part was its learning dependency. For upper regions, m_t = (1-alpha)m_(t-1) + alpha(q_t+c_t). The repair sends the retained contribution to the prior INPUT state, bypassing the prior expert's output Jacobian. A fabricated self-output edge would teach the wrong computation. Transparent event relays retain the actual neural producer, so held context receives credit at its source. In seed 1, {received:,.0f} retained-input credit deliveries were recorded and {forwarded:,.0f} forwarding operations completed, with {expired:,.0f} expired retained-input edges and {overwritten:,.0f} overwritten input-credit accumulators.'''
bounded=f'''Eligibility remains bounded to 16 neural publications per region and three learned credit hops. Ordinary source-publication lookup misses still occurred ({source_expired:,.0f} over the 1M-target run); the runtime counts them explicitly. The new path repairs the absent recurrence dependency, while retaining the existing delayed local-learning approximation and finite eligibility budget.'''
validation='''The 19 checks cover stable IDs and vectors across context/byte alignment, hold behavior, ordering, explicit unknown status, held-out combinations of familiar constituents, contiguous nonduplicated spans, frozen-evaluation isolation, exact checkpoint resume, batch independence, target independence, complete teaching-scale application, retained-input Jacobians against finite differences, correct source events, legacy parity, policy validation, document reset, transparent neural provenance and source ablation. AddressSanitizer and UndefinedBehaviorSanitizer also pass for training, evaluation, draining, reset and state import/export. The 5M build passes training and exact save/resume; the 100,198,144-parameter build passes a 2,400-byte training smoke check with finite weights.'''
interpretation='''The L3 lesion causes a much larger loss increase after repair, and replacing the entire Y decoder feature with its fitted mean also hurts more. These results show increased use of the repaired lower hierarchy. E4 and G5 still have negligible measured influence after this short run. Their predictive usefulness remains an open training result; the repair does not claim that every higher region is already doing useful language work.'''
protocol='''Both architectures use 18 regions, width 64 and 12 experts per region (1,023,744 neural parameters). Seeds 1 and 2 start from the same respective neural initializations and fixed transforms. Each receives the same first 1,000,000 next-byte targets from the existing Tiny Shakespeare training split. The full 111,539-target held-out split is evaluated at initialization and every 100,000 targets using fresh stream state and frozen learned dictionaries. The learning rate follows the existing cosine schedule from 0.08 to 0.008 over 1M targets; local rate is main/3 and prototype rate is 0.002. Training uses periodic mode with one-in-four local teaching on actual neural executions. No structural labels or word boundaries are supplied to these language runs. The separate structural tests use a staged synthetic curriculum to establish familiar lower constituents before testing new combinations.'''
timing_protocol='''Timing uses an Intel Xeon Platinum 8573C CPU, no GPU, one core, six alternating-order trials per architecture, 32,768 training targets per trial and a 1,024-target warmup from an independent snapshot. It includes forward computation, routing and learning updates, and excludes compilation, snapshot construction, checkpoint I/O and evaluation. Raw whole-run timings are also preserved, but the interleaved same-core measurement is the throughput comparison.'''
usage='''The default `fresh_model()` selects the repaired `composition` kernel. `fresh_model(kernel="native")` retains the original baseline. `finish_document()` drains pending constituents without adding input bytes or synthetic targets; `reset=True` then clears working context. `snapshot_copy()` retains the complete stream and `frozen_copy()` starts a fresh stream while preserving fitted dictionaries. Checkpoint version 28 includes composition-state ABI 2. Legacy weights can be explicitly converted with the supplied migration script, which starts new constituent, working-memory and hippocampal state. The package supplies trained repaired 1M checkpoints. Full 5M/100M language training was not performed.'''
recommendation='''Use the delivered source and repaired checkpoints for the next continuation. Keep the structural tests as invariants, retain loss and throughput tracking, and repeat the higher-region contribution measurements after additional training. The original long-trained models require explicit conversion before using the new content and event semantics.'''
md=f'''# SHoEbox composition and credit repair

12 November 2026

## Result

{summary}

{mdtable(['Seed','Original held-out NLL','Repaired held-out NLL'],endpoints)}

{timing}

## Implemented changes

{mdtable(['Component','Delivered behavior'],components)}

{mechanism}

## Retained-input credit

{credit_text}

{bounded}

## Verification and regional contribution

{validation}

The following seed-1 interventions report change in held-out NLL. Positive values mean the intervention damaged prediction.

{mdtable(['Intervention','Original delta','Repaired delta'],lesion_rows)}

{interpretation}

## Method

{protocol}

{timing_protocol}

## Running and continuing

{usage}

```bash
python -m unittest discover -s tests -v
python src/benchmark.py --kernel composition --seed 1 --core 0 --tag rerun
python src/migrate_checkpoint.py old_checkpoint.npz new_checkpoint.npz
```

Set `SHOEBOX_EXPERTS=12`, `64` or `1312` before import for the 1M, 5M or uniform 100M configuration. This repair targets width 64 and uniform regional banks. The README contains the document-boundary API and conversion details. `SHA256SUMS.json` identifies every packaged file, and the result JSON files retain source-data and checkpoint hashes.

{recommendation}
'''
(OUT/'SHoEbox_Composition_Repair_Report.md').write_text(md)
summary_data=dict(mean_nll=avg,nll_improvement=gain,relative_nll_improvement=100*gain/avg['Original'],throughput=perf,neural_parameters=1023744,composition_bytes=runs['Repaired'][0]['extra_runtime_bytes'],tests_passed=19,retained_input_credit_received=received,retained_input_credit_forwarded=forwarded,expired_retained_input_edges=expired,overwritten_retained_input_credit=overwritten,ordinary_source_lookup_misses=source_expired,lesion_deltas=delta)
(OUT/'SHoEbox_Composition_Repair_Results.json').write_text(json.dumps(summary_data,indent=2)+'\n')

font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
if font.exists():
    pdfmetrics.registerFont(TTFont('Body',str(font)));pdfmetrics.registerFont(TTFont('BodyBold',str(font.with_name('DejaVuSans-Bold.ttf'))));family='Body';bold='BodyBold'
else:family='Helvetica';bold='Helvetica-Bold'
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyX',fontName=family,fontSize=9.3,leading=13.6,spaceAfter=8,textColor=colors.HexColor('#233747')))
styles.add(ParagraphStyle(name='TitleX',fontName=bold,fontSize=23,leading=28,spaceAfter=12,textColor=colors.HexColor('#14354F')))
styles.add(ParagraphStyle(name='HeadX',fontName=bold,fontSize=13,leading=18,spaceBefore=9,spaceAfter=8,textColor=colors.HexColor('#087D85')))
styles.add(ParagraphStyle(name='SmallX',fontName=family,fontSize=8.1,leading=11,spaceAfter=6,textColor=colors.HexColor('#506273')))
styles.add(ParagraphStyle(name='CellX',fontName=family,fontSize=8.2,leading=11))
def P(t,style='BodyX'):return Paragraph(escape(t),styles[style])
def T(headers,rows,widths):
    table=Table([[P(str(v),'CellX') for v in row] for row in [headers,*rows]],colWidths=widths,repeatRows=1,hAlign='LEFT')
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E2EDF0')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#8EADB5')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#D3DFE5'))]))
    return table
story=[P('SHoEbox: repairing composition and credit','TitleX'),P('12 NOVEMBER 2026  |  IMPLEMENTATION AND MEASURED RESULTS','SmallX'),P(summary),T(['Seed','Original NLL','Repaired NLL'],endpoints,[150,175,175]),Spacer(1,12),Image(str(OUT/'SHoEbox_Composition_Repair_Curves.png'),width=500,height=206),P(timing),PageBreak(),P('What changed','TitleX'),T(['Mechanism','Delivered behavior'],components,[100,400]),Spacer(1,10),P(mechanism),PageBreak(),P('Credit and verification','TitleX'),P(credit_text),P(bounded),P('Verification','HeadX'),P(validation),P('What the hierarchy contributes','HeadX'),T(['Intervention','Original delta','Repaired delta'],lesion_rows,[270,115,115]),Spacer(1,9),P(interpretation),PageBreak(),P('Reproduction and continuation','TitleX'),P('Matched language comparison','HeadX'),P(protocol),P(timing_protocol),P('Using the repaired runtime','HeadX'),P(usage),Preformatted('python -m unittest discover -s tests -v\npython src/benchmark.py --kernel composition --seed 1 --core 0 --tag rerun\npython src/migrate_checkpoint.py old.npz new.npz',styles['Code']),Spacer(1,10),P('Set SHOEBOX_EXPERTS to 12, 64 or 1312 before importing. Source, tests, data, trained 1M checkpoints, raw measurements and SHA-256 hashes are included in the accompanying ZIP. See README.md for complete usage.'),P(recommendation)]
def footer(canvas,doc):
    canvas.saveState();canvas.setFont(family,8);canvas.setFillColor(colors.HexColor('#657887'));canvas.drawString(48,27,'SHoEbox | Composition and credit repair');canvas.drawRightString(552,27,str(doc.page));canvas.restoreState()
pdf=OUT/'SHoEbox_Composition_Repair_Report.pdf'
SimpleDocTemplate(str(pdf),pagesize=(600,780),leftMargin=50,rightMargin=50,topMargin=43,bottomMargin=45,title='SHoEbox composition and credit repair',author='Joshua Rowe / ChatGPT').build(story,onFirstPage=footer,onLaterPages=footer)
print(json.dumps({'pdf':str(pdf),'markdown':str(OUT/'SHoEbox_Composition_Repair_Report.md'),'mean_nll':avg,'throughput_ratio':perf['throughput_ratio']},indent=2))
