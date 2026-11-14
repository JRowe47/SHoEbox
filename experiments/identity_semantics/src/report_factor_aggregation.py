"""Build the stage-3 Markdown/PDF report from recorded validation results."""
from pathlib import Path
import json
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables';RESULTS=ROOT/'results'
BASE='69ca74021e2e46b9b50ef6140592cbb28e102ff2'
CASES=[('1M repeated training, 1,024 bytes','stage3_1m_repeated.json'),
       ('1M natural training, 1,024 bytes','stage3_1m_natural.json'),
       ('5M repeated training, 1,024 bytes','stage3_5m_repeated.json'),
       ('1M repeated training, 4,096 bytes','stage3_1m_long.json'),
       ('1M frozen evaluation, 4,096 bytes','stage3_1m_eval.json')]
results=[json.loads((RESULTS/file).read_text()) for _,file in CASES]
for r in results:assert r['deterministic'] and all(v['credit_integrity'] for rows in r['runs'].values() for v in rows)
rows=[];numeric=[];products=[]
for (label,_),r in zip(CASES,results):
    old=r['medians']['stage2'];new=r['medians']['stage3'];c=r['comparison']
    rows.append([label,f'{old:,.1f}',f'{new:,.1f}',f'{100*(new/old-1):+.1f}%',str(r['repeats'])])
    numeric.append([label,f'{c["logits"]["max_abs"]:.3g}',f'{c["nll_delta"]:+.3g}',
                    str(c['route_trace']['expert_disagreements']),str(c['top_byte_disagreements'])])
    f=r['runs']['stage3'][0]['factor_metrics']
    if f['weight_requests']:
        products.append([label,f'{f["weight_requests"]:,}',f'{f["weight_products"]:,}',
                         f'{100*(1-f["weight_products"]/f["weight_requests"]):.1f}%'])
log=(RESULTS/'stage3_regressions.txt').read_text();assert log.rstrip().endswith('OK')
tests=int(re.findall(r'Ran (\d+) tests',log)[-1])
assert (RESULTS/'stage3_factor_tests_5m.txt').read_text().rstrip().endswith('OK')
for name in ('stage3_factor_sanitizer.txt','stage3_completion_sanitizer.txt'):
    assert 'ASan/UBSan passed' in (RESULTS/name).read_text()
compat=json.loads((RESULTS/'stage3_compatibility.json').read_text())
assert compat['checkpoint_roundtrip_bitwise'] and compat['cross_runtime_frozen_bitwise']
count=sum(len(v) for r in results for v in r['runs'].values())
summary='Stage 3 is implemented in the default reference runtime. Repeated requests for the same saved content input now share a final update factor before the dense outer product. Router contributions are combined after their individual clipping. The learning objectives, per-request policy and existing update boundaries are preserved.'
protocol='All measurements start from an actual stage-2 checkpoint after 1,024 warmup byte targets: seed 7, width 64, all reference features, periodic training at LR 0.04 and local LR 0.04/3. One pinned CPU on the Xeon Platinum 8573C; alternating serial implementations. Timings include full logit export and exclude compilation, loading, saving, hashes and the separate route-tracing replay. The natural fixture uses the same FineWeb byte stream as the prior review. Raw samples and data/source hashes are included.'
five=results[2]['runs']
spread=lambda name:f'{min(r["bytes_per_second"] for r in five[name]):.1f}-{max(r["bytes_per_second"] for r in five[name]):.1f}'
evaluation=results[4]['runs']
eval_spread=lambda name:f'{min(r["bytes_per_second"] for r in evaluation[name]):.0f}-{max(r["bytes_per_second"] for r in evaluation[name]):.0f}'
timing_note=f'The 5M samples overlap: {spread("stage2")} B/s before and {spread("stage3")} B/s after. Its small median change is effectively flat in this measurement. The 4,096-byte training row contains one pair. Frozen timing varied from {eval_spread("stage2")} B/s before to {eval_spread("stage3")} B/s after. No factor work runs in that mode, so this control does not establish an aggregation speedup for inference.'
implementation=[
    'Weight grouping keys include destination bank, full immutable input handle (serial, region and index) and feature mask. Each request applies its own clipping, learning rate, input normalization, gate and regional teaching scale before summation. Bias updates and every saved-input derivative retain their original order.',
    'Router steps are individually clipped before summation by destination bank, input handle and routing mask. Winner and runner contributions remain signed. Even exactly cancelled steps retain the original normalization of both touched banks.',
    'Nested scopes use their outer owner. Generation-tagged indices in the saved-input cache avoid hashing and per-request allocation. Factors materialize at the existing boundary in first-occurrence order, using 64-request tiles and disjoint worker rows. Workspace grows with unique inputs, rather than repeated requests.',
    'Checkpoint v39, composition ABI 10, content ABI 1 and neural ABI 1 are unchanged. Grouping scratch and product counters are excluded from serialized state. Explicit historical kernels retain their existing code paths.']
verification=f'{tests} regression tests passed. Four focused aggregation tests also passed at 5M. The scalar oracle exercises large positive/negative gradients, request-level clipping, mixed rates, sparse/dense masks, nested scopes, frozen banks, cancelling router steps and recycled handles; relative update drift is required below 1e-6. Serial and parallel grouped rows agree bit-for-bit. Both factor-kernel and full completion ASan/UBSan harnesses passed, including the live DAG beyond 16,384 records. Leak scanning is disabled on this host.'
continuation='Stage-2 checkpoints load into stage 3. A checkpoint trained in stage 3 produces bitwise-identical frozen results in both runtimes. Exporting it through stage 2 and returning to stage 3 preserves bitwise training continuation. Same-runtime split calls, save/load, evaluation and resumed training also remain exact.'
max_update=max(r['comparison']['main_update_drift_relative_l2'] for r in results if r['mode']=='periodic')
traced=sum(r['comparison']['route_trace']['regional_decisions'] for r in results)
route_disagree=sum(r['comparison']['route_trace']['expert_disagreements'] for r in results)
mask_disagree=sum(r['comparison']['route_trace']['mask_disagreements'] for r in results)
content_disagree=sum(r['comparison']['retained_content_routes']['expert_disagreements'] for r in results)
content_common=sum(r['comparison']['retained_content_routes']['common_handles'] for r in results)
note=f'These updates reassociate floating-point sums; training is not bitwise identical to stage 2. The maximum main-parameter drift relative to the stage-2 continuation update was {max_update:.3g}. Untimed replays compared {traced:,} regional expert/mask snapshots, including held decisions: {route_disagree:,} expert and {mask_disagree:,} mask disagreements. Final retained records compared {content_common:,} common content handles with {content_disagree:,} expert disagreements. The raw comparison also reports runner choices, unmatched handles, RMS errors and complete per-variant hashes.'
remaining=[['4','Share reverse traversals','Compare a union reverse pass within each fixed-weight credit scope; then short, explicitly versioned update epochs.'],
           ['5','Native small-head MinKron and local event learning','Add compact context memory, an exact active learning frontier and directional delayed credit with targeted replay. Keep exact identities and paired hippocampal episodes.'],
           ['6','Versioned forward/learning workers','Overlap previous-event learning with current forward work and fuse bank reads/updates; compare against the same serial update schedule.']]
repro='Run src/compare_factor_aggregation.py against baseline 69ca740; add --corpus for natural text and SHOEBOX_EXPERTS=64 for 5M. Run src/check_factor_compatibility.py for cross-runtime checkpoint checks. Results are in results/stage3_*.json and validation logs beside them. Rebuild this report with src/report_factor_aggregation.py. The change bundle excludes models, datasets and compiled binaries.'
assessment='Stage 3 reduces repeated matrix-update work. Deep input-derivative traversals remain; stage 4 is next, and stage 5 remains the main change aimed at the 15,000+ bytes/s target.'
def mdtable(head,body):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(row)+' |' for row in body])
md='# SHoEbox: structural optimization stage 3\n\n13 November 2026 (America/Denver) | Baseline: `'+BASE+'`\n\n'+summary+'\n\n## Matched throughput\n\n'+mdtable(['Workload','Stage 2 B/s','Stage 3 B/s','Change','Pairs'],rows)+'\n\n'+protocol+'\n\n'+timing_note+'\n\n## Dense product reduction\n\n'+mdtable(['Workload','Requests','Products','Reduction'],products)+'\n\n## Numerical validation\n\n'+mdtable(['Workload','Max logit drift','NLL delta','Route differences','Top-byte differences'],numeric)+'\n\n'+note+'\n\n## Implementation\n\n'+'\n\n'.join('- '+x for x in implementation)+'\n\n## Verification\n\n'+verification+'\n\n'+continuation+f' All {count} timed continuations passed credit-integrity checks and repeated runs were deterministic.\n\n## Reproduction\n\n'+repro+'\n\n## Remaining updates\n\n'+mdtable(['Stage','Update','Remaining work'],remaining)+'\n\n'+assessment+'\n'
OUT.mkdir(exist_ok=True);stem='SHoEbox_Structural_Optimization_Stage3';(OUT/(stem+'.md')).write_text(md)
for folder,regular,bold in [('/usr/share/fonts/truetype/open-sans','OpenSans-Regular.ttf','OpenSans-Bold.ttf'),('/usr/share/fonts/truetype/dejavu','DejaVuSans.ttf','DejaVuSans-Bold.ttf')]:
    if (Path(folder)/regular).is_file():
        pdfmetrics.registerFont(TTFont('ReportSans',str(Path(folder)/regular)))
        pdfmetrics.registerFont(TTFont('ReportSansBold',str(Path(folder)/bold)));break
else:raise RuntimeError('Missing TrueType report font')
styles=getSampleStyleSheet()
for name,size,leading,bold in [('TitleX',23,27,True),('HeadX',12,16,True),('BodyX',9.3,13,False),('SmallX',8,11,False),('CellX',8,10.5,False)]:
    styles.add(ParagraphStyle(name=name,fontName='ReportSansBold' if bold else 'ReportSans',fontSize=size,leading=leading,
                             spaceAfter=0 if name=='CellX' else 8,textColor=colors.HexColor('#16354b')))
def P(text,style='BodyX'):return Paragraph(escape(text),styles[style])
def T(head,body,widths):
    table=Table([[P(v,'CellX') for v in row] for row in [head,*body]],colWidths=widths,repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5eef3')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#c8d8e0'))]))
    return table
story=[P('SHoEbox: aggregated updates','TitleX'),P('STRUCTURAL OPTIMIZATION STAGE 3 | 13 NOVEMBER 2026','SmallX'),P(summary),
       P('Matched throughput','HeadX'),T(['Workload','Stage 2 B/s','Stage 3 B/s','Change','Pairs'],rows,[221,81,81,80,49]),Spacer(1,9),P(protocol,'SmallX'),P(timing_note,'SmallX'),
       P('Dense product reduction','HeadX'),T(['Workload','Requests','Products','Reduction'],products,[239,96,96,81]),Spacer(1,8),
       P('Baseline: '+BASE,'SmallX'),PageBreak(),P('Correctness and continuation','TitleX'),
       T(['Workload','Max logit drift','NLL delta','Route diff.','Top-byte diff.'],numeric,[216,88,88,60,60]),Spacer(1,8),P(note),P(verification),P(continuation),
       P(f'{count} timed continuations passed credit-integrity checks. Repeated runs were deterministic within each runtime.'),
       PageBreak(),P('Implementation and next stages','TitleX')]
for i,text in enumerate(implementation,1):story.append(P(str(i)+'. '+text))
story.extend([P('Remaining optimization stages','HeadX'),T(['Stage','Update','Remaining work'],remaining,[35,151,326]),Spacer(1,10),
              P(assessment),P('Reproduction and evidence','HeadX'),P(repro,'SmallX')])
def footer(canvas,doc):
    canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor('#526575'))
    canvas.drawString(50,27,'SHoEbox | Structural optimization stage 3');canvas.drawRightString(562,27,str(doc.page))
SimpleDocTemplate(str(OUT/(stem+'.pdf')),pagesize=(612,792),leftMargin=50,rightMargin=50,topMargin=42,bottomMargin=45,
                  title='SHoEbox structural optimization stage 3').build(story,onFirstPage=footer,onLaterPages=footer)
print(json.dumps(dict(tests=tests,timed_continuations=count,throughput=rows,numerical=numeric,products=products),indent=2))
