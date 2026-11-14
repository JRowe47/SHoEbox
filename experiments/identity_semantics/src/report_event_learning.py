"""Render the stage-5 implementation report from completed, validated evidence."""
from pathlib import Path
import json,re
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables';RESULTS=ROOT/'results'
STEM='SHoEbox_Structural_Optimization_Stage5';BASE='4d2410ba0db3f87ea9ed565c6306fc5e49f6ba84'
def read(name):return json.loads((RESULTS/name).read_text())
CASES=[('1M repeated, 1,024 bytes','stage5_1m_repeated_paired.json'),('1M FineWeb, 1,024 bytes','stage5_1m_natural_paired.json'),('5M repeated, 1,024 bytes','stage5_5m_repeated_paired.json'),('1M repeated, 4,096 bytes','stage5_1m_long_paired.json')]
results=[read(f) for _,f in CASES];sweep=read('stage5_quality_sweep.json');placement=read('stage5_placement.json');compat=read('stage5_compatibility.json')
assert len(sweep['runs'])==14 and len(placement['runs'])==6
for result in [*results,sweep,placement]:
 for r in result['runs']:
  assert r['credit_integrity'] and all(v['missing']==0 for v in r['event_metrics'].values())
for f in ('stage5_full_tests.txt','stage5_5m_tests.txt','stage5_final_1m_checks.txt','stage5_final_5m_checks.txt'):
 assert (RESULTS/f).read_text().rstrip().endswith('OK'),f
assert 'ASan/UBSan passed' in (RESULTS/'stage5_sanitizer.txt').read_text()
assert compat['bitwise_logits'] and compat['bitwise_parameters'] and compat['bitwise_full_state']
throughput=[]
for (label,_),r in zip(CASES,results):
 old,new=r['medians']['stage4'],r['medians']['stage5']
 throughput.append([label,f'{old:,.1f}',f'{new:,.1f}',f'{new/old:.2f}x',str(len(r['runs'])//2)])
byname={r['config']['name']:r for r in placement['runs']}
labels={'stage4':'Stage 4 ancestry','context1':'Context only, 1 head','context2':'Context only, 2 heads','local':'Local credit only','combined1':'Combined, 1 head','combined2':'Combined, 2 heads'}
pr=[]
for name,r in byname.items():pr.append([labels[name],f'{r["bytes_per_second"]:,.1f}',f'{r["heldout_nll"]:.6f}',f'{r["reverse_metrics"]["content_vjps"]:,}'])
base=byname['stage4'];chosen=byname['combined1'];reduction=1-chosen['reverse_metrics']['content_vjps']/base['reverse_metrics']['content_vjps']
quality={r['config']['name']:r for r in sweep['runs']}
scale_rows=[]
for name,label in [('stage4','Stage 4'),('local','Local credit'),('combined1','Combined, 1 head'),('combined2','Combined, 2 heads')]:
 keys=[f'{name}_lr{s:g}' if name!='stage4' or s!=1 else 'stage4' for s in (.5,1,2)]
 scale_rows.append([label]+[f'{quality[k]["heldout_nll"]:.6f}' for k in keys])
longrow=next(r for r in results[-1]['runs'] if r['config']['name']=='stage5')
totals={k:sum(v[k] for v in longrow['event_metrics'].values()) for k in longrow['event_metrics']['Y']}
summary=('Stage 5 is implemented as the event-v1 learning profile: native one/two-head MinKron context, an exact local frontier, '
         'directional delayed credit, and bounded targeted local replay. All 18 regions, learned constituent identities and paired EC episodes remain. '
         'Use fresh_model(profile="event"). The existing default keeps stage 4 full-ancestry learning; old checkpoints retain their saved policy.')
protocol=('One pinned CPU on INTEL(R) XEON(R) PLATINUM 8573C; one training stream and one worker. Seed 7, width 64, '
          'periodic training, LR 0.04, local LR 0.04/3. Both policies start from the same base initialization and receive their own 1,024-byte warmup. '
          'Pairs alternate execution order and include full logit export. Timing excludes compilation, warmup, evaluation, serialization and generation. '
          'The final timing suite ran after the regression jobs finished. These are different learning contracts, not numerical-parity continuations.')
assessment=(f'At matched learning rates, the combined one-head policy removes {100*reduction:.1f}% of content VJPs in the FineWeb placement comparison. '
            f'Its held-out NLL is {chosen["heldout_nll"]:.6f}, versus {base["heldout_nll"]:.6f} for stage 4 '
            f'({chosen["heldout_nll"]-base["heldout_nll"]:+.6f}). The measured speeds remain below 15,000 bytes/s.')
placement_protocol=('The isolated placement comparison trains 2,048 targets after 1,024 warmup targets and evaluates 1,024 targets from the separate FineWeb validation split. '
                    'Cold frozen episodic memory is used for validation. One run per placement; full logit export is off. '
                    'This table is separate from the paired throughput measurements. Curves show 512-byte block training NLL.')
sweep_protocol=('The learning-rate sweep repeats the same 3,072-byte training prefix. Both main and local learning rates are multiplied by 0.5, 1 or 2; '
                'the directional predictor uses its fixed 0.05 update rate. Each cell is one seed/run. All policies improve with the larger rate in this short test. '
                'Sweep timings overlapped correctness work and are not used for throughput claims; the isolated placement and paired results above supply those timings.')
implementation=[
 ('MinKron context', 'Upper regions reuse signed partitions of their 64 input features to make eight-dimensional keys and values. '
  'Each head reads its prior 8 x 8 state, then applies a decay-0.98, normalized delta write with gain 0.5. A learned 64 x 8 or 64 x 16 readout replaces '
  'the second dense neural expert transform. Saved local read/input derivatives are exact; the fixed memory recurrence has no backward temporal tape. '
  'The one-head profile adds 25,600 active fitted parameters (1,057,552 total in the 1M configuration).'),
 ('Content and context separation', 'Canonical content retains its own learned dense expert, routing, structural ID, role/order bookkeeping and completion. '
  'The context lane reuses the content expert identity for diagnostics, removes the old upper input EMA, and does not run a second feedback-conditioned router. '
  'Feedback still enters the context input. Capabilities explicitly report the replaced EMA derivative and neural feedback routing as inactive.'),
 ('Exact frontier and directional credit', 'The exact reverse frontier follows the latest four constituent encodes per region. Direct held/pending owners keep '
  'their local eligibility. Every older edge keeps immutable occurrence/role metadata plus an eight-dimensional saved content key. Its actual 64-component '
  'teacher trains a separate task or local credit predictor. New constituent encodes receive one predicted local update per objective at 0.1 of that objective rate. '
  'Predicted updates never train their own predictor. This replaces historical derivatives with learned interface credit; it does not declare them exact.'),
 ('Targeted replay and consolidation', 'One priority slot per region/objective keeps the hardest available old local record. Two records at most are replayed per '
  'byte, at 0.25 of the incoming credit rate. Replay differentiates that local record exactly and teaches its child interfaces; it does not replay an entire old DAG. '
  'Displaced candidates have already taught the predictor. Collection releases old detailed ancestry while retaining live owners and compact edge keys.'),
 ('Persistence and execution', 'Version 41 includes context state and eligibility, predictor weights, frontier keys and pending replay. Save/load and snapshots '
  'preserve the queue; document drain retires it. Changing policy requires an empty/reset context and preserves fitted event parameters. '
  'Stage 5 currently uses serial scope-union updates; bank epochs are rejected until their interaction is defined in stage 6.')]
coverage=(f'In the 4,096-byte repeated continuation, stage 5 executed {int(totals["exact_nodes"]):,} exact content-node visits, '
          f'delivered {int(totals["task_boundary"]):,} task and {int(totals["local_boundary"]):,} local boundary teachers, '
          f'and applied {int(totals["replay_applied"]):,} exact local replays. It consolidated {int(totals["consolidated_records"]):,} old detailed records. '
          'Missing-event/producer counters stayed zero. Separate short coverage tests require task learning in every upper region, '
          'and frozen interventions verify decoder effects from L3, E4, V4, G5, EC-C, EC-X and Y.')
memory=(f'For the 3,072-byte FineWeb run, serialized content-expert eligibility is {chosen["content_bytes"]/1048576:.2f} MiB '
        f'(stage 4: {base["content_bytes"]/1048576:.2f} MiB), plus {chosen["event_bytes"]/1048576:.2f} MiB of event-profile state. '
        'The existing composition slab is unchanged. This short run does not establish lower total memory use: the new compact keys/context eligibility add storage. '
        'A forced chain beyond 16,384 nodes stays within the initial slab in the event policy, while the ancestry comparator retains its growth behavior.')
verification=('260 full-suite regressions passed, plus three final targeted 1M checks, twelve 5M event checks and two final targeted 5M checks. '
              'Validation covers numerical MinKron VJPs, content/context independence, causal evaluation, all-region task coverage, frozen contribution effects, '
              'replay budget, malformed extension rejection, exact irregular chunks, save/load, snapshots and document boundaries. '
              'Both 1M and 5M also passed an 8,192-byte real-stream consolidation and exact continuation check. '
              'ASan/UBSan passed training, drain, reset, import/export, stage-4 epochs and both growing/reference and consolidating/event DAGs; leak scanning was disabled.')
compatibility=('A real stage-4 v40 checkpoint from base '+BASE+' resumed with bitwise-identical logits, fitted parameters and complete state fingerprint '
               'over 1,024 continuation targets. New event checkpoints use v41; no existing checkpoint is silently migrated.')
remaining=[['6','Persistent versioned forward/learning workers','Define one serial delayed-update schedule, then implement persistent workers and ordered bank versions; compare identical semantics with one and two workers.'],
           ['6','Fuse bank reads and updates','Combine compatible prior-event updates and next forward reads after the new workload is profiled. Measure synchronization, readout and record-management costs.'],
           ['Follow-up','Qualify the event profile at scale','Longer 1M/5M learning and stability comparisons, then 100M scaling and event-rate/budget tuning. Track bytes/s, held-out loss and total memory toward 15k.']]
repro=('Run src/measure_event_learning.py with --suite sweep, --suite placement, or --suite paired. Supply --corpus and --validation for FineWeb; '
       'use SHOEBOX_EXPERTS=64 for 5M. Run src/check_event_compatibility.py against the stage-4 runtime. '
       'Run unittest discovery for tests/test_event_learning.py and tests/run_completion_sanitizer.py. '
       'Raw results and validation logs are results/stage5_*. Rebuild this report with src/report_event_learning.py. '
       'The code bundle excludes datasets, trained checkpoints and compiled binaries.')
# Standard scientific plot; shared origin follows the warmup prefix.
fig,axes=plt.subplots(1,2,figsize=(10.7,3.45),dpi=180)
for name in ('stage4','context1','local','combined1'):
 r=byname[name];points=r['curve'];label=labels[name]
 axes[0].plot([v['targets'] for v in points],[v['train_nll'] for v in points],marker='o',markersize=3,label=label)
 axes[1].plot([v['seconds'] for v in points],[v['train_nll'] for v in points],marker='o',markersize=3,label=label)
for ax in axes:ax.set_ylabel('Training NLL (512-byte blocks)');ax.grid(alpha=.2);ax.spines[['right','top']].set_visible(False)
axes[0].set_xlabel('Continuation bytes after 1,024 warmup');axes[1].set_xlabel('Cumulative training seconds');axes[1].legend(fontsize=7,frameon=False)
fig.suptitle('Stage 5 placement comparison - 1M FineWeb, matched learning rates',fontsize=11);fig.tight_layout()
OUT.mkdir(exist_ok=True);figure=OUT/'SHoEbox_Stage5_Loss_Bytes_and_Time.png';fig.savefig(figure);plt.close(fig)
def mdtable(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(row)+' |' for row in rows])
generations=[[labels[n],byname[n]['generation']['prompt']+byname[n]['generation']['text']] for n in ('stage4','combined1','combined2')]
md=f'# SHoEbox: structural optimization stage 5\n\n13 November 2026 (America/Denver) | Baseline: `{BASE}`\n\n'+summary
md+='\n\n## Matched throughput\n\n'+mdtable(['Workload','Stage 4 B/s','Stage 5 B/s','Speedup','Pairs'],throughput)+'\n\n'+protocol+'\n\n'+assessment
md+='\n\n## Placement and learning\n\n'+placement_protocol+'\n\n'+mdtable(['Placement, 1x LR','B/s','Held-out NLL','Content VJPs'],pr)
md+='\n\n![Loss by training bytes and time](SHoEbox_Stage5_Loss_Bytes_and_Time.png)\n\n### Learning-rate sweep\n\n'+sweep_protocol+'\n\n'+mdtable(['Policy','Held-out at 0.5x','Held-out at 1x','Held-out at 2x'],scale_rows)
md+='\n\n## Implementation\n\n'+'\n\n'.join('### '+head+'\n\n'+body for head,body in implementation)
md+='\n\n## Coverage and memory\n\n'+coverage+'\n\n'+memory+'\n\n## Validation and compatibility\n\n'+verification+'\n\n'+compatibility
md+='\n\n## Generation examples\n\nPrompt: “The river ”; seed 17, temperature 0.8; 64 sampled bytes restricted to ASCII plus newline. After 3,072 training bytes.\n'
for label,text in generations:md+='\n'+label+':\n\n````text\n'+text+'\n````\n'
md+='\n## Reproduction\n\n'+repro+'\n\n## Remaining updates\n\n'+mdtable(['Stage','Update','Work remaining'],remaining)+'\n\n'+assessment+'\n'
(OUT/(STEM+'.md')).write_text(md)
for folder,regular,bold in [('/usr/share/fonts/truetype/open-sans','OpenSans-Regular.ttf','OpenSans-Bold.ttf'),('/usr/share/fonts/truetype/dejavu','DejaVuSans.ttf','DejaVuSans-Bold.ttf')]:
 if (Path(folder)/regular).is_file():
  pdfmetrics.registerFont(TTFont('ReportSans',str(Path(folder)/regular)));pdfmetrics.registerFont(TTFont('ReportSansBold',str(Path(folder)/bold)));break
styles=getSampleStyleSheet()
for name,size,leading,bold in [('TitleX',22,27,True),('HeadX',12,16,True),('BodyX',9,12.5,False),('SmallX',8,11,False),('CellX',7.7,10.3,False)]:
 styles.add(ParagraphStyle(name=name,fontName='ReportSansBold' if bold else 'ReportSans',fontSize=size,leading=leading,spaceAfter=0 if name=='CellX' else 8,textColor=colors.HexColor('#16354b')))
def P(value,style='BodyX'):return Paragraph(escape(value).replace('\n','<br/>'),styles[style])
def T(head,body,widths):
 table=Table([[P(v,'CellX') for v in row] for row in [head,*body]],colWidths=widths,repeatRows=1)
 table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5eef3')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#c8d8e0'))]));return table
story=[P('SHoEbox: local event learning','TitleX'),P('STRUCTURAL OPTIMIZATION STAGE 5 | 13 NOVEMBER 2026','SmallX'),P(summary),P('Matched throughput','HeadX'),
 T(['Workload','Stage 4 B/s','Stage 5 B/s','Speedup','Pairs'],throughput,[208,88,88,79,49]),Spacer(1,10),P(protocol,'SmallX'),P(assessment),P('Baseline: '+BASE,'SmallX'),
 PageBreak(),P('Placement and learning rates','TitleX'),P(placement_protocol,'SmallX'),T(['Placement at 1x LR','B/s','Held-out NLL','Content VJPs'],pr,[209,83,110,110]),Spacer(1,8),Image(str(figure),width=512,height=165),
 P('Learning-rate sweep','HeadX'),T(['Policy','0.5x','1x','2x'],scale_rows,[185,109,109,109]),Spacer(1,8),P(sweep_protocol,'SmallX'),PageBreak(),P('Learning contract and storage','TitleX')]
for head,body in implementation:story.extend([P(head,'HeadX'),P(body)])
story.extend([PageBreak(),P('Validation and remaining work','TitleX'),P(verification),P(compatibility,'SmallX'),P('Measured coverage','HeadX'),P(coverage,'SmallX'),P('Memory accounting','HeadX'),P(memory,'SmallX'),
 P('Remaining updates','HeadX'),T(['Stage','Update','Work remaining'],remaining,[48,138,326]),Spacer(1,8),P(repro,'SmallX'),PageBreak(),P('Generation examples','TitleX'),
 P('Prompt: The river ; seed 17, temperature 0.8; 64 sampled bytes, ASCII plus newline. After 3,072 training bytes.','SmallX'),T(['Policy','Generated text (prompt included)'],generations,[128,384]),Spacer(1,12),P(assessment)])
def footer(canvas,doc):
 canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor('#526575'));canvas.drawString(50,27,'SHoEbox | Structural optimization stage 5');canvas.drawRightString(562,27,str(doc.page))
SimpleDocTemplate(str(OUT/(STEM+'.pdf')),pagesize=(612,792),leftMargin=50,rightMargin=50,topMargin=42,bottomMargin=45,title='SHoEbox structural optimization stage 5').build(story,onFirstPage=footer,onLaterPages=footer)
print(json.dumps(dict(throughput=throughput,placement=pr,vjp_reduction=reduction,heldout_delta=chosen['heldout_nll']-base['heldout_nll'],long_coverage=totals),indent=2))
