"""Build the stage-6 Markdown, PDF and loss figure from validated results."""
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
STEM='SHoEbox_Structural_Optimization_Stage6';BASE='fd3edf42819b880b6cf10fdb0fdde576c474d98a'
def read(name):return json.loads((RESULTS/name).read_text())
CASES=[('1M repeated / 1,024','stage6_1m_repeated_paired.json'),('1M FineWeb / 1,024','stage6_1m_natural_paired.json'),('5M repeated / 1,024','stage6_5m_repeated_paired.json'),('1M repeated / 4,096','stage6_1m_long_paired.json')]
results=[read(f) for _,f in CASES];sweep=read('stage6_quality_sweep.json');profile=read('stage6_profile.json');compat=read('stage6_stage5_compatibility.json');long=read('stage6_long_check_e12.json')
for result in results:
 assert result['bitwise_pipeline_parity']
for result in [*results,sweep]:
 for r in result['runs']:assert r['credit_integrity'] and all(v['missing']==0 for v in r['event_metrics'].values())
counts={}
for f in ('stage6_full_tests.txt','stage6_pipeline_tests.txt','stage6_5m_tests.txt'):
 s=(RESULTS/f).read_text();assert s.rstrip().endswith('OK'),f;counts[f]=int(re.search(r'Ran (\d+) tests',s)[1])
assert 'ASan/UBSan passed' in (RESULTS/'stage6_sanitizer.txt').read_text()
assert profile['bitwise_parity'] and compat['bitwise_full_state'] and long['bitwise_restart']
rows=[]
for (label,_),r in zip(CASES,results):
 m=r['medians'];rows.append([label]+[f'{m[k]:,.1f}' for k in ('stage5','serial_unfused','serial_fused','two_workers')])
quality={r['config']['name']:r for r in sweep['runs']}
qrows=[[label]+[f'{quality[f"{policy}_{scale:g}"]["heldout_nll"]:.6f}' for scale in (.5,1.,2.)] for policy,label in [('event','Stage 5'),('pipeline','Stage 6')]]
summary=('Stage 6 implements a persistent content-bank learning worker, a serial oracle for the same one-byte-delayed update schedule, '
         'ordered sparse bank versions, fused update kernels and fused weight installation/forward reads. All 18 regions and stage-5 event learning remain. '
         'Use fresh_model(profile="pipeline"); pipeline_workers=2 adds the learning thread. The default pipeline execution is serial with fusion.')
protocol=('One stream on an Intel Xeon Platinum 8573C. The caller is pinned to CPU 0; the two-worker case pins its learning thread to CPU 1, '
          'a separate reported core. Seed 7, width 64, periodic mode, LR 0.04 and local LR 0.04/3. Each policy receives its own 1,024-byte warmup. '
          'Results are medians of three rotated-order runs, except the 4,096-byte continuation (one run per variant). '
          'Full logit export and API quiescence are timed; compilation, warmup, evaluation, IO and generation are excluded. '
          'Timing suites run without concurrent validation jobs. Stage 5 and Stage 6 use different update schedules; the three Stage 6 execution variants are bitwise identical.')
ratios=[r['medians']['two_workers']/r['medians']['serial_fused'] for r in results[:3]]
baseline_ratios=[r['medians']['two_workers']/r['medians']['stage5'] for r in results[:3]]
recommendation=('Stage 5 remains faster on all three paired workloads; retain it for current throughput. ' if max(baseline_ratios)<1 else 'Choose the profile using the workload-specific measurements. ')
comparison=(recommendation+f'Two-worker Stage 6 delivers {min(baseline_ratios):.2f}x to {max(baseline_ratios):.2f}x Stage 5 throughput. '
            f'Two workers deliver {min(ratios):.2f}x to {max(ratios):.2f}x the fused serial throughput across the three repeated comparisons. '
            f'The fastest measured median is {max(max(r["medians"].values()) for r in results[:3]):,.1f} bytes/s. '
            'The 15,000 bytes/s target remains unmet. Worker execution is available for measurement; the measured serial and two-worker results determine its value for each workload.')
schedule=[['During byte t','Forward and content adjoints read the committed content-bank version. The worker processes factors captured during byte t-1.'],
          ['End of byte t','Join the prior job, expose its ordered bank result, then snapshot only banks touched by the current factors and submit the next job.'],
          ['Next forward read','A committed weight row can be installed into canonical storage and multiplied by the current input using the same loads.'],
          ['API return / save','Wait for computation, retain its pending result without committing it. Snapshot and checkpoint preserve the exact next boundary.']]
contract=('The content-bank update is one byte later than immediate stage-5 mutation. Each factor retains its immutable occurrence reference, '
          'input, feature mask and post-clipping update vector; local and task objectives are already differentiated before capture. '
          'Each batch records the adjoint read version, application base version and submission sequence. Scope order and per-scope router normalization remain explicit. '
          'The bank owner writes private tiles. The stream thread owns the model, live records and committed-version changes; worker code holds no pointers into those records.')
scope=('DAG traversal, content input adjoints, synthetic-credit prediction, MinKron context learning, adapters, bottom banks and readout stay on the caller. '
       'This stage overlaps the numerical content-bank update tail with the next byte. Fixed 4/8/16-byte stage-4 epochs remain a separate schedule and cannot be combined with this profile.')
fusion=('The fused kernel forms and applies one update row without materializing a full dense gradient matrix. A second fused path uses the same vector loads '
        'to install a committed row and calculate its content forward product. The unfused control uses the same dense arithmetic and scope order, with separate '
        'materialization and installation. Per-job tile reservation avoids repeated vector growth. Sparse selected-bank snapshots are used instead of a full model copy.')
checkpoint=('Checkpoint v42 persists the committed version and the completed-but-uncommitted sparse bank result. Save/load, snapshots and irregular chunks preserve it; '
            'worker count, CPU affinity and fusion are execution choices. Document drain, reset and finish_pipeline() are explicit commit boundaries. '
            'Evaluation leaves the pending result parked and fitted parameters frozen. Existing v40/v41 policies are preserved; the ordinary default remains the full-ancestry reference.')
verification=(f'{counts["stage6_full_tests.txt"]} regressions passed, followed by {counts["stage6_pipeline_tests.txt"]} final pipeline checks at 1M '
              f'and {counts["stage6_5m_tests.txt"]} at 5M. Tests cover exact serial/unfused/threaded parity, one-byte and irregular chunks, pending saves, snapshots, '
              'worker changes, native training-mode changes, causal evaluation, corrupted-state rejection, all-region task coverage, frozen banks and document boundaries. '
              'An 8,192-byte continuation crossed record consolidation and retained exact worker/restart parity. ASan/UBSan passed the extended native lifecycle; leak scanning was disabled.')
compatibility=('A real stage-5 v41 checkpoint from '+BASE+' resumed for 1,024 targets with bitwise-identical logits, fitted parameters and full-state fingerprint. '
               'Stage-6 execution controls also matched full-state fingerprints in every matched throughput comparison.')
prof=profile['profile'];pr=[]
for name,v in sorted(prof['categories'].items(),key=lambda item:-item[1]['percent']):
 if v['percent']>=1:pr.append([name.replace('_',' '),f'{v["percent"]:.1f}%'])
profile_text=(f'The instrumented serial pipeline samples {prof["sampled_bytes"]} of {prof["total_bytes"]:,} FineWeb continuation bytes after 1,024 warmup bytes. '
              'Categories are exclusive on the caller. The pipeline-boundary category includes serial bank computation, version preparation and commits. '
              'The instrumented run matches the uninstrumented logits and complete state; these percentages are attribution, not throughput claims.')
mem=[]
for (label,_),result in zip(CASES[:3],results[:3]):
 r=next(x for x in result['runs'] if x['config']['name']=='two_workers');met=r['pipeline_metrics']
 mem.append([label.split(' /')[0],f'{met["peak_queued_bytes"]/1048576:.2f}',f'{r["pending_state_bytes"]/1048576:.2f}',f'{r["peak_rss_kib"]/1024:.0f}'])
mem_text=('Queued payload counts owned factors and bank tiles for the largest submitted job; it excludes allocator overhead and the separate visible version. '
          'Pending state is the serialized result at the run end. Peak RSS is the whole process, including evaluation and diagnostic copies, so it is not a training-only resident measurement. '
          'Router-only jobs currently carry a full bank tile; compacting those results and reusing job storage are concrete follow-up opportunities.')
remaining=[['1','Reduce version preparation','Compact router-only versions, reuse queue/tile storage and measure copy/commit cost.'],
           ['2','Reduce caller-side learning work','Profile and reduce reverse traversal, local adjoints and record maintenance; retest worker overlap after each reduction.'],
           ['3','Qualify learning at scale','Longer paired 1M/5M runs, then 100M; tune learning rates, frontier size, replay budget and event rates with held-out loss and memory.'],
           ['4','Recheck the 15k budget','Track single-stream bytes/s and loss per second; keep independent-stream aggregate throughput separate.']]
repro=('Use src/measure_bank_pipeline.py --suite paired (or sweep), with --corpus/--validation for FineWeb and SHOEBOX_EXPERTS=64 for 5M. '
       'Run tests/test_bank_pipeline.py, tests/run_completion_sanitizer.py, src/check_pipeline_long.py, src/check_pipeline_compatibility.py and '
       'src/profile_bank_pipeline.py. Raw results are results/stage6_*. The code bundle contains the patch, source and evidence; no datasets, models or compiled binaries.')
OUT.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(10.7,3.4),dpi=180)
for policy,label,color in [('event','Stage 5','#496b8b'),('pipeline','Stage 6 serial fused','#c26935')]:
 r=quality[policy+'_1'];points=r['curve']
 for ax,key in zip(axes,('targets','seconds')):ax.plot([v[key] for v in points],[v['train_nll'] for v in points],marker='o',markersize=3,label=label,color=color)
for ax in axes:ax.set_ylabel('Training NLL / 512-byte block');ax.grid(alpha=.2);ax.spines[['right','top']].set_visible(False)
axes[0].set_xlabel('Continuation bytes after 1,024 warmup');axes[1].set_xlabel('Cumulative training seconds');axes[1].legend(fontsize=8,frameon=False)
fig.suptitle('Stage 6 learning comparison - 1M FineWeb, matched learning rates',fontsize=11);fig.tight_layout()
figure=OUT/'SHoEbox_Stage6_Loss_Bytes_and_Time.png';fig.savefig(figure);plt.close(fig)
def mdtable(head,body):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(row)+' |' for row in body])
heads=['Workload / continuation bytes','Stage 5 B/s','Serial unfused','Serial fused','Two workers']
md=f'# SHoEbox: structural optimization stage 6\n\n13 November 2026 (America/Denver) | Baseline: `{BASE}`\n\n'+summary
md+='\n\n## Matched throughput\n\n'+mdtable(heads,rows)+'\n\n'+protocol+'\n\n'+comparison
md+='\n\n## Versioned schedule\n\n'+mdtable(['Boundary','Behavior'],schedule)+'\n\n'+contract+'\n\n'+scope+'\n\n### Fusion\n\n'+fusion+'\n\n### Persistence\n\n'+checkpoint
md+='\n\n## Learning comparison\n\nEach policy trains the same 3,072-byte FineWeb prefix, then evaluates 1,024 separate validation bytes with cold frozen episodic memory. Main/local rates both scale by 0.5, 1 or 2; the directional predictor keeps its fixed rate. Each cell is one seed/run.\n\n'+mdtable(['Policy','Held-out NLL / 0.5x','1x','2x'],qrows)+'\n\n![Loss by bytes and time](SHoEbox_Stage6_Loss_Bytes_and_Time.png)'
for policy,label in [('event','Stage 5'),('pipeline','Stage 6')]:
 g=quality[policy+'_1']['generation'];md+='\n\n'+label+' generation (The river prompt; seed 17, temperature 0.8, 64 ASCII/newline bytes):\n\n````text\n'+g['prompt']+g['text']+'\n````'
md+='\n\n## Validation\n\n'+verification+'\n\n'+compatibility+'\n\n## Cost attribution\n\n'+profile_text+'\n\n'+mdtable(['Exclusive category','Sampled time'],pr)
md+='\n\n## Memory\n\n'+mdtable(['Workload','Peak queued payload MiB','Pending state MiB','Process peak RSS MiB'],mem)+'\n\n'+mem_text
md+='\n\n## Reproduction\n\n'+repro+'\n\n## Remaining updates\n\nThe original six implementation stages now have implementations. The remaining work follows from measurement:\n\n'+mdtable(['Order','Update','Work remaining'],remaining)+'\n\n'+comparison+'\n'
(OUT/(STEM+'.md')).write_text(md)
for folder,regular,bold in [('/usr/share/fonts/truetype/dejavu','DejaVuSans.ttf','DejaVuSans-Bold.ttf')]:
 pdfmetrics.registerFont(TTFont('ReportSans',str(Path(folder)/regular)));pdfmetrics.registerFont(TTFont('ReportSansBold',str(Path(folder)/bold)))
styles=getSampleStyleSheet()
for name,size,leading,bold in [('TitleX',22,27,True),('HeadX',12,16,True),('BodyX',9,12.5,False),('SmallX',8,11,False),('CellX',7.5,10,False)]:
 styles.add(ParagraphStyle(name=name,fontName='ReportSansBold' if bold else 'ReportSans',fontSize=size,leading=leading,spaceAfter=0 if name=='CellX' else 8,textColor=colors.HexColor('#16354b')))
def P(v,style='BodyX'):return Paragraph(escape(v).replace('\n','<br/>'),styles[style])
def T(head,body,widths):
 t=Table([[P(v,'CellX') for v in row] for row in [head,*body]],colWidths=widths,repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5eef3')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#c8d8e0'))]));return t
story=[P('SHoEbox: versioned bank learning','TitleX'),P('STRUCTURAL OPTIMIZATION STAGE 6 | 13 NOVEMBER 2026','SmallX'),P(summary),P('Matched throughput','HeadX'),T(heads,rows,[166,84,88,87,87]),Spacer(1,10),P(protocol,'SmallX'),P(comparison),P('Baseline: '+BASE,'SmallX'),PageBreak(),P('Schedule and ownership','TitleX'),T(['Boundary','Behavior'],schedule,[105,407]),Spacer(1,10),P(contract),P(scope),P('Fused bank operations','HeadX'),P(fusion),P('Checkpoints and boundaries','HeadX'),P(checkpoint),PageBreak(),P('Learning and correctness','TitleX'),P('3,072 training bytes; 1,024 held-out FineWeb bytes. Seed 7. Main/local rates scale together; the credit predictor rate stays fixed. One run per cell.','SmallX'),T(['Policy','Held-out / 0.5x','1x','2x'],qrows,[152,120,120,120]),Spacer(1,8),Image(str(figure),width=512,height=163),P('Validation','HeadX'),P(verification),P(compatibility,'SmallX')]
for policy,label in [('event','Stage 5'),('pipeline','Stage 6')]:
 g=quality[policy+'_1']['generation'];story.extend([P(label+' sample (seed 17, temperature 0.8)','SmallX'),P(g['prompt']+g['text'],'SmallX')])
story.extend([PageBreak(),P('Costs and remaining updates','TitleX'),P(profile_text,'SmallX'),T(['Exclusive category','Sampled time'],pr,[390,122]),Spacer(1,8),P('Memory accounting','HeadX'),T(['Workload','Queue MiB','Pending MiB','Peak RSS MiB'],mem,[179,108,108,117]),Spacer(1,6),P(mem_text,'SmallX'),PageBreak(),P('Remaining updates','TitleX'),T(['Order','Update','Work remaining'],remaining,[35,136,341]),Spacer(1,6),P(repro,'SmallX')])
def footer(c,d):c.setFont('ReportSans',8);c.setFillColor(colors.HexColor('#526575'));c.drawString(50,27,'SHoEbox | Structural optimization stage 6');c.drawRightString(562,27,str(d.page))
SimpleDocTemplate(str(OUT/(STEM+'.pdf')),pagesize=(612,792),leftMargin=50,rightMargin=50,topMargin=42,bottomMargin=45,title='SHoEbox structural optimization stage 6').build(story,onFirstPage=footer,onLaterPages=footer)
print(json.dumps(dict(throughput=rows,quality=qrows,worker_ratios=ratios,test_counts=counts),indent=2))
