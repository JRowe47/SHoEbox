"""Create the compact-versioning Markdown and PDF report from measured results."""
from pathlib import Path
import json,re
from html import escape
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak,Preformatted
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
BASE='fbf7c816446289bcd987a85028d7cab59389b7ed';STEM='SHoEbox_Compact_Versioning'
labels={'1m_repeated':'1M repeated / 2,048','1m_prose':'1M repository prose / 2,048','5m_repeated':'5M repeated / 2,048','1m_long':'1M repeated / 8,192'}
sets={k:json.loads((ROOT/f'results/compact_{k}_paired.json').read_text()) for k in labels}
rows=[];mem=[];quality=[];maxdrift=0.;ratios=[]
for key,data in sets.items():
 med=data['medians'];old=med['stage6_serial'];new=med['compact_serial'];ratios.append(new/old)
 rows.append([labels[key],f'{old:,.1f}',f'{new:,.1f}',f'{100*(new/old-1):+.1f}%',f'{med["compact_worker"]:,.1f}'])
 a=next(r for r in data['runs'] if r['config']['name']=='stage6_serial');b=next(r for r in data['runs'] if r['config']['name']=='compact_serial')
 mem.append([labels[key],f'{a["pending_state_bytes"]/1024:,.1f}',f'{b["pending_state_bytes"]/1024:,.1f}',f'{100*(1-b["pending_state_bytes"]/a["pending_state_bytes"]):.1f}%'])
 quality.append([labels[key],f'{a["heldout_nll"]:.6f}',f'{b["heldout_nll"]:.6f}'])
 for c in data['comparisons']:maxdrift=max(maxdrift,c['vs_stage6']['logits']['max_abs'])
compat=json.loads((ROOT/'results/compact_v42_compatibility.json').read_text())
def count(name):
 text=(ROOT/'results'/name).read_text();assert text.rstrip().endswith('OK'),name
 return int(re.findall(r'Ran (\d+) tests',text)[-1])
counts=[count(f) for f in ('compact_audit_tests.txt','compact_production_tests.txt','compact_5m_tests.txt')]
summary=('Implemented compact bank versions and removed inline observation work from production completion builds. '
         'The pipeline reuses two job arenas, shares immutable input copies by index, follows indexed per-bank scope lists, '
         'and avoids unnecessary weight installation and router-cache refreshes. All 18 regions and the one-byte-delayed learning schedule remain.')
performance=f'Serial single-stream throughput improved by {100*(min(ratios)-1):.1f}% to {100*(max(ratios)-1):.1f}% over the matched Stage-6 baseline across these workloads.'
protocol=('Intel Xeon Platinum 8573C; caller pinned to CPU 0, learning worker to CPU 1; width 64, 12 or 64 experts per region; seed 7. '
          'Each run uses 1,024 warmup bytes, LR 0.04 and local LR 0.04/3. Timings include full logit export and API quiescence in 512-byte chunks. '
          'Compilation, warmup, snapshots, inspection, evaluation and file IO are excluded. Short workloads use three repetitions in rotated order; '
          'the 8,192-byte continuation uses one run per variant. No validation process overlaps timing. Repository prose uses the baseline README '
          'with a separate baseline documentation file for held-out evaluation; this is not the earlier FineWeb benchmark.')
mechanisms=[
 ['Router-only results','528 bytes per serialized bank instead of 16,904 bytes (96.9% smaller). A dense weight tile exists only for banks with weight factors.'],
 ['Reusable storage','Two job objects retain scope, factor, input, index and tile capacity. No new batch allocation or shared-pointer reference counting per byte.'],
 ['Shared input snapshots','Each immutable occurrence input is copied once per job. Factors retain an input index; the worker owns all referenced storage.'],
 ['Indexed scope traversal','Each bank follows only its own ordered scope chain. The worker no longer scans every scope for every bank.'],
 ['Cheaper preparation/commit','Skip unused dense-gradient clearing, avoid zeroing arrays that will be overwritten, install only necessary visible weights, and refresh changed routers only.'],
 ['External observation','No per-job timers, observer branches, covariance collection, norm/histogram accounting or diagnostic-only decoder walk in the production completion loop.'],
]
inspection=('capture_memory() copies parameters and continuation buffers at a quiescent API boundary; compare_memory() compares those detached arrays. '
            'read_memory_index() scans current retained publications, constituents, classes and unresolved records. inspect_pipeline() reports active sparse bank entries, '
            'input reuse and reserved buffer capacity. Inspection does not advance the committed version or train the model. Current inventories do not reconstruct '
            'historical event counts; old counter fields are retained only for checkpoint compatibility. Execution counts, routing usage, credit reuse, clipping and identity guards remain operational.')
validation=(f'{counts[0]} regressions passed in the explicit historical observation build; {counts[1]} production pipeline tests passed at 1M and {counts[2]} at 5M. '
            'Both the full native lifecycle and production compact pipeline passed ASan/UBSan with leak scanning disabled. '
            'An 8,192-byte stream retained bitwise worker/chunk/restart equality across consolidation. '
            'Tests also cover router-only payloads, two-buffer reuse, detached inspection, no inspection side effects, malformed import rejection, '
            'all upper regions receiving task learning, frozen banks, document drains, evaluation causality and worker changes.')
rounding=(f'The compact storage/queue changes with historical diagnostics enabled preserve bitwise Stage-6 continuation, including operational state. '
          f'Removing observation code changes compiler rounding: the largest baseline-versus-production logit difference across the measured continuations is {maxdrift:.9g}. '
          'Production serial and threaded execution remain bitwise identical. Held-out scores are reported below, without a new learning-rule or hyperparameter change.')
checkpoint=('New pipeline saves use checkpoint v43 / extension 2. The loader accepts real v42 / extension-1 pending jobs; initial imported state and expanded pending banks match bitwise. '
            'A resumed 1,024-byte audit continuation is bitwise identical to Stage 6. Production continuation differs by at most '
            f'{compat["runs"][1]["continuation_logit_max_abs"]:.9g} in logits and {compat["runs"][1]["continuation_parameter_max_abs"]:.9g} in base parameters. '
            'Save, snapshots and ordinary chunk ends park the completed pending result. finish_pipeline(), document drain and reset remain explicit commit boundaries.')
memory_note=('These are pending-result sizes at the measured endpoint, not peak RSS. Reuse intentionally retains high-water buffer capacity; '
             'inspect_pipeline() reports reserved_bytes separately from active_payload_bytes. No per-byte peak/copy counter is maintained in production. '
             'The captured scopes and input arenas are transient and are not serialized.')
remaining=[['1','Reduce caller-side learning work','Optimize reverse traversal and local adjoints; retest the benefit of the learning worker after each change.'],
           ['2','Reduce record maintenance','Use the live memory indexes to identify expensive collection and retention paths while preserving eligible producers.'],
           ['3','Qualify scaling and the 15k target','Longer paired 1M/5M runs, then 100M; compare held-out loss per second, memory and single-stream throughput.']]
usage="""from shoebox_reference import fresh_model, capture_memory, compare_memory
m = fresh_model(seed=7, profile='pipeline')
before = capture_memory(m, include_stream=False)
m.run(inputs, targets, mode='periodic', lr=.04, lp=.04/3)
after = capture_memory(m, include_stream=False)
changes = compare_memory(before, after)
queue = m.inspect_pipeline()"""
repro=('Run tests/test_compact_pipeline.py and tests/test_bank_pipeline.py in the production build. Run the complete historical coverage suite with '
       'SHOEBOX_INLINE_DIAGNOSTICS=1. Run tests/run_compact_sanitizer.py and tests/run_completion_sanitizer.py. '
       'src/measure_compact_pipeline.py compares a supplied Stage-6 runtime; src/check_compact_long.py and src/check_compact_compatibility.py exercise restart and v42 continuation. '
       'src/profile_bank_pipeline.py now performs boundary inspection rather than source-level timer injection. Raw measurements are results/compact_*. '
       'The code bundle contains source, patch, tests and compact evidence; no models, datasets or binaries.')
def table(head,body):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(row)+' |' for row in body])
heads=['Workload / bytes','Stage 6 B/s','Compact serial B/s','Gain','Compact 2 workers B/s']
md='# SHoEbox: compact versioning and external inspection\n\n13 November 2026 (America/Denver) | Baseline: `'+BASE+'`\n\n'+summary+'\n\n'+performance
md+='\n\n## Throughput\n\n'+table(heads,rows)+'\n\n'+protocol
md+='\n\n## Implementation\n\n'+table(['Change','Result'],mechanisms)+'\n\n## Inspection\n\n'+inspection+'\n\n```python\n'+usage+'\n```'
md+='\n\n## Correctness and continuation\n\n'+validation+'\n\n'+rounding+'\n\n'+checkpoint
md+='\n\n'+table(['Workload / bytes','Stage 6 held-out NLL','Compact held-out NLL'],quality)
md+='\n\n## Pending result size\n\n'+table(['Workload / bytes','Stage 6 KiB','Compact KiB','Reduction'],mem)+'\n\n'+memory_note
md+='\n\n## Reproduction\n\n'+repro+'\n\n## Remaining updates\n\n'+table(['Order','Update','Work'],remaining)+'\n'
(OUT/(STEM+'.md')).write_text(md)
for name,file in [('Sans','DejaVuSans.ttf'),('SansBold','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/truetype/dejavu/'+file))
styles=getSampleStyleSheet()
for name,size,leading,bold in [('TitleC',21,26,True),('HeadC',11,15,True),('BodyC',9,12.5,False),('SmallC',8,11,False),('CellC',7.5,10.5,False)]:
 styles.add(ParagraphStyle(name=name,fontName='SansBold' if bold else 'Sans',fontSize=size,leading=leading,spaceAfter=7 if name!='CellC' else 0,textColor=colors.HexColor('#173449')))
def P(text,style='BodyC'):return Paragraph(escape(text),styles[style])
def T(head,body,widths):
 t=Table([[P(v,'CellC') for v in row] for row in [head,*body]],colWidths=widths,repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3eef4')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#c4d3df'))]));return t
story=[P('SHoEbox: compact versioning','TitleC'),P('EXTERNAL INSPECTION | 13 NOVEMBER 2026','SmallC'),P(summary),P(performance),P('Matched single-stream throughput','HeadC'),T(heads,rows,[160,85,92,60,115]),Spacer(1,8),P(protocol,'SmallC'),P('What changed','HeadC'),T(['Change','Implementation'],mechanisms[:5],[115,397]),PageBreak(),P('Correctness and memory','TitleC'),P(validation),P(rounding),P('Checkpoint compatibility','HeadC'),P(checkpoint),T(['Workload / bytes','Stage 6 NLL','Compact NLL'],quality,[262,125,125]),Spacer(1,9),P('Compact pending results','HeadC'),T(['Workload / bytes','Stage 6 KiB','Compact KiB','Reduction'],mem,[215,99,99,99]),Spacer(1,8),P(memory_note,'SmallC'),PageBreak(),P('Inspection outside training','TitleC'),P(mechanisms[-1][1]),P(inspection),Preformatted(usage,ParagraphStyle(name='CodeC',fontName='Courier',fontSize=7.3,leading=11)),Spacer(1,12),P('Reproduction','HeadC'),P(repro,'SmallC'),P('Remaining updates','HeadC'),T(['Order','Update','Work'],remaining,[42,137,333]),Spacer(1,8),P('Baseline: '+BASE,'SmallC')]
def footer(c,d):c.setFont('Sans',7.5);c.setFillColor(colors.HexColor('#536a7c'));c.drawString(50,26,'SHoEbox | Compact versioning');c.drawRightString(562,26,str(d.page))
SimpleDocTemplate(str(OUT/(STEM+'.pdf')),pagesize=(612,792),leftMargin=50,rightMargin=50,topMargin=40,bottomMargin=45,title='SHoEbox compact versioning and external inspection').build(story,onFirstPage=footer,onLaterPages=footer)
(ROOT/'results/compact_report_summary.json').write_text(json.dumps(dict(throughput=rows,pending_memory=mem,heldout=quality,max_logit_drift=maxdrift,test_counts=counts),indent=2)+'\n')
print(json.dumps(dict(throughput=rows,pending_memory=mem,max_logit_drift=maxdrift,test_counts=counts),indent=2))
