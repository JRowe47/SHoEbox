"""Render the stage 2 report from the checked, paired measurements."""
from pathlib import Path
import json
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables'
BASE='86a4204862425182c3bd30cb0b14aaaa14060441'
CASES=[('1M repeated training, 1,024 targets','stage2_1m_repeated.json'),
       ('1M natural training, 1,024 targets','stage2_1m_natural.json'),
       ('1M frozen evaluation, 4,096 targets','stage2_1m_eval.json'),
       ('5M repeated training, 1,024 targets','stage2_5m_repeated.json'),
       ('1M repeated training, 4,096 targets','stage2_1m_long.json')]
rows=[];results=[]
for label,name in CASES:
    result=json.loads((ROOT/'results'/name).read_text());results.append(result)
    assert result['bitwise_logits_and_fitted_state']
    before=result['medians']['reference'];after=result['medians']['optimized_1']
    rows.append([label,f'{before:,.1f}',f'{after:,.1f}',f'{100*(after/before-1):+.1f}%',str(result['repeats'])])
first=results[0]['runs'];old=first['reference'][0]['storage'];new=first['optimized_1'][0]['storage']
shared=first['optimized_1'][0]['publication_storage']
long=results[-1]['runs'];oldlong=long['reference'][0]['storage'];newlong=long['optimized_1'][0]['storage']
mib=lambda x:f'{x/2**20:.2f} MiB'
reduction=lambda a,b:f'{100*(1-b/a):.1f}%'
memory=[['Copied event / handle',str(shared['legacy_event_bytes'])+' B',str(shared['event_handle_bytes'])+' B',reduction(shared['legacy_event_bytes'],shared['event_handle_bytes'])],
        ['Input record, before pending arrays',str(old['input_bytes'])+' B',str(new['input_bytes'])+' B',reduction(old['input_bytes'],new['input_bytes'])],
        ['Upper archive record, before pending arrays',str(old['upper_record_bytes'])+' B',str(new['upper_record_bytes'])+' B',reduction(old['upper_record_bytes'],new['upper_record_bytes'])],
        ['Archive after 2,048 total targets',mib(old['archive_payload_bytes']),mib(new['archive_payload_bytes']),reduction(old['archive_payload_bytes'],new['archive_payload_bytes'])],
        ['Archive after 5,120 total targets',mib(oldlong['archive_payload_bytes']),mib(newlong['archive_payload_bytes']),reduction(oldlong['archive_payload_bytes'],newlong['archive_payload_bytes'])]]
count=sum(len(v) for r in results for v in r['runs'].values())
full=all(r['physical_state_identical'] for r in results)
log=(ROOT/'results/stage2_regressions.txt').read_text();tests=int(re.search(r'Ran (\d+) tests',log)[1]);assert log.rstrip().endswith('OK')
assert (ROOT/'results/stage2_storage_tests_5m.txt').read_text().rstrip().endswith('OK')
assert 'ASan/UBSan passed' in (ROOT/'results/stage2_sanitizer.txt').read_text()
for name in ['stage2_train_compatibility.json','stage2_eval_compatibility.json']:
    assert json.loads((ROOT/'results'/name).read_text())['bitwise_continuation']
remaining=[['3','Aggregate clipped update factors','Combine final per-request factors for the same immutable input and bank within each update scope, before forming dense outer products. Preserve clipping, masks, rates and mutation boundaries.'],
           ['4','Share reverse traversals','Compare a union reverse pass inside each fixed-weight credit scope; then test short, explicitly versioned update epochs.'],
           ['5','Small native MinKron context and local event learning','Keep exact content/occurrence identities and paired hippocampal episodes. Add compact contextual memory, an active learning frontier and directional delayed credit with targeted replay.'],
           ['6','Versioned forward/learning workers','After reducing work, overlap previous-event learning and current-event forward computation with explicit weight versions; compare against the same serial schedule.']]
summary=f'Stage 2 is implemented in the default reference runtime: shared publication payloads, cached role contributions and sparse pending gradients. Archived record payload falls {reduction(old["archive_payload_bytes"],new["archive_payload_bytes"])} in the 1M repeated-text comparison. All {count} timed continuations preserve logits, loss, fitted parameters, routing and recurrent/content/memory state bit-for-bit.'
implementation=[
    'Publication copies retain occurrence identity, roles, freshness, primary clocks and neural origin alongside shared content/context handles. Histories, pending frames, consumers and relays reuse the payload. Explicit vector edits detach it before writing.',
    'The bounded role cache reuses permuted structural values, semantic residuals and context values. It keys on immutable payload handles, including changed learned codes and intervention copies. Receiver gains, signs, weights and summation order remain at their original arithmetic boundaries.',
    'Neural and retained-input gradient arrays allocate only for pending work and release when empty. The collector obtains pending roots from the existing indices instead of scanning every archived gradient. All live derivative records and depth policies remain available.',
    'Checkpoint v39, composition ABI 10 (including the ABI 9 import), hot lace state and neural archive ABI 1 retain their wire formats through explicit conversion. Shared handles and role caches are runtime-only. Historical kernels keep their original storage.']
protocol='Matched actual stage-1 checkpoint after 1,024 warmup targets; width 64, seed 7, all reference features, periodic training at LR 0.04 and local LR 0.04/3. One pinned CPU on an Intel Xeon Platinum 8573C; implementations run serially in alternating order after the regression/sanitizer work finishes. Timings include full logit export and exclude compilation, loading, saving and hashing. The table reports median bytes/s and pairs per row. The natural-text sample uses the same available FineWeb stream as the earlier review; its data hash is recorded.'
eval_runs=results[2]['runs']
spread=lambda name:(min(r['bytes_per_second'] for r in eval_runs[name]),max(r['bytes_per_second'] for r in eval_runs[name]))
eval_old=spread('reference');eval_new=spread('optimized_1')
timing_note=f'Small median changes overlap the observed timing variation. Frozen evaluation ranged from {eval_old[0]:,.0f} to {eval_old[1]:,.0f} B/s before and {eval_new[0]:,.0f} to {eval_new[1]:,.0f} B/s after; all samples, including the slow outlier, remain in the results. The 4,096-target training row is one paired observation.'
verification=f'{tests} regression tests passed, including the three new publication/sparse-storage checks. Six storage tests also passed with 64 experts per region (5M). ASan/UBSan passed training, drain, reset, sparse ownership, cache detach, hot/archive round trips, and a live semantic DAG beyond 16,384 nodes. Leak scanning was disabled because the host blocks LeakSanitizer. Old-to-new-to-old checkpoint continuation passed both after training and after frozen evaluation followed by training.'
if full:verification+=' The complete serialized state fingerprint also matches in all timed continuations.'
else:verification+=' See the raw measurements for serialized fingerprints; all compared learning and output fields match.'
workspace=f'The 2,048-target endpoint holds {shared["neural_gradient_blocks"]} neural and {shared["input_gradient_blocks"]} retained-input gradient blocks ({shared["gradient_payload_bytes"]/1024:.1f} KiB total). It records {shared["content_cache_hits"]:,} content-cache hits and {shared["context_cache_hits"]:,} context-cache hits during the continuation. Archive figures include allocated pending gradient payloads and bottom temporal records; they exclude map/allocator overhead. Handle size measures copy volume, not the entire shared publication allocation.'
repro='Use src/compare_optimized_runtime.py against an unchanged checkout of baseline 86a4204. Results are in results/stage2_*.json; validation logs are stored beside them. Use src/check_storage_compatibility.py with and without --bridge-mode eval. Rebuild this report with python experiments/identity_semantics/src/report_publication_storage.py. No trained checkpoint, dataset or compiled binary is included in the change bundle.'
def mdtable(head,values):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(row)+' |' for row in values])
md='# SHoEbox: structural optimization stage 2\n\n13 November 2026 (America/Denver) | Baseline: `'+BASE+'`\n\n'+summary+'\n\n## Implementation\n\n'+'\n\n'.join('- '+x for x in implementation)+'\n\n## Memory and copying\n\n'+mdtable(['Measure','Stage 1','Stage 2','Reduction'],memory)+'\n\n'+workspace+'\n\n## Matched throughput\n\n'+mdtable(['Workload','Before B/s','After B/s','Change','Pairs'],rows)+'\n\n'+protocol+'\n\n'+timing_note+'\n\nThis stage changes storage and reuse. The full ancestry learning algorithm remains in place; the larger reductions in repeated gradient work are the next updates.\n\n## Verification\n\n'+verification+'\n\n## Reproduction\n\n'+repro+'\n\n## Remaining updates, in review order\n\n'+mdtable(['Stage','Update','Work remaining'],remaining)+'\n'
OUT.mkdir(exist_ok=True);stem='SHoEbox_Structural_Optimization_Stage2';(OUT/(stem+'.md')).write_text(md)
# Embed real font files: unembedded base fonts render inconsistently on some hosts.
font_candidates=[('/usr/share/fonts/truetype/open-sans','OpenSans-Regular.ttf','OpenSans-Bold.ttf'),
                 ('/usr/share/fonts/truetype/dejavu','DejaVuSans.ttf','DejaVuSans-Bold.ttf')]
for folder,regular,bold in font_candidates:
    if (Path(folder)/regular).is_file() and (Path(folder)/bold).is_file():
        pdfmetrics.registerFont(TTFont('ReportSans',str(Path(folder)/regular)))
        pdfmetrics.registerFont(TTFont('ReportSansBold',str(Path(folder)/bold)))
        break
else:
    raise RuntimeError('Install Open Sans or DejaVu Sans TrueType fonts to render the report')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleX',fontName='Helvetica-Bold',fontSize=23,leading=27,spaceAfter=12,textColor=colors.HexColor('#14354f')))
styles.add(ParagraphStyle(name='BodyX',fontName='Helvetica',fontSize=9.7,leading=14,spaceAfter=9,textColor=colors.HexColor('#20354a')))
styles.add(ParagraphStyle(name='HeadX',fontName='Helvetica-Bold',fontSize=13,leading=17,spaceBefore=7,spaceAfter=8,textColor=colors.HexColor('#087d85')))
styles.add(ParagraphStyle(name='CellX',fontName='Helvetica',fontSize=8.3,leading=11))
styles.add(ParagraphStyle(name='SmallX',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=10,textColor=colors.HexColor('#526575')))
for name in ['TitleX','BodyX','HeadX','CellX','SmallX']:
    styles[name].fontName='ReportSansBold' if name in ['TitleX','HeadX'] else 'ReportSans'
def P(text,style='BodyX'):return Paragraph(escape(text),styles[style])
def T(head,body,widths):
    t=Table([[P(v,'CellX') for v in row] for row in [head,*body]],colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eff3')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#7c9ca8')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d2dfe4'))]))
    return t
story=[P('SHoEbox: shared publications','TitleX'),P('STRUCTURAL OPTIMIZATION STAGE 2  |  13 NOVEMBER 2026','SmallX'),P(summary),P('Memory and copying','HeadX'),T(['Measure','Stage 1','Stage 2','Reduction'],memory,[235,88,88,101]),Spacer(1,10),P(workspace),P('Baseline: '+BASE,'SmallX'),PageBreak(),P('Implementation and validation','TitleX')]
for i,x in enumerate(implementation,1):story.append(P(str(i)+'. '+x))
story.extend([P('Verification','HeadX'),P(verification),P('Reproduction','HeadX'),P(repro),PageBreak(),P('Throughput and next updates','TitleX'),T(['Workload','Before B/s','After B/s','Change','Pairs'],rows,[224,85,85,70,48]),Spacer(1,10),P(protocol,'SmallX'),P(timing_note,'SmallX'),P('Remaining updates','HeadX'),T(['Stage','Update','Work remaining'],remaining,[37,138,337])])
def footer(canvas,doc):
    canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor('#526575'))
    canvas.drawString(50,27,'SHoEbox | Structural optimization stage 2');canvas.drawRightString(562,27,str(doc.page))
SimpleDocTemplate(str(OUT/(stem+'.pdf')),pagesize=(612,792),leftMargin=50,rightMargin=50,topMargin=42,bottomMargin=45,title='SHoEbox structural optimization stage 2').build(story,onFirstPage=footer,onLaterPages=footer)
print(json.dumps({'tests':tests,'timed_continuations':count,'full_state_bitwise':full,'memory':memory,'throughput':rows},indent=2))
