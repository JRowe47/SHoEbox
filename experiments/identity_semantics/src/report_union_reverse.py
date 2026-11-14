"""Build the stage-4 report and loss/time figure from recorded measurements."""
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
BASE='961a18320f45ec60359f37d818cd928d202ea01a'
STEM='SHoEbox_Structural_Optimization_Stage4'
CASES=[('1M repeated, 1,024 bytes','stage4_1m_repeated.json'),
       ('1M natural, 1,024 bytes','stage4_1m_natural.json'),
       ('5M repeated, 1,024 bytes','stage4_5m_repeated.json'),
       ('1M repeated, 4,096 bytes','stage4_1m_long.json')]
results=[json.loads((RESULTS/name).read_text()) for _,name in CASES]
for r in results:assert r['deterministic'] and all(v['credit_integrity'] for rows in r['runs'].values() for v in rows)
sweep=json.loads((RESULTS/'stage4_epoch_sweep.json').read_text());assert len(sweep['runs'])==13
assert all(r['credit_integrity'] for r in sweep['runs'])
compat=json.loads((RESULTS/'stage4_compatibility.json').read_text())
assert compat['state_fingerprint_equal'] and compat['nll_equal'] and all(compat['arrays_bitwise'].values())
log=(RESULTS/'stage4_regressions.txt').read_text();assert log.rstrip().endswith('OK')
tests=int(re.findall(r'Ran (\d+) tests',log)[-1])
assert (RESULTS/'stage4_union_5m_tests.txt').read_text().rstrip().endswith('OK')
assert 'ASan/UBSan passed' in (RESULTS/'stage4_completion_sanitizer.txt').read_text()
throughput=[];numeric=[]
for (label,_),r in zip(CASES,results):
    old,new=r['medians']['stage3'],r['medians']['stage4'];c=r['comparison']
    throughput.append([label,f'{old:,.1f}',f'{new:,.1f}',f'{100*(new/old-1):+.1f}%',str(r['repeats'])])
    numeric.append([label,f'{c["logits"]["max_abs"]:.3g}',f'{c["nll_delta"]:+.3g}',
                    str(c['route_trace']['expert_disagreements']),str(c['top_byte_disagreements'])])
root=sweep['runs'][0]
def find(epoch,scale=1.):
    return next(r for r in sweep['runs'] if r['policy']==dict(policy='union',epoch_bytes=epoch,update_scale=scale))
selected=[root]+[find(e) for e in (1,4,8,16)]
labels=['Request-wise','Scope union','4-step epoch','8-step epoch','16-step epoch']
epoch_rows=[]
for label,r in zip(labels,selected):
    epoch_rows.append([label,f'{r["bytes_per_second"]:,.1f}',f'{r["train_nll"]:.6f}',f'{r["heldout_nll"]:.6f}',
                       f'{r["reverse_metrics"]["content_vjps"]:,}'])
scale_rows=[]
for e in (1,4,8,16):
    scale_rows.append(['Scope' if e==1 else str(e)+' steps']+[f'{find(e,k)["heldout_nll"]:.6f}' for k in (.5,1.,2.)])
removed=1-find(1)['reverse_metrics']['content_vjps']/root['reverse_metrics']['content_vjps']
removed16=1-find(16)['reverse_metrics']['content_vjps']/root['reverse_metrics']['content_vjps']
max_logit=max(r['comparison']['logits']['max_abs'] for r in results)
traced=sum(r['comparison']['route_trace']['regional_decisions'] for r in results)
route_diff=sum(r['comparison']['route_trace']['expert_disagreements'] for r in results)
mask_diff=sum(r['comparison']['route_trace']['mask_disagreements'] for r in results)
top_diff=sum(r['comparison']['top_byte_disagreements'] for r in results)
timed=sum(len(rows) for r in results for rows in r['runs'].values())
assert route_diff==mask_diff==top_diff==0
assert all(r['reverse_metrics']['clipped_coordinates']==r['reverse_metrics']['clipped_router_steps']==0 for r in sweep['runs'])
assert all(v['reverse_metrics']['clipped_coordinates']==v['reverse_metrics']['clipped_router_steps']==0 for r in results for v in r['runs']['stage4'])
summary=('Stage 4 is implemented in the default reference runtime. Compatible teaching roots now share one reverse pass over their union of constituent ancestry. '
         'All reference regions, content/context channels, directional teaching and paired episodic memory remain active. '
         'Short, explicitly versioned expert-bank epochs are implemented as comparison policies.')
protocol=(f'Matched continuations start from real stage-3 checkpoints after 1,024 warmup byte targets, with seed 7, width 64, '
          f'periodic training, LR 0.04 and local LR 0.04/3. One pinned CPU on {results[0]["cpu_model"]}; '
          'alternating serial implementations. Timings include full logit export and exclude compilation, checkpoint IO, hashing and route tracing. '
          'The natural fixture is the same FineWeb stream used in the previous review. The 4,096-byte row contains one pair.')
parity=(f'Across {traced:,} traced regional decisions, there were {route_diff} expert and {mask_diff} mask disagreements, and {top_diff} top-byte disagreements. '
        f'Maximum logit drift was {max_logit:.3g}; absolute NLL differences stayed below 5e-9. '
        'No content-coordinate or router-step clipping fired in these matched continuations or in the epoch sweep. '
        'Each implementation was deterministic across repeats; all timed continuations passed credit-integrity checks.')
sweep_protocol=('The epoch comparison restores one 1M request-wise checkpoint after 1,024 FineWeb warmup targets, then trains another 2,048 bytes. '
                'Validation uses 1,024 targets from a separate FineWeb split with cold, frozen episodic memory. '
                'Each policy/scale has one measured continuation. Sweep timings exclude full logit export, held-out evaluation and generation; '
                'they are separate from the matched table above. The update-scale sweep changes constituent credit strength only; other learning rates stay fixed.')
sweep_assessment=(f'Scope union at 1x reduced content VJPs by {100*removed:.1f}% and matched the request-wise held-out loss to 3e-9. '
                  f'The 16-step policy at 1x reduced VJPs by {100*removed16:.1f}% and reached {find(16)["bytes_per_second"]:,.0f} B/s '
                  f'({find(16)["bytes_per_second"]/root["bytes_per_second"]:.2f}x the sweep control), with held-out NLL higher by '
                  f'{find(16)["heldout_nll"]-root["heldout_nll"]:.6f}. '
                  'Scope union at 1x remains the default because it retains the existing update boundaries and magnitude. '
                  'The short sweep makes 4/8/16-step epochs available for longer quality comparisons; it does not change the default to an epoch schedule.')
implementation=[
    'Union execution: group roots by teaching objective and complete regional teaching-scale vector. Fold each signed learning rate into the incoming adjoint, normalize by the sum of absolute rates, visit unique nodes in reverse creation order, and restore the rate mass at each parameter update. Frozen recipients still transmit derivatives. Singleton groups use the original arithmetic path.',
    'Learning policy 2: content-coordinate clipping applies to the group mean, while router clipping and normalization apply to the combined nonzero step. A cancelled router group performs no normalization. Linear adjoints and unclipped update factors combine algebraically, but this is an explicit learning rule, not a promise of request-wise clipping equivalence. Policy 1 remains available as the request-wise comparator.',
    'Mutation order: scope-level semantic VJPs finish before shared expert weights can change. Existing neural updates then run, followed by the existing content-bank commit. Exact source roles, masks, gates, receiver scales and content-expert eligibility remain in use. No ancestry depth or credit budget is shortened.',
    'Epoch scope: 4/8/16-step modes freeze reads of shared expert W, bias and router banks, accumulate neural bank writes in the original arrays, and queue constituent roots until the boundary. The decoder, predictors, learned bottom-state parameters, dictionaries and recurrent state stay online. These are expert-bank epochs, not whole-model frozen minibatches.',
    'Persistence: checkpoint v40 saves the learning policy, bank version, epoch position, bank snapshot and pending roots. Queued roots retain their ancestry during collection. Save/load and snapshot copies never force an epoch boundary; evaluation uses the visible bank version. Document draining and context reset finish pending teaching. A bank snapshot uses 3.48 MiB at 1M or 18.56 MiB at 5M, plus variable pending-root storage.',
    'Compatibility and diagnostics: v39 and earlier completion checkpoints load with request-wise learning. Existing constituent ABI 10 and content/neural ABI 1 remain unchanged. Old runtimes reject the new wrapper version. Visit and semantic norm counters describe executed union work; missing-producer counters continue to identify unresolved dependencies.']
verification=(f'{tests} full-suite regressions passed, plus three targeted 5M checks for mixed policies/frozen regions, partial-epoch checkpoint continuation and visible evaluation weights. '
              'Eight focused 1M tests are included in the full-suite total. The scalar oracle verifies the unclipped union against request-wise updates and separately checks the changed clipping rule. '
              'ASan/UBSan passed with full reference training, document draining, all three epoch lengths, malformed-state rejection, queued-root ownership and live DAG growth beyond 16,384 records. '
              'Leak scanning is disabled on this host.')
compatibility=('A real stage-3 checkpoint resumed under policy 1 with bitwise-identical logits, parameters, composition state, content routing, route traces and complete state fingerprint. '
               'Same-runtime irregular chunks, partial-epoch save/load and snapshot copies also passed bitwise continuation checks. '
               f'The {timed} matched timed continuations and all 13 sweep continuations passed credit integrity. No checkpoints, datasets or compiled binaries are included in Git or the change bundle.')
remaining=[['5','Native small-head MinKron and local event learning','Implement one/two small context heads, an exact active learning frontier, directional delayed credit and targeted replay. Preserve exact occurrence IDs and paired hippocampal episodes.'],
           ['6','Persistent versioned forward/learning workers','Overlap previous-event learning with current forward work and fuse compatible bank reads/updates. Compare against the same serial update schedule.']]
assessment=('Stage 4 removes repeated derivative work while keeping the full hierarchy. Stage 5 is next and remains the main change aimed at 15,000+ bytes/s: '
            'make learning cost follow active new events rather than repeatedly revisited ancestry.')
repro=('Run src/compare_union_reverse.py against stage-3 runtime 961a183; add --corpus for natural text and SHOEBOX_EXPERTS=64 for the 5M case. '
       'Run src/sweep_union_epochs.py with separate --corpus and --validation paths, then src/check_union_compatibility.py. '
       'Raw results are results/stage4_*.json; validation logs are beside them. Rebuild this report with src/report_union_reverse.py.')
OUT.mkdir(exist_ok=True)
# Static publication figure: the five 1x policies, using block NLL and cumulative
# measured training seconds. Keep the raw four-point curves in the evidence.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(9.6,3.3),layout='constrained')
for label,r,color in zip(labels,selected,['#697785','#155f91','#319c87','#d58d28','#9364b4']):
    for ax,key in zip(axes,['targets','seconds']):
        ax.plot([v[key] for v in r['curve']],[v['train_nll'] for v in r['curve']],marker='o',ms=3,lw=1.6,label=label,color=color)
for ax in axes:ax.grid(alpha=.2);ax.set_ylabel('Training NLL (512-byte block)')
axes[0].set_xlabel('Continuation bytes');axes[1].set_xlabel('Cumulative training seconds')
axes[0].set_title('Loss by training bytes');axes[1].set_title('Loss by elapsed training time')
axes[1].legend(fontsize=7,loc='upper right')
figure=OUT/'SHoEbox_Stage4_Loss_Bytes_and_Time.png';fig.savefig(figure,dpi=180);plt.close(fig)
def mdtable(head,body):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(row)+' |' for row in body])
generations=[]
for label,r in zip(labels,selected):
    g=r['generation'];generations.append((label,g['prompt']+g['text']))
md='# SHoEbox: structural optimization stage 4\n\n13 November 2026 (America/Denver) | Baseline: `'+BASE+'`\n\n'+summary
md+='\n\n## Matched throughput\n\n'+mdtable(['Workload','Stage 3 B/s','Stage 4 B/s','Change','Pairs'],throughput)+'\n\n'+protocol
md+='\n\n## Numerical comparison\n\n'+mdtable(['Workload','Max logit drift','NLL delta','Route differences','Top-byte differences'],numeric)+'\n\n'+parity
md+='\n\n## Fixed bank-version epochs\n\n'+sweep_protocol+'\n\n'+mdtable(['Policy, 1x scale','B/s','Training NLL','Held-out NLL','Content VJPs'],epoch_rows)
md+='\n\n'+sweep_assessment+'\n\n### Reverse update-scale sweep\n\n'+mdtable(['Policy','Held-out at 0.5x','Held-out at 1x','Held-out at 2x'],scale_rows)
md+='\n\n![Training loss versus continuation bytes and seconds](SHoEbox_Stage4_Loss_Bytes_and_Time.png)\n\nBlock training NLL; one continuation per policy; all plotted policies use scale 1x.'
md+='\n\n## Implementation\n\n'+'\n\n'.join('- '+v for v in implementation)
md+='\n\n## Validation and compatibility\n\n'+verification+'\n\n'+compatibility
md+='\n\n## Generation examples\n\nSame prompt, seed 17, temperature 0.8, 64 sampled bytes, ASCII plus newline; after the 2,048-byte continuation.\n'
for label,value in generations:md+='\n'+label+':\n\n````text\n'+value+'\n````\n'
md+='\n## Reproduction\n\n'+repro+'\n\n## Remaining updates\n\n'+mdtable(['Stage','Update','Work remaining'],remaining)+'\n\n'+assessment+'\n'
(OUT/(STEM+'.md')).write_text(md)
for folder,regular,bold in [('/usr/share/fonts/truetype/open-sans','OpenSans-Regular.ttf','OpenSans-Bold.ttf'),('/usr/share/fonts/truetype/dejavu','DejaVuSans.ttf','DejaVuSans-Bold.ttf')]:
    if (Path(folder)/regular).is_file():
        pdfmetrics.registerFont(TTFont('ReportSans',str(Path(folder)/regular)))
        pdfmetrics.registerFont(TTFont('ReportSansBold',str(Path(folder)/bold)));break
styles=getSampleStyleSheet()
for name,size,leading,bold in [('TitleX',22,27,True),('HeadX',12,16,True),('BodyX',9.1,12.8,False),('SmallX',8,11,False),('CellX',7.7,10.3,False)]:
    styles.add(ParagraphStyle(name=name,fontName='ReportSansBold' if bold else 'ReportSans',fontSize=size,leading=leading,
                             spaceAfter=0 if name=='CellX' else 8,textColor=colors.HexColor('#16354b')))
def P(value,style='BodyX'):return Paragraph(escape(value).replace('\n','<br/>'),styles[style])
def T(head,body,widths):
    table=Table([[P(v,'CellX') for v in row] for row in [head,*body]],colWidths=widths,repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5eef3')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#c8d8e0'))]))
    return table
story=[P('SHoEbox: union reverse learning','TitleX'),P('STRUCTURAL OPTIMIZATION STAGE 4 | 13 NOVEMBER 2026','SmallX'),P(summary),
       P('Matched throughput','HeadX'),T(['Workload','Stage 3 B/s','Stage 4 B/s','Change','Pairs'],throughput,[217,83,83,80,49]),Spacer(1,9),P(protocol,'SmallX'),
       P('Numerical comparison','HeadX'),T(['Workload','Max logit drift','NLL delta','Route diff.','Top-byte diff.'],numeric,[208,92,92,60,60]),Spacer(1,9),P(parity),
       P('Baseline: '+BASE,'SmallX'),PageBreak(),P('Short bank-version epochs','TitleX'),P(sweep_protocol,'SmallX'),
       T(['Policy at 1x','B/s','Train NLL','Held-out NLL','Content VJPs'],epoch_rows,[143,78,88,97,106]),Spacer(1,9),P(sweep_assessment),
       P('Reverse update-scale sweep','HeadX'),T(['Policy','Held-out at 0.5x','Held-out at 1x','Held-out at 2x'],scale_rows,[125,129,129,129]),Spacer(1,12),
       Image(str(figure),width=512,height=176),P('Block training NLL; one continuation per policy; plotted policies use scale 1x.','SmallX'),
       PageBreak(),P('Implementation and validation','TitleX')]
for i,v in enumerate(implementation,1):story.append(P(str(i)+'. '+v))
story.extend([P('Verification','HeadX'),P(verification,'SmallX'),P(compatibility,'SmallX'),PageBreak(),P('Samples and remaining stages','TitleX'),
              P('Same prompt, seed 17, temperature 0.8, 64 sampled bytes, ASCII plus newline; after the 2,048-byte continuation.','SmallX'),
              T(['Policy at 1x','Generated continuation (prompt included)'],generations,[108,404]),Spacer(1,12),
              P('Remaining optimization stages','HeadX'),T(['Stage','Update','Work remaining'],remaining,[35,147,330]),Spacer(1,10),P(assessment),
              P('Reproduction and evidence','HeadX'),P(repro,'SmallX')])
def footer(canvas,doc):
    canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor('#526575'))
    canvas.drawString(50,27,'SHoEbox | Structural optimization stage 4');canvas.drawRightString(562,27,str(doc.page))
SimpleDocTemplate(str(OUT/(STEM+'.pdf')),pagesize=(612,792),leftMargin=50,rightMargin=50,topMargin=42,bottomMargin=45,
                  title='SHoEbox structural optimization stage 4').build(story,onFirstPage=footer,onLaterPages=footer)
print(json.dumps(dict(tests=tests,timed_continuations=timed,throughput=throughput,epoch_rows=epoch_rows,scope_vjp_reduction=removed),indent=2))
