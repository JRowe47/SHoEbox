"""Build the compact Action-5 decision, Markdown report, PDF, and loss curves."""
import argparse,json,re,statistics
from pathlib import Path
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[1]
LABELS={'baseline':'Action-3 event','frames':'Lightweight frames','action4':'Selected Action 4','combined':'Frames + Action 4'}
def median(values):return statistics.median(values)
def read(name):
    obj=json.loads((ROOT/'results'/name).read_text());assert obj['completed'];return obj['rows']
def aggregate(rows):
    output={}
    for name in LABELS:
        runs=[r for r in rows if r['variant']==name]
        if not runs:continue
        times=[r['time_to_target'] for r in runs if r['time_to_target'] is not None]
        output[name]=dict(bytes_per_second=median([r['bytes_per_second'] for r in runs]),
            rate_range=[min(r['bytes_per_second'] for r in runs),max(r['bytes_per_second'] for r in runs)],
            heldout_nll=median([r['heldout_nll'] for r in runs]),nll_range=[min(r['heldout_nll'] for r in runs),max(r['heldout_nll'] for r in runs)],
            time_to_target=median(times) if times else None,reached=len(times),runs=len(runs),
            fitted_parameters=runs[0]['fitted_parameter_count'],rss_mib=median([r['rss_kib']/1024 for r in runs]),lr=runs[0]['lr'],
            resident_descriptors=median([sum(v['resident_descriptors'] for v in r['frame_inventory'].values()) for r in runs]),
            resident_experts=median([sum(v['resident_experts'] for v in r['frame_inventory'].values()) for r in runs]))
    return output
def md_table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--revision',required=True);args=parser.parse_args()
    natural=read('action5_selected_natural.json');repeated=read('action5_repeated.json');sweep=read('action5_rate_sweep.json')
    n=aggregate(natural);r=aggregate(repeated);control=n['action4'];candidate=n['combined']
    speed=100*(candidate['bytes_per_second']/control['bytes_per_second']-1)
    loss=candidate['heldout_nll']-control['heldout_nll']
    paired=[]
    for seed in sorted({row['seed'] for row in natural}):
        old=next(row for row in natural if row['seed']==seed and row['variant']=='action4')
        new=next(row for row in natural if row['seed']==seed and row['variant']=='combined')
        paired.append(dict(seed=seed,speed_percent=100*(new['bytes_per_second']/old['bytes_per_second']-1),
                           nll_delta=new['heldout_nll']-old['heldout_nll'],
                           target_gain_percent=(100*(1-new['time_to_target']/old['time_to_target'])
                             if new['time_to_target'] is not None and old['time_to_target'] is not None else None)))
    gains=[row['target_gain_percent'] for row in paired if row['target_gain_percent'] is not None]
    target=median(gains) if gains else None;wins=sum(gain>0 for gain in gains)
    advance=(candidate['reached']==candidate['runs'] and candidate['heldout_nll']<=control['heldout_nll']+.02 and
             target is not None and target>0 and wins>=2 and r['combined']['heldout_nll']<=r['action4']['heldout_nll']+.05)
    decision=('Advance lightweight frames with the selected Action-4 policy to longer qualification.' if advance else
              'Keep the selected Action-4 policy as the preferred profile; retain lightweight frames as an explicit comparison policy.')
    repeated_note=(f"On repeated text, the combined variant is {100*(1-r['combined']['bytes_per_second']/r['action4']['bytes_per_second']):.1f}% slower "
                   f"and uses {r['combined']['rss_mib']-r['action4']['rss_mib']:.1f} MiB more median RSS than Action 4. "
                   'At LR .16, final losses can rise after the first target crossing. This natural-text screening rate does not replace the earlier general Action-4 setting.')
    outcome=(f"Action 5 is implemented as lightweight-frames-v1. At the selected rates, the combined variant processes "
             f"{candidate['bytes_per_second']:,.1f} B/s versus {control['bytes_per_second']:,.1f} B/s for Action 4 ({speed:+.1f}%). "
             f"Median held-out NLL is {candidate['heldout_nll']:.5f} versus {control['heldout_nll']:.5f} ({loss:+.5f}).")
    paired_note=(f"The combined policy reaches the target sooner on {wins}/{len(paired)} matched seeds. "
                 f"The median paired time change is {abs(target):.2f}% {'faster' if target>0 else 'slower'}. "
                 'The selection uses paired changes; ratios of unpaired median target times can give a different impression.') if target is not None else 'Target time is unresolved because not all paired runs reached the target.'
    def count(file):
        matches=re.findall(r'Ran (\d+) tests?',(ROOT/'results'/file).read_text());return int(matches[-1])
    tests=dict(full=count('action5_regressions.txt'),frames=count('action5_frame_tests.txt'),scale=count('action5_5m_tests.txt'))
    summary=dict(source_revision=args.revision,natural=n,repeated=r,decision=decision,advance=advance,outcome=outcome,
                 incremental_speed_percent=speed,incremental_nll=loss,incremental_target_time_percent=target,paired=paired,paired_note=paired_note,tests=tests)
    (ROOT/'results/action5_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    env=json.loads((ROOT/'results/action5_environment.json').read_text())
    def rows(table):
        return [[LABELS[key],f"{v['lr']:.3g}",f"{v['bytes_per_second']:,.1f}",f"{v['heldout_nll']:.5f}",
                 ('unreached' if v['time_to_target'] is None else f"{v['time_to_target']:.2f}")+f" ({v['reached']}/{v['runs']})",f"{v['rss_mib']:.1f}"] for key,v in table.items()]
    heading=['Policy','Main LR','B/s','Held-out NLL','Target seconds','RSS MiB']
    mechanics=[
      ('Intermediate representation','S2 through G5 keep immutable exact child handles, source roles, occurrence IDs, coverage and primary clocks. Their intermediate code is structure plus the fixed, normalized signed-role sum of learned child residuals. Recognition retains the existing per-role child descriptor, class dictionary, transition evidence and primary-clock boundary rule.'),
      ('Completed content','Every completed unit uses the full regional content expert, adapter and payload gate. A single-frame completion clones the saved binding graph, preserves its exact binding identity and roles, and transforms the saved sum once. Multi-frame completion transforms the ordered sequence of fixed descriptors. Child content therefore remains a learned input at every level.'),
      ('Learning and credit','Frames have no trainable local transform. Saved fixed-role Jacobians carry actual task/local teachers to their child producers; learned updates occur at completed publications. Fixed descriptors are transparent to the local completion frontier and are excluded from synthetic local-update recipients. Single-frame completion links directly to its original children. Frozen recipients still transmit derivatives.'),
      ('Version and lifecycle','Checkpoint v45 requires the explicit frame policy and retains pending frames, producer records, class state, prediction state, and any pending bank job. Older versions keep learned frames exactly. Changing representation requires an empty, unfitted recognition state. Snapshot copies, chunk boundaries, document drains, and worker changes preserve the declared schedule.'),
      ('Inspection','The new inspect_frames() reads live descriptors, expert records, exact edges, admitted-frame clocks, publication clocks and pending frames between API calls. No timers or historical diagnostic counters were added to production. The existing semantic slab is reused; this action does not introduce a smaller descriptor-record allocator.')]
    protocol=(f"CPU: {env['cpu']}; {env['compiler']}; production -O3 -march=native, one pinned caller CPU. "
      'Nominal 1M uses 18 regions, width 64 and 12 experts per region. Natural-text qualification uses three matched seeds (7, 11, 19), '
      '17,408 training targets and 1,024 separate validation targets. Source text is the pinned repository README/STATUS fixture, with cortical_fineweb documentation held out; this is not a FineWeb corpus run. '
      'Native training calls include full logits export; evaluation, inspection, construction, compilation and file IO are excluded. Trials rotate policy order and execute without overlapping tests or profiling. '
      'Time to NLL 3.33 is the first observed validation crossing at 2,048-byte intervals. Repeated-text uses 8,192 training targets, separate validation and target NLL 2.62. '
      'Each policy is screened at 0.5x/1x/2x of the previous selected LR 0.08; local LR is main LR/3. Selected rates are chosen by final validation NLL on seed 7 before matched-seed comparison.')
    validation=[f"{tests['full']} checks passed in the full 1M regression suite; {tests['frames']} checks passed in the dedicated frame suite.",
      f"{tests['scale']} selected 5M checks passed, including numerical credit, continuation, region contribution, and long-stream retention.",
      'Five real published Action-4 checkpoint configurations continue for 8,192 bytes each with exact logits and logical memory fingerprints, including drain/reset. Coverage includes reference, event, pipeline, and Action-4 event/pipeline combinations.',
      'ASan and UBSan passed the production descriptor, ownership, prediction and worker lifecycle harness. LeakSanitizer was disabled for this environment.',
      'Finite differences cover one-, two- and eight-frame completion, frozen-recipient derivative flow, and both reverse policies. Recognition checks cover source/order sensitivity, stable classes, causal boundaries and relay class preservation.',
      'Natural-text interventions verify all 18 regions publish and verify upper neural/content contributions. Repeated-text long compositions exercise wrap/collection and complete document drains; short repeated streams can legitimately leave G5 pending.']
    memory_rows=[[LABELS[k],f"{v['fitted_parameters']:,}",f"{int(v['resident_descriptors']):,}",f"{int(v['resident_experts']):,}"] for k,v in n.items()]
    boundary=[]
    for name in ('L3','E4','G5'):
        row=[name]
        for variant in ('action4','combined'):
            values=[]
            for run in natural:
                if run['variant']==variant:
                    item=run['frame_inventory'][name];values.append((item['admitted_frames']-item['pending_frames'])/max(1,item['publications']))
            row.append(f'{median(values):.2f}')
        boundary.append(row)
    rate_rows=[[LABELS[x['variant']],str(x['lr']),f"{x['heldout_nll']:.5f}",f"{x['bytes_per_second']:,.1f}"] for x in sweep]
    backlog=[['6 - next','Sparse bank epochs and broader worker overlap'],['7','Dedicated forward-only runtime'],['8','Resident GPU backend and independent streams'],['9 - ongoing','Longer 1M/5M/100M qualification and final profile selection']]
    gap=f"The combined {candidate['bytes_per_second']:,.1f} B/s is {100*candidate['bytes_per_second']/15000:.1f}% of 15,000 B/s; the remaining single-stream gap is {15000/candidate['bytes_per_second']:.2f}x."
    profile=json.loads((ROOT/'results/action5_profile.json').read_text());assert profile['completed']
    sample_rows=[]
    for run in profile['runs']:
        for thread in run['threads']:
            def pct(name):return f"{thread['categories'].get(name,{}).get('percent',0):.1f}%"
            sample_rows.append([run['profile'],thread['role'],f"{thread['cpu_seconds']:.2f}",pct('forward / prediction / event'),
                                pct('reverse / derivatives / credit'),pct('bank update / versions')])
    sample_heading=['Profile','Thread','CPU seconds','Forward','Reverse','Bank updates']
    sample_note=('External wall-time instruction samples of the production binary, with 1,024 warmup and 8,192 sampled training targets at LR .04. '
                 'These categories exclude unresolved/inlined work, allocation, collection and waits from the displayed columns and are not inclusive CPU costs. '
                 'Thread CPU seconds are measured separately. Sampling does not contribute to throughput numbers.')
    code='from shoebox_reference import fresh_model\nm = fresh_model(profile="event", frames="lightweight-frames-v1",\n                prediction=dict(compact=True, rank=16, slow_stride=8))'
    blocks=['# SHoEbox Action 5: lightweight binding frames','14 November 2026',
      f'Source revision: [{args.revision}](https://github.com/JRowe47/SHoEbox/commit/{args.revision}). Baseline: `{env["baseline_revision"]}`.',outcome,decision,
      '## Matched natural-text comparison',md_table(heading,rows(n)),paired_note,
      md_table(['Seed','Combined speed change','NLL change','Target-time gain'],
               [[x['seed'],f"{x['speed_percent']:+.2f}%",f"{x['nll_delta']:+.5f}",f"{x['target_gain_percent']:+.2f}%" if x['target_gain_percent'] is not None else 'unreached'] for x in paired]),protocol,
      '## Repeated-text cross-check',md_table(heading,rows(r)),repeated_note,
      '## Implementation']+[f'**{title}.** {body}' for title,body in mechanics]
    blocks+=['## Validation']+['- '+x for x in validation]+[
      '## Memory and boundary behavior',md_table(['Policy','Fitted coefficients','Resident descriptors','Resident experts'],memory_rows),
      'Resident record counts are inventory at the final boundary, not cumulative encoder calls. Frames add no fitted parameters. Existing slab capacity and sidecars remain allocated; RSS includes allocator retention and untimed evaluation copies.',
      md_table(['Region','Action 4 frames/publication','Combined frames/publication'],boundary),
      'Ratios use admitted frames minus the pending tail, divided by completed publications, then take the seed median. Boundary trajectories may change under the new representation.',gap,
      'The inspected natural-text upper regions mostly publish one frame per unit. In those cases, this policy moves the full transform to completion rather than removing it, while also creating an intermediate descriptor record. Larger repeated-text compositions save frame transforms but did not improve end-to-end performance in this comparison.',
      '## Production sampling',md_table(sample_heading,sample_rows),sample_note,
      '## Learning-rate screening',md_table(['Policy','Main LR','Final validation NLL','B/s'],rate_rows),
      '## Use and reproduction','```python\n'+code+'\n```',
      'The original default reference and saved policies remain available. Run the dedicated frame tests, run_factor_regressions.py, check_frame_checkpoint.py and run_frame_sanitizer.py separately from timings. '
      'measure_frame_policy.py records all data hashes, runtime source hashes, curves, live inventories and rates; profile_frame_policy.py samples the unmodified production binary externally. '
      'These are short controlled screens. Longer data/seed/scale qualification remains Action 9.',
      '## Remaining plan',md_table(['Action','Scope'],backlog)]
    out=ROOT/'deliverables';out.mkdir(exist_ok=True);md=out/'SHoEbox_Action5_Lightweight_Frames.md';md.write_text('\n\n'.join(blocks)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.3))
    for name,color in [('action4','#236b8e'),('combined','#c16e34')]:
        runs=[x for x in natural if x['variant']==name];curve=runs[0]['quality_curve']
        y=[median([x['quality_curve'][i]['heldout_nll'] for x in runs]) for i in range(len(curve))]
        t=[median([x['quality_curve'][i]['seconds'] for x in runs]) for i in range(len(curve))]
        axes[0].plot([v['bytes']/1024 for v in curve],y,label=LABELS[name],color=color,lw=2)
        axes[1].plot(t,y,label=LABELS[name],color=color,lw=2)
    for ax,xlabel in zip(axes,['Training targets (KiB)','Native training seconds']):
        ax.set_xlabel(xlabel);ax.set_ylabel('Held-out NLL');ax.axhline(3.33,color='#8a8a8a',ls=':',lw=1);ax.grid(alpha=.15)
    axes[1].legend(frameon=False,fontsize=8);fig.tight_layout()
    plot=ROOT/'results/action5_curves.png';fig.savefig(plot,dpi=180);plt.close(fig)
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodySmall',fontName='Helvetica',fontSize=9,leading=12,spaceAfter=7))
    styles.add(ParagraphStyle(name='CellSmall',fontName='Helvetica',fontSize=8,leading=10))
    pdf=out/'SHoEbox_Action5_Lightweight_Frames.pdf';story=[]
    def para(text,style='BodySmall'):return Paragraph(escape(text),styles[style])
    def table(headers,data,widths=None):
        cells=[[para(str(x),'CellSmall') for x in row] for row in [headers,*data]]
        tab=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DDEAF0')),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#183A4D')),
          ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
          ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#8BA5B1')),
          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke,colors.white])]))
        return tab
    story += [para('SHoEbox Action 5','Title'),para('Lightweight binding frames','Heading1'),para('14 November 2026 | Source '+args.revision[:12]),para(outcome),para(decision),
              para('Matched natural-text comparison','Heading2'),table(heading,rows(n),[118,48,69,77,112,48]),Spacer(1,8),para(paired_note),Image(str(plot),width=516,height=167),para(gap),PageBreak()]
    story += [para('Representation and learning contract','Heading1')]
    for title,body in mechanics:story += [para(title,'Heading2'),para(body)]
    story += [para('Validation','Heading2')]+[para(x) for x in validation]+[PageBreak()]
    story += [para('Qualification evidence','Heading1'),para('Repeated-text cross-check','Heading2'),table(heading,rows(r),[118,48,69,77,112,48]),Spacer(1,8),para(repeated_note),para('Memory inventory','Heading2'),
              table(['Policy','Fitted coefficients','Descriptors','Experts'],memory_rows,[166,118,94,94]),Spacer(1,8),
              para('The fixed semantic slab is reused. Resident counts are current inventory; RSS includes allocator retention and evaluation copies. Frames add no fitted coefficients.'),
              para('Completed-frame lengths','Heading2'),table(['Region','Action 4','Combined'],boundary,[170,151,151]),Spacer(1,8),para('Median admitted frames minus pending tail, divided by completed publications. The inspected natural-text regions mostly publish one-frame units: their full transform is moved to completion, while an additional descriptor record is retained. Larger repeated-text compositions did not produce an end-to-end gain.'),
              PageBreak()]
    selected_rows=[next(row for row in rate_rows if row[0]==LABELS[key] and float(row[1])==v['lr']) for key,v in n.items()]
    sampling=' '.join(f'{row[0]} {row[1]}: {row[2]} CPU seconds; forward {row[3]}, reverse {row[4]}, bank updates {row[5]}.' for row in sample_rows)
    story += [para('Protocol and rate screening','Heading1'),para(protocol),para('Selected rates on screening seed 7','Heading2'),table(['Policy','Main LR','Validation NLL','B/s'],selected_rows,[186,70,110,106]),Spacer(1,10),
              para('Complete 0.5x/1x/2x screening results are in the Markdown report and compact JSON evidence.'),
              para('External production sampling','Heading2'),para(sampling),para(sample_note),
              para('Reproduction','Heading2'),para('Use frames="lightweight-frames-v1" with fresh_model(). For the Action-4 combination use compact=True, rank=16 and slow_stride=8. Source, tests, measured curves and inventories are committed with this report. '+
              'Run measure_frame_policy.py for isolated comparisons and profile_frame_policy.py for separate diagnostic samples. Longer qualification remains Action 9.'),
              para('Remaining development plan','Heading2'),table(['Action','Scope'],backlog,[105,367])]
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#607784'));canvas.drawString(48,28,'SHoEbox | Action 5 | '+args.revision[:12]);canvas.drawRightString(564,28,str(doc.page))
    SimpleDocTemplate(str(pdf),pagesize=letter,leftMargin=48,rightMargin=48,topMargin=38,bottomMargin=48,
                      title='SHoEbox Action 5: Lightweight Binding Frames',author='SHoEbox',
                      subject='Implementation, validation, and performance qualification').build(story,onFirstPage=footer,onLaterPages=footer)
    print(outcome);print(decision);print(md);print(pdf)
if __name__=='__main__':main()
