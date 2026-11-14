"""Render the completed Action-3 evidence as ordinary Markdown and PDF files."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
NAMES=dict(reference='Reference',event='Event',compact_serial='Compact serial',compact_threaded='Compact threaded')

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--revision',required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    read=lambda name:json.loads((ROOT/'results'/f'action3_{name}.json').read_text())
    s=read('summary');env=read('environment');profile=read('profile_after')
    assert s['all_exact'] and profile['completed']
    url='https://github.com/JRowe47/SHoEbox/commit/'+a.revision
    long_rows=[]
    for c in s['long_comparisons']:
        info=s['long_memory'][c['profile']]
        long_rows.append([NAMES[c['profile']],len(c['after_rates']),f"{c['before_bps']:,.1f}",f"{c['after_bps']:,.1f}",
                          f"{100*(c['speedup']-1):+.1f}%",f"{info['heldout_nll']:.6f}"])
    short_rows=[[c['workload'].title(),NAMES[c['profile']],f"{c['before_bps']:,.1f}",f"{c['after_bps']:,.1f}",
                 f"{100*(c['speedup']-1):+.1f}%"] for c in s['short_comparisons']]
    memory_rows=[]
    for name,info in s['long_memory'].items():
        w=info['workspace'];memory_rows.append([NAMES[name],f"{info['before_rss_mib']:.1f} / {info['after_rss_mib']:.1f}",
            f"{w['local_workspace_bytes']/1024:.1f}",f"{w['reverse_workspace_bytes']/1024:.1f}"])
    profile_rows=[]
    for row in profile['runs']:
        for t in row['threads']:
            value=lambda key:t['categories'].get(key,{}).get('percent',0)
            profile_rows.append([NAMES[row['profile']]+(' worker' if t['role']!='caller' else ''),f"{t['cpu_seconds']:.2f}",
                f"{value('reverse / derivatives / credit'):.1f}",f"{value('bank update / versions'):.1f}",
                f"{value('forward / prediction / event'):.1f}",f"{value('collection / retention'):.1f}"])
    intro=('Action 3 is implemented in the default completion runtime. Reusable reverse plans, indexed local-credit '
           'accumulators, compatible adjoint packets and publication-driven scheduling preserve the existing learning contract. '
           'All 64 production runs (32 before/after pairs) match exact logits, fitted parameters, held-out NLL and logical continuation.')
    finding=(f"The best long-run median is {s['best_long_bps']:,.1f} B/s in compact serial, a 3.4% gain over Action 2. "
             f"This is {100*s['best_long_bps']/15000:.1f}% of the 15,000 B/s target, leaving a {s['target_gap']:.2f}x gap. "
             'The long reference pair gains 2.9%; compact threaded is effectively flat at +0.4%. '
             'Short document-boundary medians decline 2.9% for reference and 1.4% for threaded. The result is workload-dependent.')
    variation=('Natural-text compact serial spans 1136-1172 B/s before and 1165-1192 after; threaded spans 1159-1190 '
               'before and 1108-1177 after. Compact values are medians of three rotated pairs. Reference and event use '
               'one long pair each; short measurements use three pairs per profile and workload. These are matched seed-7 fixture runs.')
    implemented=[
        ('Reverse reuse','Teaching-policy keys preserve exact scale bits and objective separation. Group membership and rate-mass '
         'accumulation keep first-occurrence order. Reusable traversal plans validate immutable slab handles, event frontier clocks '
         'and replay mode; they retain topological batch boundaries. Cached data contains no weight-dependent derivatives.'),
        ('Indexed local credit','Generation-tagged accumulators replace temporary ordered maps. Ordered region/tick views preserve '
         'the previous reduction and mutation sequence. Neural and input reuse denominators remain separate, and frozen recipients '
         'continue transmitting derivatives. Compatible dense requests share bank-row loads before the ordered writer runs.'),
        ('Ready execution','Publications wake dependent regions through a ready bitset. The queue drains in the existing ascending '
         'causal order, once per region per step; late lower-index work carries to the next step. A1/B1/Y keep their byte clocks. '
         'Feedback reads the previous snapshot, while relays and document drains keep their existing behavior.'),
        ('Regional packing','Fixed recipes are prepared together. A1/B1 share the observed byte preparation; compatible S2/R2/F2 '
         'recipes gather immutable child handles together. EC bookkeeping and common context/feedback edge reads are fused. '
         'Numerical encoding and online teaching retain their original regional boundaries and bank read versions.'),
        ('Inspection and workers','Workspace and retained-plan batch inventories are read only at API boundaries. Production adds '
         'no inline observers or historical counters. Existing content-row and pipeline worker paths were retested. The dedicated '
         'contribution-intervention kernel keeps exhaustive source probing for its diagnostic contract.')]
    policy=('No learning-policy, checkpoint-version or fitted-parameter change was introduced. Workspaces and indexes are derived '
            'and rebuilt at API entry or invalidation; saved wire formats remain readable. The measured kernels show no floating-point '
            'drift or routing divergence. All 18 logical regions, source/occurrence clocks, content/context separation and paired episodes remain.')
    validation=[
        '299 regression tests passed at the nominal 1M size, covering local derivatives, mixed objectives, clipping, frozen regions, '
        'relays, retained records, event frontiers, contribution checks, bank epochs and worker continuation.',
        'Seven selected 5M tests passed, including exact packed/scalar adjoints, stale-plan invalidation, serial/threaded/unfused '
        'parity, byte chunks and pending checkpoints.',
        'Actual Action-2 checkpoints resume exactly through 8,192 bytes in reference, one-head event, two-head event and compact '
        'pipeline configurations. Detached regional inventories also match after single-byte steps, followed by document drain and reset.',
        'ASan and UBSan passed the accumulator, derivative-packet, scheduler, stale-generation/frontier and reference/event/worker '
        'lifecycle scenarios. Leak detection was disabled (detect_leaks=0).']
    memory_note=('RSS is endpoint process memory and includes allocator retention from earlier untimed validation copies. Workspace '
                 'columns are reserved vector payload, excluding hash-table and allocator overhead. The threaded endpoint RSS increase '
                 'is about 20 MiB; this change does not establish an overall process-memory reduction.')
    batches=('At the long endpoint, handle-valid cached reference plans contain ready batches of 1-13 records; event and compact '
             'plans contain batches of 1-8. The last local objective retains 44 adjoint slots for reference and 16 for event/compact. '
             'These describe retained workspace layouts, not historical execution frequencies.')
    sampling=('Separate 4,096-byte natural-text runs sample the same production binaries at 250 Hz per thread. Percentages below '
              'are exclusive instruction-pointer samples, with inlined work assigned to its containing symbol; they are not inclusive '
              'CPU-cost estimates. Thread CPU time is measured separately. Unresolved, allocation and wait samples are omitted from '
              'the category columns, so rows need not sum to 100%. Sampling never overlaps throughput measurement.')
    bottleneck=('Reverse/derivative work remains the largest classified caller component: 36.5% of compact-serial samples and '
                '37.4% of compact-threaded caller samples. In compact serial, constituent_union_flush alone is 18.8%, pattern '
                'classification 10.7%, pipeline computation 8.2% and context updates 6.5%. The threaded caller uses 3.49 CPU seconds '
                'and its worker 0.45 seconds in the sampled continuation. More worker threads alone will not remove the caller cost.')
    protocol=(f"Host: {env['cpu_model']}; {env['compiler']}; -O3 -march=native -std=c++17 -pthread. Caller CPU 0, worker CPU 1; "
              '18 regions, width 64, 12 experts per region. Reference has 1,031,952 fitted parameters; event/compact have 1,057,552. '
              'Each run uses 1,024 warmup targets, then 2,048 short or 16,384 long targets in 512-target API chunks. Timings include '
              'full logits export, required worker quiescence and document drains. Compilation, warmup, validation, inspection and '
              'file IO are excluded. LR is 0.04 and local LR 0.04/3. Runs are sequential and isolated.')
    data=('The pinned Action-1 fixtures are reused: repeated sentences; README/STATUS natural text; and 512-target documents. '
          'Validation has 1,024 separate targets. Natural validation comes from cortical_fineweb documentation. These are repository '
          'text fixtures, not a FineWeb corpus run. Per-trial data/binary hashes, quality curves, CPU time and memory inventories '
          'are preserved in action3_*.json. Longer multi-seed learning qualification remains Action 9.')
    backlog=[['4 - next','Compact prediction and fast/slow Y'],['5','Lightweight binding frames'],
             ['6','Sparse bank epochs and broader worker overlap'],['7','Dedicated forward-only runtime'],
             ['8','Resident GPU backend and independent streams'],['9 - ongoing','Longer 1M/5M/100M qualification, tuning and final profile selection']]
    blocks=['# SHoEbox Action 3: reusable learning and ready-event execution',
            '14 November 2026 (America/Denver) | Completion report',
            f"Published revision: [{a.revision}]({url}). Baseline: `{s['baseline_revision']}`.",intro,
            '## Longer natural-text continuation',finding,
            table(['Profile','Pairs','Before B/s','After B/s','Change','Held-out NLL'],long_rows),variation,
            '## Short paired throughput', 'Three rotated pairs; 2,048 targets after warmup.',
            table(['Workload','Profile','Before B/s','After B/s','Change'],short_rows),
            '## Implemented checkpoints']
    blocks.extend(f'{i}. **{title}.** {body}' for i,(title,body) in enumerate(implemented,1))
    blocks += [policy,'## Validation']+['- '+item for item in validation]
    blocks += ['## Memory and batch sizes',table(['Profile','RSS before / after MiB','Local KiB','Reverse KiB'],memory_rows),
               memory_note,batches,'## Production profile',sampling,
               table(['Profile / thread','CPU s','Reverse %','Banks %','Forward %','Retention %'],profile_rows),bottleneck,
               '## Measurement scope',protocol,data,'## Remaining plan',table(['Action','Scope'],backlog),
               '## Reproduction and evidence',
               'Use `src/compare_ready_execution.py`, `src/check_ready_checkpoint.py`, `src/profile_current_baseline.py` and '
               '`tests/run_ready_sanitizer.py` under `experiments/identity_semantics`. The baseline checkout is published Action 2. '
               'Run correctness, timing and sampling separately. The source manifest pins every runtime file.',
               f"Runtime SHA-256: `{s['runtime_sha256']}`.",
               'The download bundle contains changed source, tests, compact evidence and an apply-ready patch. Temporary models, '
               'datasets, compiled binaries and intermediate snapshots are excluded.']
    stem=a.output_dir/'SHoEbox_Action3_Ready_Execution';stem.with_suffix('.md').write_text('\n\n'.join(blocks)+'\n')

    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body3',fontName='Helvetica',fontSize=9.3,leading=12.8,spaceAfter=7,textColor=colors.HexColor('#263747')))
    styles.add(ParagraphStyle(name='Cell3',fontName='Helvetica',fontSize=8,leading=10.5))
    styles['Title'].fontName='Helvetica-Bold';styles['Title'].fontSize=22;styles['Title'].leading=25;styles['Title'].textColor=colors.HexColor('#173A4A')
    styles['Heading1'].fontSize=14;styles['Heading1'].leading=17;styles['Heading1'].spaceAfter=7
    story=[]
    def para(text):story.append(Paragraph(escape(text),styles['Body3']))
    def heading(text):story.append(Paragraph(escape(text),styles['Heading1']))
    def grid(headers,rows,widths):
        values=[[Paragraph(escape(str(v)),styles['Cell3']) for v in row] for row in [headers]+rows]
        t=Table(values,colWidths=[v*528/sum(widths) for v in widths],repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E2EDF1')),('VALIGN',(0,0),(-1,-1),'TOP'),
                              ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                              ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
                              ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#A6BBC4')),
                              ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F4F7F8')])]))
        story.extend([t,Spacer(1,9)])
    story.append(Paragraph('SHoEbox Action 3<br/>Reusable learning and ready execution',styles['Title']))
    para('14 November 2026 | Completion report | Nominal 1M model')
    story.append(Paragraph('Published: <link href="'+url+'">'+a.revision[:12]+'</link> | Baseline: '+s['baseline_revision'][:12],styles['Body3']))
    para(intro);heading('16,384-byte natural-text continuation')
    grid(['Profile','Pairs','Before B/s','After B/s','Change','Held-out NLL'],long_rows,[120,36,75,75,60,85])
    para(finding);para(variation);heading('Short paired throughput')
    grid(['Workload','Profile','Before B/s','After B/s','Change'],short_rows,[80,125,85,85,65])
    para('Three rotated pairs per workload and profile; 2,048 measured targets after 1,024 warmup targets.')
    story.append(PageBreak());heading('Implemented checkpoints')
    for i,(title,body) in enumerate(implemented,1):para(f'{i}. {title}. {body}')
    para(policy);heading('Validation')
    for item in validation:para(item)
    story.append(PageBreak());heading('Memory and production profile')
    grid(['Profile','RSS before / after MiB','Local KiB','Reverse KiB'],memory_rows,[130,145,80,85])
    para(memory_note);para(batches)
    grid(['Profile / thread','CPU s','Reverse %','Banks %','Forward %','Retention %'],profile_rows,[155,45,65,55,65,70])
    para(sampling);para(bottleneck);heading('Remaining plan');grid(['Action','Scope'],backlog,[80,448])
    story.append(PageBreak());heading('Measurement and reproduction')
    para(protocol);para(data)
    heading('Source and checkpoint contract')
    para('The baseline is published Action 2 ('+s['baseline_revision'][:12]+'). The source manifest and binary hashes identify the exact tested implementation. '
         'Sampling and timed continuations used identical production binaries. No checkpoint extension or numerical-policy change was required.')
    para('All paired held-out values are identical. The single-seed learning curves therefore measure the same trajectory; the throughput differences are execution costs. '
         'The full reference remains the default, while event and pipeline retain their explicitly selected learning schedules.')
    heading('Reproduce from the repository root')
    for command in [
        'python experiments/identity_semantics/src/compare_ready_execution.py --baseline /path/to/action2-checkout --output comparison.json',
        'python experiments/identity_semantics/src/check_ready_checkpoint.py --baseline /path/to/action2-checkout --output checkpoint_checks.json',
        'python experiments/identity_semantics/src/profile_current_baseline.py --targets 4096 --hz 250 --output profile.json',
        'python experiments/identity_semantics/tests/run_ready_sanitizer.py']:
        para(command)
    para('Run correctness, timing and sampling separately. Use 12 experts and production diagnostics disabled for throughput. '
         'The published short suite uses 2,048 targets and three repetitions on repeated/document fixtures; long reference/event use one 16,384-target pair, '
         'and long compact profiles use three rotated 16,384-target pairs. Full invocation settings and all per-trial results are retained in the evidence.')
    heading('Deliverables');para(blocks[-1])
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#CFDCE2'));canvas.line(42,38,570,38)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#61737D'))
        canvas.drawString(42,25,'SHoEbox | Action 3 | '+a.revision[:12]);canvas.drawRightString(570,25,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(stem.with_suffix('.pdf')),pagesize=(612,792),rightMargin=42,leftMargin=42,topMargin=36,bottomMargin=49,
                          title='SHoEbox Action 3: reusable learning and ready execution',author='SHoEbox')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(stem.with_suffix('.md'));print(stem.with_suffix('.pdf'))

if __name__=='__main__':main()
