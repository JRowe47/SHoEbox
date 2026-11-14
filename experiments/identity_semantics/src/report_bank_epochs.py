"""Generate the compact Action-6 qualification summary and downloadable report."""
import argparse,json,re,statistics,subprocess,sys
from pathlib import Path
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from measure_current_baseline import source_manifest
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Preformatted
median=statistics.median
LABELS={'event':'Action 4 event','legacy':'One-byte pipeline','e4l1':'4-byte serial','e4l1-w2':'4-byte + worker',
        'e4l1-dense':'4-byte dense','e16l0':'16-byte zero lag','e16l1':'16-byte serial','e16l1-w2':'16-byte + worker'}

def read(name):return json.loads((ROOT/'results'/name).read_text())
def aggregate(rows):
    output=[]
    for rate in sorted({r['lr'] for r in rows}):
        for variant in dict.fromkeys(r['variant'] for r in rows):
            a=[r for r in rows if r['lr']==rate and r['variant']==variant]
            if not a:continue
            crossings=[r['time_to_target'] for r in a if r['time_to_target'] is not None]
            output.append(dict(variant=variant,lr=rate,runs=len(a),bps=median(r['bytes_per_second'] for r in a),
                bps_range=[min(r['bytes_per_second'] for r in a),max(r['bytes_per_second'] for r in a)],
                nll=median(r['heldout_nll'] for r in a),target_seconds=median(crossings) if crossings else None,
                reached=len(crossings),rss_mib=median(r['rss_kib']/1024 for r in a),
                arena_mib=median(r['pipeline']['reserved_bytes']/1048576 for r in a),
                pending_mib=median(r['pipeline']['pending_state_bytes']/1048576 for r in a),
                cpu_seconds=median(r['process_cpu_seconds'] for r in a)))
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);a=p.parse_args()
    natural=read('action6_selected_natural.json')['rows'];workers=read('action6_workers.json')['rows'];repeated=read('action6_repeated.json')['rows']
    n=aggregate(natural);w=aggregate(workers);r=aggregate(repeated);scale=aggregate(read('action6_5m_storage.json')['rows'])
    paired=[]
    for rate in sorted({v['lr'] for v in natural}):
        for seed in sorted({v['seed'] for v in natural}):
            old=next(v for v in natural if v['variant']=='event' and v['seed']==seed and v['lr']==rate)
            new=next(v for v in natural if v['variant']=='e4l1-w2' and v['seed']==seed and v['lr']==rate)
            paired.append(dict(seed=seed,lr=rate,speed_percent=100*(new['bytes_per_second']/old['bytes_per_second']-1),
                nll_delta=new['heldout_nll']-old['heldout_nll'],
                target_gain_percent=100*(1-new['time_to_target']/old['time_to_target']) if old['time_to_target'] and new['time_to_target'] else None))
    base=next(x for x in n if x['variant']=='event' and x['lr']==.16)
    selected=next(x for x in n if x['variant']=='e4l1-w2' and x['lr']==.16)
    primary=[v for v in paired if v['lr']==.16]
    speed=median(v['speed_percent'] for v in primary);gain=median(v['target_gain_percent'] for v in primary if v['target_gain_percent'] is not None)
    delta=selected['nll']-base['nll'];wins=sum(v['target_gain_percent'] is not None and v['target_gain_percent']>0 for v in primary)
    worker=next(x for x in w if x['variant']=='e4l1-w2');serial=next(x for x in w if x['variant']=='e4l1');dense=next(x for x in w if x['variant']=='e4l1-dense')
    advance=speed>0 and gain>0 and wins>=2 and delta<=.02
    decision=('Advance four-byte sparse epochs with one-epoch lag and one learning worker as the preferred measured CPU training profile.' if advance else
              'Retain Action 4 as the preferred profile; keep sparse epochs as an explicit measured alternative.')
    outcome=(f"Action 6 is implemented. At main/local LR 0.16 / 0.05333, the four-byte worker profile reaches median {selected['bps']:,.1f} B/s "
             f"versus {base['bps']:,.1f} B/s for Action 4. The median paired throughput gain is {speed:.1f}%, and time to NLL 3.33 improves "
             f"by {gain:.1f}% ({wins}/3 seeds faster). Median final NLL is {selected['nll']:.5f} versus {base['nll']:.5f}.")
    env=read('action6_environment.json');profile=read('action6_profile.json')
    profile_rows=[]
    for run in profile['runs']:
        for t in run['threads']:
            profile_rows.append([LABELS.get(run['profile'],run['profile']),t['role'].replace('_',' '),f"{t['cpu_seconds']:.2f}",
              f"{t['categories'].get('wait / syscall',{}).get('percent',0):.1f}%",
              f"{t['categories'].get('allocation / copying',{}).get('percent',0):.1f}%",
              f"{t['categories'].get('other / unresolved',{}).get('percent',0):.1f}%"])
    summary=dict(source_revision=a.revision,baseline='dc3b9b0994565f9c5fc6ebcabbe2d2172ca60ce8',natural=n,workers=w,repeated=r,scale_5m=scale,paired=paired,
                 primary_speed_percent=speed,primary_target_gain_percent=gain,primary_nll_delta=delta,advance=advance,decision=decision,outcome=outcome,
                 tests=dict(production_final=37,scale_5m=4,broad_production_passed=122,broad_total=124,audit_counter_checks=2,step_clock=2),
                 remaining=['7: forward-only runtime','8: resident GPU and stream batching','9: longer multi-scale qualification and 15k B/s target'])
    (ROOT/'results/action6_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    manifest=source_manifest()
    manifest['published_implementation_revision']=a.revision
    (ROOT/'results/action6_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def quality_rows(table):
        return [[LABELS.get(x['variant'],x['variant']),f"{x['lr']:.2f}",f"{x['bps']:,.1f}",f"{x['nll']:.5f}",
                '-' if x['target_seconds'] is None else f"{x['target_seconds']:.2f} ({x['reached']}/{x['runs']})",f"{x['arena_mib']:.1f}"] for x in table]
    quality_head=['Profile','LR','B/s','Held-out NLL','Target seconds','Bank MiB']
    storage_rows=[[LABELS.get(x['variant'],x['variant']),f"{x['bps']:,.1f}",f"{x['arena_mib']:.1f}",f"{x['nll']:.5f}"] for x in w]
    protocol=(f"Host: {env['cpu_model']}; {env['compiler']}; production -O3 -march=native -std=c++17 -pthread. "
      "Caller CPU 0; learning worker CPU 1. Nominal 1M: 18 regions, width 64, 12 experts per region; "
      "1,014,800 fitted parameters with compact rank-16 forecasting and Y stride 8. Learned frames are retained. "
      "Natural-text qualification uses 17,408 training targets, 1,024 separate validation targets and matched seeds 7, 11, 19. "
      "Data is the pinned repository README/STATUS fixture and held-out cortical_fineweb documentation, not a FineWeb corpus run. "
      "Only native training calls are timed, including logits export and worker quiescence. Compilation, construction, evaluation, inspection and file IO are excluded. "
      "All training bytes, including startup, count in elapsed training time; there is no excluded warmup for these quality comparisons. "
      "Target time is the first observed crossing at 2,048-byte validation intervals. Timings, tests and profiling run separately.")
    tuning=("The short screen tests 30 combinations: event, historical pipeline and all eight epoch/lag pairs at main LR .08/.16/.32, with local LR main/3. "
      "A 17,408-byte seed-7 rate comparison and three matched seeds at .16 and .32 extend that screen. The primary comparison retains the prior selected .16 rate. "
      "Repeated-text checks use .08, 8,192 targets, three seeds and target NLL 2.62; document checks use four 512-target documents and irregular 257-byte chunks.")
    mechanics=[
      ('Sparse versions and ownership','The committed upper content banks are immutable during each epoch. Unchanged banks use references; only changed W tiles and bias/router vectors are copied. Two reused arenas retain shared input values and clipped low-rank factors. Router-only jobs do not copy W. The dense-copy oracle uses identical arithmetic.'),
      ('Epochs and worker lag','Lengths 1/4/8/16 and lag 0/1 epochs are separate configuration fields. Lag 0 commits at the end of the epoch. Lag 1 reads the prior visible bank version, commits the preceding job after the current epoch, and applies the new factors to that committed base. Each job records read/base versions, commit sequence, objective mask and step interval. One caller owns commits; the worker only owns immutable job data.'),
      ('Learning boundaries','Root union, per-request clipping and router normalization retain their original scopes. Roots are consumed into factors before API boundaries and are not merged across bytes. Event replay remains per step, followed by terminal synthetic credit. MinKron context, adapters, forecasting, bottom experts and decoder retain online schedules; slow Y retains its separate stride.'),
      ('Supported policies','One or two MinKron context heads, local event or full ancestry credit, compact forecasting, fast/slow Y and either frame policy are supported. This supplies sparse ancestry epochs for MinKron context. The older dense-context reverse epoch remains unchanged and cannot be combined with this schedule. The bank clock counts training bytes plus forced document-drain passes, separately from primary byte clocks.'),
      ('Checkpoint and reset','Version 46 and wire extension 3 preserve completed pending results and unfinished factor arenas. Save/load, snapshots, worker changes and arbitrary chunks preserve continuation. Frozen evaluation parks pending jobs and uses committed banks. A reset bug is repaired: final replay factors are now committed before their producer records are released. Historical pipeline checkpoint formats remain loadable.')]
    verification=("Final production: 37 tests passed, including all eight schedules, worker/serial/unfused/dense parity, partial restarts, an 8,193-byte wrap/collection continuation, "
      "task coverage, frozen banks, malformed imports, reset replay and causal frozen evaluation. Four targeted 5M checks and two explicit step-clock checks passed. "
      "The broader 124-test sweep passed 122 in production; its two tests that require diagnostic counters passed in the separate audit build. "
      "ASan/UBSan passed every epoch/lag pair with serial and worker execution. LeakSanitizer is unavailable under this host's ptrace environment; address and undefined-behavior checks ran with leak detection disabled.")
    storage=(f"At LR .08 over three rotated 8,192-byte trials, sparse four-byte serial jobs use {serial['arena_mib']:.1f} MiB of reserved bank arenas versus "
      f"{dense['arena_mib']:.1f} MiB for dense copies. Their held-out NLL is identical. Sparse serial is {100*(serial['bps']/dense['bps']-1):.1f}% faster. "
      f"Adding one learning worker improves the median throughput by {100*(worker['bps']/serial['bps']-1):.1f}% over the same serial schedule. "
      "Sixteen-byte jobs retain roughly twice the arena storage and do not beat the four-byte worker configuration in this screen. Bank MiB excludes the main model, record archives and allocator overhead; full process RSS is in the JSON results.")
    scale_note=('The 5M storage cross-check uses 64 experts per region and three rotated 4,096-target runs at LR .08. Sparse and dense jobs produce identical held-out loss; the comparison isolates version copying and storage at the larger bank count.')
    profiling=("External production instruction sampling uses a separate diagnostic run with 1,024 warmup and 8,192 measured targets at main/local LR .04/.01333. "
      "Samples are wall-time instruction positions, not inclusive CPU attribution. Synchronization and copying are observed outside training. "
      "The four-byte worker uses 0.68 CPU seconds versus 5.28 for the caller; reverse/credit work accounts for 37.0% of caller samples. This favors one learning worker; a larger disjoint-bank worker team is not selected. Most worker wall samples are unresolved libc locations, so the table does not measure a precise worker wait fraction. Inlined work is attributed to its containing symbol. No inline timers or historical monitoring counters were added.")
    remaining=(f"The measured {selected['bps']:,.0f} B/s profile remains {15000/selected['bps']:.2f}x short of 15,000 B/s on this host and export mode. "
      "Next: Action 7, dedicated forward-only runtime; Action 8, resident GPU and independent-stream batching; Action 9, longer 1M/5M/100M qualification, "
      "joint tuning and final time-to-quality selection. This action publishes compact code and evidence without datasets, trained models or temporary snapshots.")
    command="""m = fresh_model(
    profile='epochs', pipeline_workers=2,
    bank_epochs=dict(epoch_bytes=4, lag_epochs=1, storage='sparse'),
    prediction=dict(compact=True, rank=16, slow_stride=8))
# Ordinary run/save calls preserve incomplete epochs.
# m.finish_pipeline() is an explicit commit boundary."""
    def md_table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(row)+' |' for row in rows])
    md=['# SHoEbox Action 6: sparse bank epochs and worker overlap','14 November 2026',
        f"Implementation: [{a.revision[:7]}](https://github.com/JRowe47/SHoEbox/commit/{a.revision}); baseline: `dc3b9b0`.",outcome,decision,
        '## Matched natural-text qualification',md_table(quality_head,quality_rows(n)),protocol,tuning,
        '## Storage and worker selection',md_table(['Profile','B/s','Reserved bank MiB','Held-out NLL'],storage_rows),storage,
        '## 5M storage cross-check',md_table(quality_head,quality_rows(scale)),scale_note,
        '## Repeated-text check',md_table(quality_head,quality_rows(r)),
        '## Implementation']+[f'**{title}.** {text}' for title,text in mechanics]+[
        '```python\nfrom shoebox_reference import fresh_model\n'+command+'\n```','## Validation',verification,
        '## External production sampling',md_table(['Profile','Thread','CPU seconds','Wait samples','Copy/alloc samples','Other/unresolved'],profile_rows),profiling,
        '## Remaining work',remaining,
        'Reproduction: `src/measure_bank_epochs.py`, `src/profile_bank_epochs.py`, `tests/test_bank_epochs.py` and `tests/run_bank_epoch_sanitizer.py`. Full policy: `BANK_EPOCH_POLICY.md`. Compact raw evidence: `results/action6_*`.']
    dest=ROOT/'deliverables';dest.mkdir(exist_ok=True);stem=dest/'SHoEbox_Action6_Sparse_Bank_Epochs'
    stem.with_suffix('.md').write_text('\n\n'.join(md)+'\n')
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='Body6',fontName='Helvetica',fontSize=9.2,leading=12,spaceAfter=7))
    styles.add(ParagraphStyle(name='Small6',fontName='Helvetica',fontSize=8,leading=10,spaceAfter=6))
    story=[]
    def para(t,small=False):story.append(Paragraph(escape(t),styles['Small6' if small else 'Body6']))
    def heading(t):story.append(Paragraph(escape(t),styles['Heading2']))
    def table(head,rows,widths):
        data=[[Paragraph(escape(str(c)),styles['Small6']) for c in row] for row in [head]+rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dbe7ed')),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#7893a3')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d9e1e6')),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story.extend([t,Spacer(1,9)])
    story.append(Paragraph('SHoEbox / Action 6',styles['Title']));para('Sparse bank epochs and worker overlap | 14 November 2026')
    para(f'Implementation {a.revision[:12]} | baseline dc3b9b0',True);para(outcome);para(decision)
    heading('Matched natural-text qualification');table(quality_head,quality_rows(n),[126,35,65,72,92,55]);para(protocol);para(tuning)
    story.append(PageBreak());heading('Storage and worker selection');table(['Profile','B/s','Bank MiB','Held-out NLL'],storage_rows,[165,85,90,105]);para(storage)
    heading('5M storage cross-check');table(quality_head,quality_rows(scale),[126,35,65,72,92,55]);para(scale_note)
    heading('Repeated-text check');table(quality_head,quality_rows(r),[126,35,65,72,92,55]);heading('Validation');para(verification)
    story.append(PageBreak());heading('Implementation and lifecycle')
    for title,text in mechanics:para(title+'. '+text)
    story.append(Preformatted('from shoebox_reference import fresh_model\n'+command,ParagraphStyle(name='Code6',fontName='Courier',fontSize=8,leading=11)))
    story.append(PageBreak());heading('External production sampling');table(['Profile','Thread','CPU s','Wait','Copy/alloc','Other'],profile_rows,[112,95,52,55,65,66]);para(profiling)
    heading('Remaining work');para(remaining);heading('Reproduce and inspect')
    para('Use src/measure_bank_epochs.py for the rotated timing and quality comparisons; src/profile_bank_epochs.py for separate external samples; and tests/test_bank_epochs.py plus tests/run_bank_epoch_sanitizer.py for validation.')
    para('BANK_EPOCH_POLICY.md defines supported combinations and continuation semantics. results/action6_* contains the raw curves, source hashes, memory inventory and test logs. No trained models, data snapshots or compiled binaries are included.')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#526875'));canvas.drawString(44,25,'SHoEbox | Action 6 | 14 November 2026');canvas.drawRightString(A4[0]-44,25,str(doc.page))
    SimpleDocTemplate(str(stem.with_suffix('.pdf')),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=38,bottomMargin=42,
                      title='SHoEbox Action 6: Sparse Bank Epochs').build(story,onFirstPage=footer,onLaterPages=footer)
    print(outcome);print(decision)
if __name__=='__main__':main()
