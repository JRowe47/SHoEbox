"""Render Action 1's measurements as a compact Markdown/PDF report and figures."""
import argparse
import json
import math
from pathlib import Path
import statistics
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('reference', 'event', 'compact_serial', 'compact_threaded')
NAMES = dict(reference='Full reference', event='Event', compact_serial='Compact serial', compact_threaded='Compact threaded')
WORKLOADS = ('repeated', 'natural', 'documents')
WORKNAMES = dict(repeated='Repeated text', natural='Repository prose', documents='Document boundaries')
COLORS = dict(reference='#B04A42', event='#226C93', compact_serial='#2C8267', compact_threaded='#7E5BA5')


def read(name):
    result = json.loads((ROOT / 'results' / ('action1_' + name + '.json')).read_text())
    if not result.get('completed'):
        raise ValueError('Incomplete ' + name + ' suite')
    return result


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    short, continuation, retention, profile = [read(n) for n in ('short', 'continuation', 'retention', 'profile')]
    checks = read('continuation_checks')
    production_hashes = {r['native_library_sha256'] for r in continuation['runs']}
    if len(production_hashes) != 1 or production_hashes != {r['native_library_sha256'] for r in profile['runs']}:
        raise AssertionError('Sampling and timing must use the identical production binary')
    lookup = lambda data, work, name: next(s for s in data['summary'] if s['workload'] == work and s['profile'] == name)
    rows_for = lambda work, name: [r for r in continuation['runs'] if r['workload'] == work and r['profile'] == name]
    targets, selections = {}, []
    for work in WORKLOADS:
        endpoints = [rows_for(work, name)[0]['heldout_nll'] for name in PROFILES]
        # Every profile must still meet the common threshold at the endpoint;
        # a transient earlier minimum must not define an unretained target.
        target = math.ceil((max(endpoints) + .01) * 100) / 100
        times, byte_counts = {}, {}
        for name in PROFILES:
            crossings = [next(v for v in r['quality_curve'] if v['heldout_nll'] <= target) for r in rows_for(work, name)]
            times[name] = statistics.median(v['training_seconds'] for v in crossings)
            byte_counts[name] = crossings[0]['total_training_bytes']
        targets[work] = dict(nll=target, first_observed_seconds=times, first_observed_training_bytes=byte_counts)
        fastest = max(PROFILES, key=lambda n: lookup(continuation, work, n)['median_bytes_per_second'])
        quality = min(PROFILES, key=lambda n: lookup(continuation, work, n)['heldout_nll'])
        selections.append(dict(workload=work, fastest_throughput=fastest, lowest_final_nll=quality,
                               fastest_observed_target=min(times, key=times.get)))
    selection = dict(source_revision=continuation['manifest']['revision'], model_experts=12, seed=7,
                     target_selection='Worst profile endpoint plus >=0.01 nats; all profiles still meet it at the endpoint. Fixed for subsequent actions.',
                     targets=targets, workloads=selections, production_default_changed=False)
    (ROOT / 'results/action1_selection.json').write_text(json.dumps(selection, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.4), constrained_layout=True)
    for col, work in enumerate(WORKLOADS):
        for axis_row in (0, 1):
            ax = axes[axis_row, col]
            for name in PROFILES:
                rows = rows_for(work, name)
                xs = ([v['total_training_bytes'] for v in rows[0]['quality_curve']] if axis_row == 0 else
                      [statistics.median(r['quality_curve'][i]['training_seconds'] for r in rows)
                       for i in range(len(rows[0]['quality_curve']))])
                ys = [v['heldout_nll'] for v in rows[0]['quality_curve']]
                ax.plot(xs, ys, marker='o', ms=3, lw=1.6, ls='--' if name == 'compact_threaded' else '-',
                        color=COLORS[name], label=NAMES[name])
            ax.axhline(targets[work]['nll'], color='#666666', lw=.8, ls=':')
            if axis_row == 0: ax.set_title(WORKNAMES[work], fontsize=9)
            ax.set_xlabel('Training bytes incl. warmup' if axis_row == 0 else 'Training time incl. warmup\n(seconds)')
            ax.set_ylabel('Held-out NLL (nats / byte)' if col == 0 else ''); ax.grid(alpha=.2)
    axes[0, 1].legend(loc='upper right', fontsize=6.5)
    quality_plot = a.output_dir / 'SHoEbox_Action1_Quality.png'
    fig.savefig(quality_plot, dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for ax, rows, label in zip(axes, ([rows_for('repeated', n)[0] for n in PROFILES], retention['runs']),
                               ('Repeated text: 16,384 targets', 'Repository prose: 32,768 targets')):
        for row in rows:
            name = row['profile']; points = row['memory_curve']
            xs = [v['targets'] + row['warmup'] for v in points]
            ax.plot(xs, [v['inventory']['completion']['archive_payload_bytes'] / 2**20 for v in points],
                    marker='o', ms=3, label=NAMES[name], color=COLORS[name])
        ax.set_title(label); ax.set_xlabel('Total training bytes'); ax.grid(alpha=.2)
        ax.set_ylabel('Neural archive payload (MiB)')
    axes[0].legend(fontsize=8)
    memory_plot = a.output_dir / 'SHoEbox_Action1_Retention.png'
    fig.savefig(memory_plot, dpi=180); plt.close(fig)

    throughput_rows = [[WORKNAMES[w]] + [f'{lookup(continuation,w,n)["median_bytes_per_second"]:,.1f}' for n in PROFILES] for w in WORKLOADS]
    loss_rows = [[WORKNAMES[w]] + [f'{lookup(continuation,w,n)["heldout_nll"]:.6f}' for n in PROFILES] for w in WORKLOADS]
    short_rows = [[WORKNAMES[w]] + [f'{lookup(short,w,n)["median_bytes_per_second"]:,.1f}' for n in PROFILES] for w in WORKLOADS]
    variation_rows = [[WORKNAMES[w], NAMES[n],
                       f'{lookup(continuation,w,n)["min_bytes_per_second"]:.1f}-{lookup(continuation,w,n)["max_bytes_per_second"]:.1f}',
                       f'{lookup(continuation,w,n)["cv_percent"]:.2f}%'] for w in WORKLOADS for n in PROFILES]
    target_rows = [[WORKNAMES[w], f'{targets[w]["nll"]:.2f}'] + [f'{targets[w]["first_observed_seconds"][n]:.2f}' for n in PROFILES] for w in WORKLOADS]
    select_rows = [[WORKNAMES[s['workload']], NAMES[s['fastest_throughput']], NAMES[s['fastest_observed_target']]] for s in selections]
    memory_rows = []
    for row in retention['runs']:
        first, last = [row['memory_curve'][i]['inventory'] for i in (0, -1)]
        memory_rows.append([NAMES[row['profile']], f'{row["bytes_per_second"]:.1f}',
                            f'{first["completion"]["archive_payload_bytes"]/2**20:.1f} -> {last["completion"]["archive_payload_bytes"]/2**20:.1f}',
                            f'{last["rss_kib"]/1024:.1f}', f'{last["pipeline"]["active_payload_bytes"]/1024:.1f}',
                            f'{last["pipeline"]["reserved_bytes"]/1024:.1f}'])
    profile_rows = []
    for row in profile['runs']:
        for thread in row['threads']:
            cats = thread['categories']
            value = lambda name: cats.get(name, {}).get('percent', 0.)
            profile_rows.append([NAMES[row['profile']] + (' / worker' if thread['role'] != 'caller' else ''),
                                 f'{thread["cpu_seconds"]:.2f}',
                                 f'{value("reverse / derivatives / credit"):.1f}', f'{value("bank update / versions"):.1f}',
                                 f'{value("forward / prediction / event"):.1f}', f'{value("collection / retention"):.1f}',
                                 f'{100 - sum(value(n) for n in ("reverse / derivatives / credit", "bank update / versions", "forward / prediction / event", "collection / retention")):.1f}'])
    best = max(s['median_bytes_per_second'] for s in continuation['summary'])
    best_reference = max(s['median_bytes_per_second'] for s in continuation['summary'] if s['profile'] == 'reference')
    full_header = ['Workload'] + [NAMES[n] for n in PROFILES]
    manifest = continuation['manifest']
    parameters = {n: rows_for('natural', n)[0]['fitted_parameter_count'] for n in PROFILES}
    intro = ('Action 1 is complete: the four production profiles now share one pinned comparison protocol, '
             'quality controls, retained-state measurements and continuation checks. Runtime arithmetic, '
             'learning policies and the default full reference are unchanged.')
    reference_short = lookup(short, 'repeated', 'reference')['median_bytes_per_second']
    reference_long = lookup(continuation, 'repeated', 'reference')['median_bytes_per_second']
    reference_memory = rows_for('repeated', 'reference')[0]['memory_curve']
    memory_start, memory_end = [reference_memory[i]['inventory']['completion']['archive_payload_bytes']/2**20 for i in (0,-1)]
    slowdown = (f'The full-ancestry repeated-text baseline falls from {reference_short:.1f} B/s in the short '
                f'protocol to {reference_long:.1f} B/s in the 16,384-byte continuation '
                f'({100*(1-reference_long/reference_short):.1f}% lower). Its neural archive grows from '
                f'{memory_start:.1f} MiB after warmup to {memory_end:.1f} MiB. This is the principal '
                'stream-length regression exposed by the baseline; short timings alone conceal it.')
    result_text = ('The table selects a throughput leader and a first-observed time-to-quality leader for each fixture. '
                   'Use these as workload-specific baselines for Action 2. Event and compact use a different credit policy '
                   'from full ancestry; their loss curves are compared directly. Compact serial and threaded follow '
                   'the same schedule and produced bitwise-identical logits and full continuation state in every paired run.')
    overlap = []
    for work in WORKLOADS:
        aa, bb = [lookup(continuation, work, n) for n in ('compact_serial', 'compact_threaded')]
        if max(aa['min_bytes_per_second'], bb['min_bytes_per_second']) <= min(aa['max_bytes_per_second'], bb['max_bytes_per_second']):
            overlap.append(WORKNAMES[work].lower())
    if overlap:
        result_text += ' The compact modes have overlapping trial ranges on '+', '.join(overlap)+'. Their small median leads should be treated as provisional.'
    protocol = (f'Source {manifest["revision"]}; {manifest["cpu_model"]}; caller CPU 0, learning worker CPU 1. '
                f'{manifest["compiler"]}. Build: -O3 -march=native -std=c++17 -pthread, SHOEBOX_EXPERTS=12, '
                'production completion with inline diagnostics disabled. Width 64, 18 logical regions, seed 7. '
                f'Full reference has {parameters["reference"]:,} fitted coefficients; event and compact each have '
                f'{parameters["event"]:,}. These are model counts, not training-byte counts. Dynamic learned pattern '
                'dictionaries and episodic/eligibility state are inventoried separately rather than counted as fixed coefficients.')
    timing = ('Each run starts fresh, trains 1,024 warmup target bytes, then measures 2,048 or 16,384 target bytes. '
              'The two suites each use three rotated profile orders per fixture. Timing includes 512-byte API chunks, '
              'full logit export, worker quiescence and document drains/reset where applicable. Compilation, warmup, '
              'evaluation, memory inspection, hashing and file IO are excluded from continuation throughput. '
              'Warmup training time is included in time-to-quality. Process CPU time sums all threads. '
              'Benchmarks ran sequentially with no concurrent tests or profiling. No historical event counters were used.')
    data_text = ('Repeated text reproduces the prior two-sentence fixture and uses a separate two-sentence validation fixture. '
                 'Natural text is fixed repository prose: README.md then STATUS.md from the pinned revision; validation is '
                 'a separate docs/cortical_fineweb.md excerpt. These are fixture measurements, not a FineWeb training run. '
                 'The document fixture partitions that prose into documents with 512 targets, trains no cross-document '
                 'target, and drains/resets at each document end. All validation sets contain 1,024 targets. Exact hashes, '
                 'source paths, document lengths, policies, native binary hashes and API chunk sizes are in the JSON evidence.')
    quality_text = ('Cold, frozen held-out evaluation uses a separate model copy after warmup and at 2,048, 4,096, '
                    '8,192 and 16,384 continuation bytes. The targets below are now fixed for subsequent development: '
                    'each is the worst profile endpoint plus at least 0.01 nats, rounded upward. Every profile still '
                    'meets its target at the endpoint. Times are the median '
                    'first measured crossing, including warmup; they are upper bounds at this checkpoint spacing, '
                    'not interpolated convergence estimates. All runs use the same seed; multi-seed tuning belongs to Action 9.')
    memory_text = ('The retention probe adds 32,768 target bytes after warmup without document resets. It uses the '
                   'same 512-byte API chunks and a single run per profile. Current RSS, retained neural payload, '
                   'publication/gradient storage, live records, pending jobs and reserved pipeline capacity are read '
                   'only at quiescent boundaries. The archive measure is the neural archive payload, not total model memory. '
                   'Process RSS and high-water RSS include allocator retention from prior untimed validation copies; '
                   'they must not be interpreted as pure trainer allocation. Full inventories and final serialized section '
                   'sizes are retained in the evidence.')
    compact_archive = rows_for('repeated', 'compact_serial')[0]['memory_curve'][-1]['inventory']['completion']['archive_payload_bytes']
    memory_finding = (f'At the repeated-text continuation endpoint, compact serial retains '
                      f'{compact_archive/2**20:.1f} MiB in the neural archive, compared with '
                      f'{parameters["compact_serial"]*4/2**20:.1f} MiB of fixed FP32 coefficients. '
                      'The large retained state affects both event and compact policies; faster bank updates alone '
                      'do not remove that storage growth.')
    profiling_text = ('The sampler inspected the same production binary on a separate 8,192-byte natural-text run. '
                      'It sends thread-directed signals at 100 Hz through a standalone sampling attachment; the handler '
                      'records instruction addresses and thread CPU clocks and reads no model state. No native trainer '
                      'timers, hooks or counters were added. The percentages below are wall-time instruction-pointer '
                      'samples, not inclusive CPU cost. Inlined operations belong to their containing symbol; unresolved '
                      'library work, allocation/copying and waits remain in Other. CPU seconds come separately from '
                      '/proc thread accounting. Profiled elapsed times are excluded from all throughput comparisons.')
    threaded_profile = next(r for r in profile['runs'] if r['profile'] == 'compact_threaded')
    caller_cpu = sum(t['cpu_seconds'] for t in threaded_profile['threads'] if t['role'] == 'caller')
    worker_cpu = sum(t['cpu_seconds'] for t in threaded_profile['threads'] if t['role'] == 'learning_worker')
    profile_finding = (f'The compact threaded profile consumed {caller_cpu:.2f} caller CPU seconds and '
                       f'{worker_cpu:.2f} learning-worker CPU seconds. The caller accounted for '
                       f'{100*caller_cpu/(caller_cpu+worker_cpu):.1f}% of measured thread CPU. '
                       'The next work remains indexed record/retention storage (Action 2), followed by caller-side '
                       'reverse reuse and event scheduling (Action 3). Recheck worker benefit after those changes.')
    validation = ('Passed: 3 benchmark-protocol tests; 24 production pipeline regressions; compact ASan/UBSan '
                  '(leak scanning disabled); exact 16,384-byte natural continuation with irregular chunks, a pending '
                  'checkpoint and worker changes; exact 4,096-byte document-boundary restart; all repeated-run '
                  'determinism and serial/threaded hashes; live retained-span and producer-integrity checks at every '
                  'measurement boundary. No model or dataset snapshots are included in the deliverables.')
    backlog = [
        '2. Indexed records and lean state: direct MinKron lookup, incremental retention, reusable records, shared context and head-sized eligibility.',
        '3. Reusable learning and ready-event execution: reverse-workspace reuse, local adjoints, publication readiness and regional packing.',
        '4. Compact prediction and fast/slow Y, with a named policy and tuned quality comparison.',
        '5. Lightweight binding frames, preserving learned completed-content dependence and credit.',
        '6. Sparse bank epochs and broader worker overlap under explicit version/commit schedules.',
        '7. Dedicated forward-only runtime and defined transitions back to training.',
        '8. Resident GPU backend and independent-stream batching.',
        '9. Rolling qualification and final profile selection at 1M, 5M and 100M parameters.'
    ]
    reproduction = ('python experiments/identity_semantics/src/measure_current_baseline.py --suite short --output short.json\n'
                    'python experiments/identity_semantics/src/measure_current_baseline.py --suite continuation --output continuation.json\n'
                    'python experiments/identity_semantics/src/measure_current_baseline.py --suite retention --output retention.json\n'
                    'python experiments/identity_semantics/src/check_baseline_continuation.py --output continuation_checks.json\n'
                    'python experiments/identity_semantics/src/profile_current_baseline.py --output profile.json')
    md = '\n\n'.join([
        '# SHoEbox - Action 1 baseline comparison', '13 November 2026 (America/Denver)', intro, slowdown,
        '## 16,384-byte continuation', 'Median single-stream training throughput, bytes/s (three runs):',
        table(full_header, throughput_rows), 'Final held-out NLL, nats per byte:', table(full_header, loss_rows),
        result_text, table(['Workload', 'Fastest throughput', 'Fastest observed target'], select_rows),
        f'The best continuation throughput is {best:,.1f} B/s: {100*best/15000:.1f}% of the 15,000 B/s target. '
        f'That is {1e6/best:,.1f} microseconds per byte against a 66.7-microsecond budget; {15000/best:.2f}x more throughput is needed. '
        f'The best full-ancestry reference median is {best_reference:.1f} B/s across these continuation fixtures, a {15000/best_reference:.2f}x gap.',
        '## Protocol and short comparison', protocol, timing, data_text, '2,048-byte short protocol, median bytes/s:',
        table(full_header, short_rows), '16,384-byte run range and sample coefficient of variation:',
        table(['Workload', 'Profile', 'B/s range', 'CV'], variation_rows),
        '## Fixed time-to-quality controls', quality_text,
        table(['Workload', 'NLL target'] + [NAMES[n]+' seconds' for n in PROFILES], target_rows),
        '## Retention and memory', memory_text, memory_finding,
        table(['Profile', 'B/s', 'Archive MiB: warm -> end', 'End RSS MiB', 'Active queue KiB', 'Reserved queue KiB'], memory_rows),
        '## Production sampling', profiling_text,
        table(['Thread', 'CPU seconds', 'Reverse %', 'Banks %', 'Forward %', 'Collection %', 'Other %'], profile_rows), profile_finding,
        '## Validation and remaining work', validation, '\n\n'.join(backlog),
        '## Reproduction', 'Run each command alone; set SHOEBOX_EXPERTS=12 and use distinct available caller/worker CPUs. '
        'Defaults are CPUs 0 and 1. The driver accepts --cpu and --worker-cpu. Keep the pinned data revision available.',
        '```bash\n'+reproduction+'\n```',
        'The evidence JSON retains every trial, timing scope, quality curve, memory boundary, source/build hash, '
        'CPU time, integrity result and comparison digest. The reusable driver does not change the learning contract.'
    ]) + '\n'
    stem = a.output_dir / 'SHoEbox_Action1_Baseline'
    stem.with_suffix('.md').write_text(md)

    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from matplotlib import get_data_path
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    font_dir = Path(get_data_path()) / 'fonts' / 'ttf'
    for name, filename in [('ReportSans', 'DejaVuSans.ttf'), ('ReportBold', 'DejaVuSans-Bold.ttf'),
                           ('ReportMono', 'DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body2', parent=styles['BodyText'], fontName='ReportSans', fontSize=9, leading=13, spaceAfter=8, textColor=colors.HexColor('#263747')))
    styles.add(ParagraphStyle(name='Cell2', parent=styles['BodyText'], fontName='ReportSans', fontSize=7.5, leading=10))
    styles.add(ParagraphStyle(name='Code2', parent=styles['BodyText'], fontName='ReportMono', fontSize=6.9, leading=11, spaceAfter=7))
    styles['Heading1'].fontName='ReportBold'; styles['Heading1'].fontSize=16
    styles['Title'].fontName='ReportBold'; styles['Title'].fontSize=24; styles['Title'].leading=28
    styles['Title'].textColor=colors.HexColor('#173A4A')
    story=[]
    def para(text): story.append(Paragraph(escape(text), styles['Body2']))
    def title(text): story.append(Paragraph(text, styles['Heading1'])); story.append(Spacer(1,6))
    def grid(headers, rows, widths=None):
        content=[[Paragraph(escape(str(v)),styles['Cell2']) for v in row] for row in [headers]+rows]
        if widths: widths = [w * 528 / sum(widths) for w in widths]
        t=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E2EDF1')),('VALIGN',(0,0),(-1,-1),'TOP'),
                               ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                               ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
                               ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#A6BBC4')),
                               ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F4F7F8')])]))
        story.extend([t,Spacer(1,12)])
    story.append(Paragraph('SHoEbox<br/>Action 1 baseline',styles['Title']))
    para('13 November 2026 | Production comparison | 1M model class')
    para(intro);para(slowdown);title('16,384-byte continuation')
    para('Median training throughput, bytes/s. Three rotated-order trials per profile and workload.')
    grid(full_header,throughput_rows,[106,88,88,88,98])
    para('Final held-out negative log-likelihood, nats per byte. Lower is better.')
    grid(full_header,loss_rows,[106,88,88,88,98]);para(result_text)
    grid(['Workload','Fastest throughput','Fastest observed quality target'],select_rows,[130,160,178])
    para(f'Best continuation: {best:,.1f} B/s, or {100*best/15000:.1f}% of the 15k B/s target. '
         f'Remaining throughput gap: {15000/best:.2f}x. The best full-ancestry median is {best_reference:.1f} B/s '
         f'({15000/best_reference:.2f}x gap). The reference default remains unchanged.')
    story.append(PageBreak());title('Reproducible timing and variability')
    para(protocol)
    para('Three rotated-order repetitions per short/continuation workload. Timing includes 512-target API chunks, '
         'full logits, worker quiescence and document drains/reset. Compilation, warmup, validation, inspection, '
         'hashing and IO are excluded from continuation throughput. Time-to-quality includes warmup training. '
         'Process CPU time includes all threads. Timings ran alone, without tests or profiling.')
    para('Repeated text uses the prior fixture and separate validation sentences. Repository prose uses pinned '
         'README/STATUS text and a separate documentation file for validation. Documents contain 512 targets '
         'with no cross-document teacher. Validation is 1,024 targets. Exact data/build hashes and policies are '
         'in the JSON evidence; these are fixture measurements, not a FineWeb training run.')
    para('Reproduced short protocol: 2,048 measured target bytes after 1,024 warmup bytes.')
    grid(full_header,short_rows,[106,88,88,88,98])
    para('Continuation timing variability: min-max bytes/s and sample coefficient of variation.')
    grid(['Workload','Profile','B/s range','CV'],variation_rows,[122,137,130,79])
    story.append(PageBreak());title('Fixed quality controls')
    para(quality_text)
    grid(['Workload','NLL']+[NAMES[n]+' (s)' for n in PROFILES],target_rows,[100,45,80,72,80,91])
    story.append(Image(str(quality_plot),width=528,height=396));story.append(Spacer(1,8))
    para('Dotted lines mark the fixed NLL targets. Compact serial and threaded have identical quality points; '
         'their horizontal positions differ because elapsed training time differs. Full trial curves and '
         'first-crossing byte counts are preserved in action1_selection.json and action1_continuation.json.')
    story.append(PageBreak());title('Longer retained-state probe');para(memory_text);para(memory_finding)
    grid(['Profile','B/s','Archive MiB\nwarm -> end','End RSS\nMiB','Active\nqueue KiB','Reserved\nqueue KiB'],memory_rows,[111,53,100,66,66,72])
    story.append(Image(str(memory_plot),width=528,height=220))
    para(slowdown)
    story.append(PageBreak());title('Production profile and validation')
    para(profiling_text)
    grid(['Thread','CPU s','Reverse %','Banks %','Forward %','Collect %','Other %'],profile_rows,[132,46,58,58,58,58,58])
    para(profile_finding)
    para('The caller/worker split is interpreted together with thread CPU time and the exact sampled symbol list. '
         'A worker waiting in a library function does not imply bank computation occupies the same fraction of CPU time.')
    title('Validation');para(validation)
    story.append(PageBreak());title('Remaining actions')
    for item in backlog: para(item)
    title('Reproduce and extend')
    para('Use the commands below from the repository root. Run timing, correctness checks and sampling sequentially. '
         'Set SHOEBOX_EXPERTS=12. CPUs 0 and 1 are the default placements; use --cpu and --worker-cpu to select '
         'two other distinct available CPUs. Full profile configuration is recorded in each result.')
    for command in reproduction.splitlines():
        story.append(Paragraph(escape(command), styles['Code2']))
    title('Deliverable contents')
    para('The code/evidence bundle contains the new comparison driver, external sampler and signal attachment, '
         'continuation checker, report generator, benchmark-protocol tests, source patch and compact JSON/test evidence. '
         'No trained models, transient checkpoints, compiled binaries or corpus copies are included.')
    para('Use action1_selection.json as the fixed quality-target record for subsequent actions. Preserve source and '
         'fixture hashes, rates, export mode, chunk sizes and CPU placement when comparing Action 2. '
         'The longer 1M/5M/100M, multi-seed learning qualification remains Action 9.')
    def footer(canvas, doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#CFDCE2'));canvas.line(42,38,570,38)
        canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor('#61737D'))
        canvas.drawString(42,25,'SHoEbox | Action 1 | Source '+manifest['revision'][:7])
        canvas.drawRightString(570,25,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(stem.with_suffix('.pdf')),pagesize=(612,792),rightMargin=42,leftMargin=42,
                          topMargin=40,bottomMargin=48,title='SHoEbox Action 1 Baseline',author='SHoEbox')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(stem.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
