"""Export the measured widened-port profile as ordinary downloadable files."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak


def markdown_table(headers, rows):
    return ('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+
            ''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    read = lambda name:json.loads((args.run/name).read_text())
    s,ref,opt,comparison = [read(name) for name in ('results.json','benchmark_reference.json',
                                                  'benchmark_optimized.json','benchmark_comparison.json')]
    operators = read('operator_comparison.json')
    audit = read('endpoint_audit.json')
    assert s['complete'] and s['trained_targets']==1000 and audit['passed']
    args.out.mkdir(parents=True,exist_ok=True)
    stem = args.out/'SHoEbox_Cortical_Wide_1K_Profile'
    a,z = s['initial_probe'],s['final_probe']
    h = s['history']
    early = sum(s['allocation'][name]['parameters'] for name in ('A1','B1','L2','S2','R2','F2'))
    speedup = opt['bytes_per_second']/ref['bytes_per_second']
    reduction = 100*(1-opt['peak_rss_mib']/ref['peak_rss_mib'])
    config_headers = ['Setting','Previous 10M','Wide profile']
    config_rows = [
        ['Total parameters','9,982,290',f"{s['parameters']:,}"],
        ['Context / compact / primary ports','1 / 2 / 4 bytes','4 / 8 / 16 bytes'],
        ['Region widths: orders 1 / 2 / later','64 / 64 / 64','128 / 96 / 64'],
        ['Expert hidden widths: 1 / 2 / later','16 / 16 / 16','32 / 24 / 16'],
        ['Experts selected per layer','1 of 64','4 of 32'],
        ['HAT dictionary: early / later','64 / 64','256 / 128'],
        ['Active HAT symbols: early / later','2 / 2','8 / 4'],
        ['Gradient horizon','2 observations','32 observations']]
    perf_headers = ['Matched 128-target batch','Reference','Optimized']
    perf_rows = [['Forward + backward',f"{ref['forward_seconds']+ref['backward_seconds']:.2f} s",
                  f"{opt['forward_seconds']+opt['backward_seconds']:.2f} s"],
                 ['Throughput',f"{ref['bytes_per_second']:.2f} bytes/s",f"{opt['bytes_per_second']:.2f} bytes/s"],
                 ['Peak process RSS',f"{ref['peak_rss_mib']/1024:.2f} GiB",f"{opt['peak_rss_mib']/1024:.2f} GiB"],
                 ['Unique storage referenced by saves',f"{ref['saved_unique_storage_bytes']/2**30:.2f} GiB",
                  f"{opt['saved_unique_storage_bytes']/2**30:.2f} GiB"]]
    port_headers = ['Port','Soft H: before → after','JS: before → after','Unique frames: before → after','Final largest share']
    port_rows = []
    for key,v in z['ports'].items():
        before = a['ports'][key]
        port_rows.append([key.replace(':',' / '),
                          f"{before['conditional_entropy_bits']:.3f} → {v['conditional_entropy_bits']:.3f}",
                          f"{before['conditioning_js_bits']:.5f} → {v['conditioning_js_bits']:.5f}",
                          f"{before['unique_complete_frames']} → {v['unique_complete_frames']}",
                          f"{100*v['most_common_frame_fraction']:.1f}%"])
    old_headers = ['Primary port','Old 4-byte unique frames','New 16-byte unique frames']
    old_rows = [[name,s['old_100k_probe']['ports'][name+':4']['unique_complete_frames'],
                 z['ports'][name+':16']['unique_complete_frames']] for name in ('A1','B1','L2')]
    lag_headers = ['Region','Lag 0 RMS','Lag 2 RMS','Lag 8 RMS','Lag 16 RMS','Lag 31 RMS']
    lag_rows = []
    for name in ('A1','B1','L2'):
        values={r['lag']:r['gradient_rms'] for r in s['temporal_gradients'] if r['region']==name}
        lag_rows.append([name,*[f'{values[lag]:.2e}' for lag in (0,2,8,16,31)]])
    hotspot_headers = ['Operator','Calls','Self CPU','Share','Positive self allocation']
    hotspot_rows = [[row['operator'],f"{row['calls']:,}",f"{row['self_cpu_seconds']:.4f} s",
                     f"{100*row['self_cpu_seconds']/operators['optimized_total_self_cpu_seconds']:.1f}%",
                     f"{row['positive_self_allocated_bytes']/2**20:.1f} MiB"]
                    for row in operators['optimized_top_self'][:6]]
    interpretation = read('interpretation.json')
    summary = (f"Built {s['parameters']:,} parameters and completed exactly 1,000 FineWeb byte targets in eight updates. "
               f"The fixed training-slice probe fell from {a['nll']:.4f} to {z['nll']:.4f} nats/byte. "
               f"Matched execution is {100*(speedup-1):.1f}% faster, with {reduction:.1f}% lower peak RSS.")
    allocation = (f"Orders 1 and 2 hold {100*early/s['parameters']:.1f}% of parameters. Fewer dormant experts fund wider "
                  "active paths and independent output heads. Active expert matrix capacity increases 16x / 9x / 4x "
                  "at orders 1 / 2 / later. External observations and final emitted characters stay single bytes.")
    temporal = ("One 16-byte frame is one publication, not sixteen historical observations. Fourfold wider incoming "
                "ports receive fourfold slower initial HAT decay rates; B1 keeps its slower clock. Rates remain learned. "
                "The optimizer warms up over 256 targets, then holds 0.001 rather than tapering during cold start.")
    runtime = (f"Training forward/backward/update time: {s['training_seconds']:.2f} s "
               f"({s['training_bytes_per_second']:.2f} bytes/s); the extra temporal-gradient audit took "
               f"{s['temporal_diagnostic_seconds']:.2f} s. Monitor work totals "
               f"{sum(r['monitor_seconds'] for r in h):.2f} s and is included. Fixed probes and separate profiles are additional.")
    memory = ("The first instrumented attempt hit the memory limit after 640 targets: its saved-tensor hook retained "
              "autograd tensor owners. Detached storage aliases fixed that instrumentation leak, and the run restarted "
              "from the original seed. The completed checkpoint was trained on exactly 1,000 scored targets in eight optimizer steps. "
              f"Corrected forward RSS settles at {max(r['rss_after_forward_mib'] for r in h[3:])/1024:.2f} GiB. "
              f"Boundary compaction reduces carried state from {h[-1]['state_bytes_before_compaction']:,} to "
              f"{h[-1]['state_bytes_after_compaction']:,} bytes without changing its forward values.")
    optimizations = ("Implemented: algebraic metadata projection before broadcasting; per-forward codebook/router/lace "
                     "caches; terminal HAT reads with all recurrent writes intact; private-bank dispatch; removal of unused "
                     "lace work; shared diagnostic softmax; and exact detached-frame compaction. The new preset balances "
                     "published HAT symbols only. Reference and optimized execution share this objective.")
    checks = (f"All {audit['tests_passed']} checks passed, including FP64 value/state/gradient equivalence, heterogeneous "
              "regional dimensions, temporal gradient connectivity, checkpoint reload, entropy sanity and hook ownership. "
              f"Matched FP32 maximum logit difference: {comparison['max_logit_difference']:.3g}; greedy predictions match.")
    commands = ('python -m shoebox.profile_wide --mode train --backend optimized \\\n'
                '  --data artifacts/cortical_wide_profile/profile_train.bin \\\n'
                '  --out artifacts/wide_reproduction --targets 1000 --batch 4 --context 32 \\\n'
                '  --threads 1 --lr 0.001 --warmup-targets 256\n')
    text = '# Cortical SHoEbox: wider ports / 1,000-byte profile\n\nNovember 8, 2026\n\n'+summary+'\n\n'
    text += '## Reconfiguration\n\n'+markdown_table(config_headers,config_rows)+'\n'+allocation+'\n\n'+temporal+'\n\n'
    text += ('## Measured runtime and memory\n\n'+markdown_table(perf_headers,perf_rows)+'\n'+runtime+'\n\n'+memory+'\n\n'+
             'Both backends use B=4, T=32, the same initialized weights, and eight warmup observations. '
             'Benchmarks run in separate processes; no optimizer update. Saved storage is deduplicated referenced storage, '
             'not simultaneous RSS. The earlier 100k run used B=200/T=2 and is not a matched throughput baseline.\n\n')
    text += '## What the ports and temporal gradients show\n\n'+interpretation['ports']+'\n\n'
    text += markdown_table(port_headers,port_rows)+'\n'
    text += ('Each probe contains 128 publications per port. H and JS are bits per byte (maximum H=8). '
             'JS = entropy of the mean distribution minus mean conditional entropy. Unique counts concern complete frames. '
             'The before/after probes use identical text and reset state; these are training-slice diagnostics.\n\n')
    text += markdown_table(old_headers,old_rows)+'\nThe old checkpoint has 100k targets of training; the new one has 1k. '
    text += 'These are diagnostic code-usage observations, not a controlled model-quality comparison.\n\n'
    text += markdown_table(lag_headers,lag_rows)+'\n'+interpretation['temporal']+'\n\n'
    text += 'The temporal audit uses only the last prediction’s task loss at initialization; no auxiliary loss.\n\n'
    text += '## Operator costs and next work\n\n'+optimizations+'\n\n'+markdown_table(hotspot_headers,hotspot_rows)+'\n'
    text += interpretation['hotspots']+'\n\nOperator timings cover one observation per four streams at the final checkpoint, '
    text += 'including backward and profiler overhead. Allocation is cumulative traffic, not memory retained.\n\n'+checks+'\n\n'
    text += '## Reproduce\n\n```bash\n'+commands+'```\n\n'
    text += ('The package includes the exact 1,004 input bytes (1,000 scored targets plus four prefixes), checkpoint, '
             'optimizer/state/RNG, topology, JSONL logs, probes, operator profiles and source. '
             'See docs/cortical_wide_profile.md for all benchmark/profile commands.\n\n')
    text += f"Runtime: PyTorch {s['torch']}, Python {s['python']}, {s['threads']} CPU thread, AMD EPYC 9V74.\n\n"
    text += f"Training source SHA-256: `{s['source_sha256']}`.\n\nData SHA-256: `{s['data_sha256']}`.\n\n"
    text += 'Source dataset: [FineWeb, sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb).\n'
    stem.with_suffix('.md').write_text(text)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    xs=[r['trained_targets'] for r in h]
    axes[0,0].plot(xs,[r['nll'] for r in h],marker='o',color='#0d9488',label='Incoming training windows')
    axes[0,0].plot([0,1000],[a['nll'],z['nll']],marker='s',ls='--',color='#b45309',label='Fixed training-slice probe')
    axes[0,0].set(title='Cold-start loss',xlabel='Trained byte targets',ylabel='Nats / byte')
    axes[0,0].legend(fontsize=8)
    axes[0,1].plot(xs,[r['rss_after_forward_mib']/1024 for r in h],marker='o',color='#2563eb',label='Corrected forward RSS')
    failed=[json.loads(line) for line in (args.run/'failed_instrumented_attempt/history.jsonl').read_text().splitlines()]
    axes[0,1].plot([r['trained_targets'] for r in failed],[r['rss_after_forward_mib']/1024 for r in failed],
                   marker='x',ls='--',color='#b91c1c',label='Leaking instrumentation')
    axes[0,1].set(title='Memory across windows',xlabel='Trained byte targets',ylabel='GiB process RSS')
    axes[0,1].legend(fontsize=8)
    for name,color in [('A1','#0d9488'),('B1','#2563eb'),('L2','#b45309')]:
        rows=sorted([r for r in s['temporal_gradients'] if r['region']==name],key=lambda r:r['lag'])
        axes[1,0].semilogy([r['lag'] for r in rows],[max(r['gradient_rms'],1e-38) for r in rows],
                           marker='o',label=name,color=color)
    axes[1,0].set(title='Task gradient to earlier regional latents',xlabel='Observation lag',ylabel='Gradient RMS at initialization')
    axes[1,0].legend(fontsize=8)
    names=['A1','B1','L2']
    axes[1,1].bar([-.2,.8,1.8],[a['ports'][n+':16']['unique_complete_frames'] for n in names],.4,label='Wide, before')
    axes[1,1].bar([.2,1.2,2.2],[z['ports'][n+':16']['unique_complete_frames'] for n in names],.4,label='Wide, after 1k')
    axes[1,1].set(xticks=range(3),xticklabels=names,ylim=(0,135),title='Complete 16-byte frame diversity',ylabel='Unique frames / 128 publications')
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.16,axis='y')
    plot=args.out/'SHoEbox_Cortical_Wide_1K_Profile.png'
    fig.savefig(plot,dpi=180);plt.close(fig)

    for name,family,weight in [('DV','DejaVu Sans','normal'),('DV-Bold','DejaVu Sans','bold'),('DV-Mono','DejaVu Sans Mono','normal')]:
        pdfmetrics.registerFont(TTFont(name,findfont(FontProperties(family=family,weight=weight))))
    styles=getSampleStyleSheet()
    styles['Title'].fontName=styles['Heading2'].fontName='DV-Bold'
    styles.add(ParagraphStyle(name='Small',fontName='DV',fontSize=8.6,leading=11.5,spaceAfter=7))
    styles.add(ParagraphStyle(name='CellSmall',fontName='DV',fontSize=7.4,leading=9.5))
    styles.add(ParagraphStyle(name='MonoSmall',fontName='DV-Mono',fontSize=6.8,leading=9,spaceAfter=5))
    def para(value,style='Small'):return Paragraph(escape(str(value)),styles[style])
    def table(headers,rows,widths):
        result=Table([[para(v,'CellSmall') for v in row] for row in [headers,*rows]],colWidths=widths,repeatRows=1)
        result.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dbeafe')),
                                   ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f5f9'),colors.white]),
                                   ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                                   ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        return result
    flow=[para('Cortical SHoEbox','Title'),para('Wider transport / 1,000-byte FineWeb profile | November 8, 2026'),
          para(summary),table(config_headers,config_rows,[226,150,150]),Spacer(1,10),para(allocation),para(temporal),
          para('Matched execution','Heading2'),table(perf_headers,perf_rows,[266,130,130]),Spacer(1,8),para(runtime),
          para('Separate processes; same B=4/T=32 and weights. Saved storage is referenced storage, not simultaneous RSS.'),
          PageBreak(),para('Port usage and temporal credit','Heading2'),para(interpretation['ports']),
          table(port_headers,port_rows,[56,124,124,126,96]),Spacer(1,8),
          para('128 publications per port. Entropy and JS are bits per byte. JS measures changing soft distributions; '
               'unique counts and largest share describe the hard complete frames. Identical training-slice probes, reset state.'),
          table(old_headers,old_rows,[126,200,200]),Spacer(1,6),
          para('Old checkpoint: 100k trained targets. New checkpoint: 1k. This is code-usage diagnosis, not a controlled quality comparison.'),
          table(lag_headers,lag_rows,[56,94,94,94,94,94]),Spacer(1,6),para(interpretation['temporal']),
          para('Gradient audit at initialization: last prediction task loss alone, with no auxiliary loss.'),
          PageBreak(),para('Runtime findings','Heading2'),
          Image(str(plot),width=526,height=334.7),Spacer(1,7),para(memory),para(optimizations),
          table(hotspot_headers,hotspot_rows,[156,55,70,55,190]),Spacer(1,6),para(interpretation['hotspots']),
          PageBreak(),para('Validation and reproduction','Heading2'),para(checks),
          para('The operator profile covers one observation per stream at the final checkpoint, including backward and '
               'profiling overhead. Allocations are cumulative traffic, not live memory.'),
          para('Reproduction command','Heading2')]
    for line in commands.splitlines():flow.append(para(line,'MonoSmall'))
    flow += [para('Use a fresh output directory. The package contains exact input bytes, checkpoint, optimizer, runtime state, '
                  'RNG and logs. Full benchmark commands and the allocation rationale are in docs/cortical_wide_profile.md.'),
             para('The 32-observation training horizon is independent of MinKron’s internal computation chunks, which retain '
                  'their gradient connections. The profile runner does not resume an unfinished run.'),
             para(f"Runtime: PyTorch {s['torch']}; Python {s['python']}; one CPU thread; AMD EPYC 9V74."),
             para('Training source SHA-256'),para(s['source_sha256'],'MonoSmall'),
             para('Input data SHA-256'),para(s['data_sha256'],'MonoSmall'),
             para('Data: huggingface.co/datasets/HuggingFaceFW/fineweb, sample-10BT. The exact bytes are packaged.')]
    def footer(canvas,doc):
        canvas.setFont('DV',8);canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(43,22,'SHoEbox | measured wide-port profile')
        canvas.drawRightString(569,22,str(doc.page))
    SimpleDocTemplate(str(stem.with_suffix('.pdf')),pagesize=(612,792),leftMargin=43,rightMargin=43,
                      topMargin=37,bottomMargin=37,title='Cortical SHoEbox: wider ports / 1K profile',
                      author='Joshua Rowe / Codex').build(flow,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps({'markdown':str(stem.with_suffix('.md')),'pdf':str(stem.with_suffix('.pdf')),'plot':str(plot)}))


if __name__=='__main__':main()
