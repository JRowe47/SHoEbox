"""Create concise Markdown/PDF evidence for the MinKron fusion run."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,findfont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,Image,PageBreak


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    s=json.loads((args.run/'fusion_summary.json').read_text())
    training=json.loads((args.run/'results.json').read_text())
    assert s['complete'] and s['tests_passed']==23 and s['endpoint_reload_exact']
    a,b=s['matched']['baseline'],s['matched']['fused']
    args.out.mkdir(parents=True,exist_ok=True)
    stem=args.out/'SHoEbox_MinKron_Fusion_Report'
    title='SHoEbox: MinKron fusion'
    lead=(f"Implemented a native CPU recurrence and analytic backward for the wide cortical model. "
          f"The matched 128-target batch is {100*(s['matched']['speedup']-1):.1f}% faster than the previous optimized "
          f"path and uses {s['matched']['peak_rss_reduction_percent']:.1f}% less peak process memory.")
    headers=['Metric','Previous optimized','Fused MinKron']
    rows=[['Forward',f"{a['forward_seconds']:.2f} s",f"{b['forward_seconds']:.2f} s"],
          ['Backward',f"{a['backward_seconds']:.2f} s",f"{b['backward_seconds']:.2f} s"],
          ['Forward + backward',f"{a['forward_seconds']+a['backward_seconds']:.2f} s",f"{b['forward_seconds']+b['backward_seconds']:.2f} s"],
          ['Throughput',f"{a['bytes_per_second']:.2f} bytes/s",f"{b['bytes_per_second']:.2f} bytes/s"],
          ['Peak process RSS',f"{a['peak_rss_mib']/1024:.2f} GiB",f"{b['peak_rss_mib']/1024:.2f} GiB"],
          ['Unique storage referenced by saves',f"{a['saved_unique_storage_bytes']/2**30:.2f} GiB",f"{b['saved_unique_storage_bytes']/2**30:.2f} GiB"],
          ['Saved-tensor calls',f"{a['saved_tensor_calls']:,}",f"{b['saved_tensor_calls']:,}"]]
    protocol=('Both paths: 14,976,882 parameters, four streams, 32-observation live gradient window, identical '
              'initialized weights and data, eight warmup observations, one CPU thread, separate processes. '
              'Compilation and warmup are excluded. Saved storage is deduplicated referenced storage, not peak RSS.')
    fit=(f"The fused run completed exactly 1,000 FineWeb byte targets in eight updates, taking "
         f"{s['training_seconds']:.2f} seconds for forward/backward/optimizer work "
         f"({s['training_bytes_per_second']:.2f} bytes/s). The separate temporal-gradient audit took "
         f"{training['temporal_diagnostic_seconds']:.2f} seconds. Training peak RSS was "
         f"{s['training_peak_rss_mib']/1024:.2f} GiB; carried state remained {s['state_bytes']:,} bytes. "
         f"The fixed training-slice probe changed from {s['initial_probe_nll']:.6f} to {s['final_probe_nll']:.6f} nats/byte.")
    change=('The HAT now scans the original MinKron rank-one update directly in compiled C++ instead of forming '
            'pairwise time matrices and solving a triangular system. The backward pass differentiates those '
            'updates analytically in reverse order. The three single-step MinKron cells in every regional '
            'SHoE also use the fused recurrence. Projections, direction/read normalization, routing and heads '
            'remain PyTorch operations.')
    memory=('The state tape grows linearly with incoming sequence length: O(BTHKV), with no T-by-T recurrence '
            'intermediates. Inference allocates no tape. Final states own compact storage, so detaching at '
            'the TBPTT boundary releases history. Model dimensions, ports, optimizer schedule and the training '
            'horizon are unchanged.')
    checks=('All 23 tests pass. Checks include FP64 outputs, recurrent states and every input/parameter gradient '
             'against the original recurrence, finite-difference gradcheck, zero and clamped directions, '
             '65-position sequences spanning original chunks, multiple calls with live state gradients, '
             'heterogeneous cortical regions and portable fallback. The endpoint reload reproduces its probe exactly. '
             'A separate one-thread/two-thread audit gives bit-identical values and gradients.')
    numeric=(f"The matched FP32 batch has maximum logit difference {s['matched']['max_logit_difference']:.3g} "
              'and identical greedy predictions. The fused scan changes floating-point evaluation order; '
              'training is not bit-identical through discrete routing. The previous optimized 1k run scored '
              '5.060832 on the same training probe; the fused run scores 5.066980.')
    op_a,op_b=s['optimized_operators'],s['fused_operators']
    traffic=('The operator profile, using one observation per four streams at the same final checkpoint, '
             f"reduces cumulative positive allocation traffic from {op_a['positive_allocation_bytes']/2**20:.1f} "
             f"to {op_b['positive_allocation_bytes']/2**20:.1f} MiB. These allocations are traffic, not retained memory. "
             'Dense matrix operations and selected-expert indexing remain substantial costs; consolidating '
             'expert gradient accumulation is the next focused target.')
    deployment=('Enable with model.set_execution_backend("fused"). Linux CPU FP32/FP64 builds once with '
                'the local C++ compiler; binaries are CPU-specific and are not shipped. Other cases use '
                'the portable optimized path. SHOEBOX_FUSED=0 disables the kernel. The profiling command '
                'requires a working native kernel. Higher-order derivatives use the portable backend; '
                'the native kernel supports first-order reverse-mode training.')
    command=('python -m shoebox.profile_wide --mode train --backend fused \\\n'
             '  --data artifacts/cortical_minkron_fusion/profile_train.bin \\\n'
             '  --out artifacts/fusion_reproduction --targets 1000 --batch 4 --context 32\n')
    def md_table(head,body):
        return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(row)+' |\n' for row in body)
    text=f'# {title}\n\nNovember 8, 2026\n\n'+lead+'\n\n'+md_table(headers,rows)+'\n'+protocol+'\n\n'+fit+'\n\n'
    text+='## Implementation\n\n'+change+'\n\n'+memory+'\n\n## Validation\n\n'+checks+'\n\n'+numeric+'\n\n'
    text+='## Remaining runtime costs\n\n'+traffic+'\n\n## Use and reproduce\n\n'+deployment+'\n\n```bash\n'+command+'```\n\n'
    text+='The package contains the exact 1,004 input bytes, completed checkpoint, optimizer/state/RNG, port diagnostics, operator profiles and checksums.\n\n'
    text+='Full commands and equations: docs/cortical_minkron_fusion.md. The source fingerprint includes the C++ kernel.\n\n'
    text+=f"Runtime: PyTorch {training['torch']}, Python {training['python']}, AMD EPYC 9V74, one CPU thread.\n\n"
    text+=f"Source SHA-256: `{s['source_sha256']}`.\n\nData SHA-256: `{s['data_sha256']}`.\n\n"
    text+='Dataset: [FineWeb, sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb).\n'
    stem.with_suffix('.md').write_text(text)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(9,2.7),constrained_layout=True)
    for ax,values,label in [(axes[0],[a['bytes_per_second'],b['bytes_per_second']],'Throughput (bytes/s)'),
                            (axes[1],[a['peak_rss_mib']/1024,b['peak_rss_mib']/1024],'Peak RSS (GiB)')]:
        bars=ax.bar(['Previous optimized','Fused MinKron'],values,color=['#64748b','#0d9488'],width=.6)
        ax.bar_label(bars,fmt='%.2f',padding=3);ax.set_ylabel(label);ax.set_ylim(0,max(values)*1.2);ax.grid(axis='y',alpha=.15)
    chart=args.out/'SHoEbox_MinKron_Fusion_Profile.png';fig.savefig(chart,dpi=180);plt.close(fig)
    for name,family,weight in [('DV','DejaVu Sans','normal'),('DV-Bold','DejaVu Sans','bold'),('DV-Mono','DejaVu Sans Mono','normal')]:
        pdfmetrics.registerFont(TTFont(name,findfont(FontProperties(family=family,weight=weight))))
    styles=getSampleStyleSheet();styles['Title'].fontName=styles['Heading2'].fontName='DV-Bold'
    styles.add(ParagraphStyle(name='BodySmall',fontName='DV',fontSize=9,leading=12.5,spaceAfter=8))
    styles.add(ParagraphStyle(name='MonoSmall',fontName='DV-Mono',fontSize=7,leading=10,spaceAfter=4))
    def para(value,style='BodySmall'):return Paragraph(escape(value),styles[style])
    table=Table([[para(cell) for cell in row] for row in [headers,*rows]],colWidths=[248,139,139])
    table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dbeafe')),
                               ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f5f9'),colors.white]),
                               ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6)]))
    flow=[para(title,'Title'),para('Native CPU forward/backward | November 8, 2026'),para(lead),table,
          Spacer(1,12),Image(str(chart),width=526,height=157.8),Spacer(1,8),para(protocol),para(fit),
          PageBreak(),para('Implementation and validation','Heading2'),para(change),para(memory),para(checks),para(numeric),
          para('Remaining work','Heading2'),para(traffic),para('Use and reproduce','Heading2'),para(deployment)]
    flow.extend(para(line,'MonoSmall') for line in command.splitlines())
    flow += [para('The run ZIP includes the exact corpus, checkpoint, optimizer/state/RNG and raw diagnostics. '
                  'Full equations and commands are in docs/cortical_minkron_fusion.md.'),
             para('Source SHA-256','BodySmall'),para(s['source_sha256'],'MonoSmall'),
             para('Data SHA-256','BodySmall'),para(s['data_sha256'],'MonoSmall')]
    def footer(canvas,doc):
        canvas.setFont('DV',8);canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(43,22,'SHoEbox | measured MinKron fusion')
        canvas.drawRightString(569,22,str(doc.page))
    SimpleDocTemplate(str(stem.with_suffix('.pdf')),pagesize=(612,792),leftMargin=43,rightMargin=43,
                      topMargin=37,bottomMargin=37,title=title,author='Joshua Rowe / Codex').build(flow,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps({'markdown':str(stem.with_suffix('.md')),'pdf':str(stem.with_suffix('.pdf')),'chart':str(chart)}))


if __name__=='__main__':main()
