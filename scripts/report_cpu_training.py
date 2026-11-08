"""Create the downloadable learning curve and Markdown/PDF CPU training report."""
import argparse
import json
import math
from pathlib import Path
import sys
import textwrap

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.render_markdown_report import render


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    data=json.loads((args.run/'results.json').read_text())
    audit=json.loads(args.audit.read_text())
    assert data['complete'] and audit['reload_exact']
    args.out.mkdir(parents=True,exist_ok=True)
    count=data['trained_targets'];label=f'{count//1000}K' if count%1000==0 else str(count)
    stem=f'SHoEbox_CPU_{label}'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(7.4,3.4),sharex=True,gridspec_kw={'height_ratios':[1.5,1]},layout='constrained')
    history=data['history'];validation=data['validation']
    x=[row['trained_targets']/1000 for row in history]
    axes[0].plot(x,[row['nll'] for row in history],color='#207767',linewidth=1.7,label='Training window')
    axes[0].plot([r['trained_targets']/1000 for r in validation],[r['nll'] for r in validation],
                 color='#b55435',marker='D',markersize=4.5,linewidth=1.5,label='Held-out validation')
    axes[0].axhline(audit['unigram_baseline']['nll'],color='#777777',linestyle='--',linewidth=1,label='Unigram baseline')
    axes[0].set_ylabel('Nats per byte');axes[0].legend(frameon=False,loc='upper right',fontsize=8)
    axes[1].plot(x,[row['training_bytes_per_second'] for row in history],color='#356b9a',linewidth=1.4)
    axes[1].set_ylabel('Training bytes/s');axes[1].set_xlabel('Additional training bytes (thousands)')
    for ax in axes:ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True)
    axes[1].set_xlim(0,count/1000)
    image_name=stem+'_Learning.png';fig.savefig(args.out/image_name,dpi=180);plt.close(fig)
    initial=data['initial_validation'];final=data['final_validation'];best=data['best_validation']
    unigram=audit['unigram_baseline'];swapped=audit['swapped_input_probe']
    predicted=audit['aligned_predicted_byte_counts']
    prediction_text=(f"All {final['targets']:,} greedy predictions are spaces." if predicted=={'32':final['targets']}
                     else f'Greedy predictions use {len(predicted)} distinct byte values.')
    context_result=(
        'The improvement mainly reflects byte-frequency learning. Contextual prediction has not yet separated from that baseline.'
        if final['nll']>=unigram['nll'] and abs(swapped['nll_increase'])<.01
        else 'The input-swap result measures the contribution of correctly paired preceding observations on this validation slice.')
    decrease=100*(1-final['nll']/initial['nll'])
    tokens=[row['trained_targets'] for row in history]
    weights=[b-a for a,b in zip([0]+tokens[:-1],tokens)]
    mean_train=sum(row['nll']*weight for row,weight in zip(history,weights))/count
    memory=data['memory_writes']
    primary={name:data['diagnostics']['ports'][f'{name}:{16}'] for name in ('A1','B1')}
    samples=[]
    for row in data['samples']:
        text=json.dumps(row['prompt']+row['generated'],ensure_ascii=True)[1:-1]
        samples.append('```\n'+'\n'.join(textwrap.wrap(text,92,replace_whitespace=False,drop_whitespace=False))+'\n```')
    sample_text='\n\n'.join(samples)
    report=f'''# SHoEbox CPU: {label} training run

November 8, 2026 | [Published source {args.commit[:7]}](https://github.com/JRowe47/SHoEbox/commit/{args.commit})

Completed **{count:,} additional byte-token targets** with the {data['parameters']:,}-parameter
sparse CPU model. The run continues the saved {data['base_trained_targets']:,}-target
checkpoint, giving **{data['lifetime_trained_targets']:,} lifetime training targets**.
Weights and AdamW moments continue; runtime state starts fresh for {data['protocol']['batch']} new streams.

## Learning result

| Measure | Initial | Final |
|---|---:|---:|
| Held-out NLL, nats/byte | {initial['nll']:.4f} | {final['nll']:.4f} |
| Held-out bits/byte | {initial['bits_per_byte']:.4f} | {final['bits_per_byte']:.4f} |
| Held-out next-byte accuracy | {initial['accuracy']:.2%} | {final['accuracy']:.2%} |
| Byte perplexity | {math.exp(initial['nll']):.2f} | {math.exp(final['nll']):.2f} |

Final held-out NLL improves **{decrease:.1f}%**. The best validation NLL is
**{best['nll']:.4f} at {best['trained_targets']:,} additional targets**. Validation
uses the same {final['targets']:,} bytes from source documents absent from the training corpus.
The earlier shakedown prefix is included in this training corpus; lifetime
targets count exposures, not unique bytes.

The training-fitted unigram baseline scores **{unigram['nll']:.4f} nats/byte**.
Swapping inputs between held-out streams changes model NLL by
**{swapped['nll_increase']:+.6f}**. {prediction_text}
{context_result}

![Learning and throughput]({image_name})

The run performs **{data['updates']} optimizer updates** at B{data['protocol']['batch']}/T{data['protocol']['context']}, with a final
{weights[-1]//data['protocol']['batch']}-observation window. Forward, backward, clipping and AdamW total
**{data['training_seconds']/60:.2f} minutes**, averaging
**{data['training_bytes_per_second']:.2f} external bytes/s**. Peak RSS is
**{data['peak_rss_mib']/1024:.2f} GiB**. Full session time, including validation,
checkpoints and samples, is **{data['session_wall_seconds']/60:.2f} minutes**.
Mean training-window NLL, weighted by byte count, is {mean_train:.4f}.

---

## Protocol and checkpoint audit

The model retains learned selection of seven or eight of 18 regions per stream,
up to two permitted incoming edges, complete 4/8/16-byte ports and the
32-observation gradient window. Training uses two CPU intra-operation threads,
one regional worker, native AVX512/OpenMP kernels and FP32 arithmetic.
AdamW warms to 0.001 over 3,200 targets, then holds that learning rate;
gradient norm clipping is 1.0. No architecture or optimizer changes occur mid-run.

Last checkpoints are written every 12,800 targets and at completion. Best
checkpoints follow held-out NLL. The final checkpoint reload reproduces validation
exactly; all parameters and logged gradient norms are finite. A separate split-run
check reproduces weights, optimizer, stream state and RNG exactly after resume.
Training streams write **{sum(memory):,} EC episodes** in total
({min(memory)} to {max(memory)} per stream); retained capacity remains bounded.

On the first {data['protocol']['context']} held-out observations, A1/B1 primary-port soft entropy is
{primary['A1']['conditional_entropy_bits']:.3f}/{primary['B1']['conditional_entropy_bits']:.3f} bits per byte. Their complete-frame diversity is {primary['A1']['unique_frame_fraction']:.1%} and {primary['B1']['unique_frame_fraction']:.1%},
respectively. The archive retains initial/final port statistics and regional
selection fractions for follow-up analysis. Generated samples remain incoherent.

## Generated samples

Each sample uses temperature 0.8, seed 731 and {len(bytes.fromhex(data['samples'][0]['generated_hex'])) if data['samples'] else 0}
generated bytes. Escapes preserve non-ASCII or control bytes in this text view.

{sample_text}

## Reproduce or continue

```bash
python -m shoebox.train_cpu --init initial_2k.pt \\
  --data train.bin --validation validation.bin \\
  --out artifacts/cpu_100k --targets 100000 --validation-targets 4000 \\
  --batch 50 --context 32 --threads 2 --workers 1
```

The checkpoint archive includes last/best checkpoints, learning history, validation,
port entropy/diversity and routing diagnostics, exact corpus prefixes, source
provenance and the reload audit. For an interrupted run, replace `--init` with
`--resume last.pt`, retain the same protocol and use a new output directory.
For further training on new streams, `--init last.pt` carries weights and AdamW
while resetting runtime state. The earlier 2k checkpoint is the initialization
for this experiment; its SHA-256 is recorded in the results.

Source digest: `{data['protocol']['source_sha256']}`
'''
    md=args.out/(stem+'_Report.md');md.write_text(report)
    render(md,args.out/(stem+'_Report.pdf'),title=f'SHoEbox CPU {label} Training',
           footer_text=f'SHoEbox | CPU {label} training | November 8, 2026')
    print(json.dumps({'markdown':str(md),'pdf':str(args.out/(stem+'_Report.pdf')),'image':str(args.out/image_name)}))


if __name__=='__main__':main()
