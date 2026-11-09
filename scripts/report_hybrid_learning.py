"""Render the measured Phase A comparison and downloadable report artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, Preformatted
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import inch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy',type=Path,required=True)
    p.add_argument('--latent',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    runs=[json.loads((path/'results.json').read_text()) for path in (args.legacy,args.latent)]
    audits=[json.loads((path/'audit.json').read_text()) for path in (args.legacy,args.latent)]
    calibration=json.loads((args.calibration/'calibration.json').read_text())
    for run,audit in zip(runs,audits):
        if not (run['complete'] and audit['exact_score_reproduction'] and audit['source_matches']):
            raise ValueError('Both arms need complete, audited endpoints.')
    legacy,latent=runs; la,ha=audits
    a,b=[run['evaluation'][-1] for run in runs]
    for key in ('train_sha256','validation_sha256','data_manifest_sha256','batch','context','updates','lr','threads','source_sha256'):
        if legacy['protocol'][key]!=latent['protocol'][key]:
            raise ValueError('Unmatched '+key)
    for key in ('nll','accuracy','targets'):
        if legacy['evaluation'][0][key]!=latent['evaluation'][0][key]:
            raise ValueError('Initial score mismatch')
    delta=b['nll']-a['nll']
    relative=100*(a['nll']-b['nll'])/a['nll']
    cfg=latent['protocol']['objective']
    table_rows=[['Next-byte NLL',f"{a['nll']:.6f}",f"{b['nll']:.6f}"],
                ['Next-byte accuracy',f"{100*a['accuracy']:.3f}%",f"{100*b['accuracy']:.3f}%"],
                ['Decoder swap penalty',f"{legacy['upper_swap_delta_nll']:+.6f}",f"{latent['upper_swap_delta_nll']:+.6f}"],
                ['Decoder + feedback penalty',f"{legacy['upper_all_swap_delta_nll']:+.6f}",f"{latent['upper_all_swap_delta_nll']:+.6f}"],
                ['Training bytes/s',f"{legacy['training_bytes_per_second']:.1f}",f"{latent['training_bytes_per_second']:.1f}"],
                ['Peak RSS (MiB)',f"{legacy['peak_rss_mib']:.0f}",f"{latent['peak_rss_mib']:.0f}"],
                ['Stored parameters',f"{legacy['parameters']['total']:,}",f"{latent['parameters']['total']:,}"],
                ['Inference parameters',f"{legacy['parameters']['inference_parameters']:,}",f"{latent['parameters']['inference_parameters']:,}"]]
    ground=[]
    for name,item in b['regional'].items():
        ground.append([name,f"{item['huber']:.5f}",f"{item['persistence']:.5f}",f"{item['shuffled_future']:.5f}",
                       f"{100*(item['shuffled_future']-item['huber'])/item['shuffled_future']:.2f}%",
                       f"{item['pairs']:,} / {item['possible']:,}"])
    gains={k:(v['shuffled_future']-v['huber'])/v['shuffled_future'] for k,v in b['regional'].items()}
    best=max(gains,key=gains.get); worst=min(gains,key=gains.get)
    outcome=(f"Phase A reaches {b['nll']:.6f} nats/byte versus {a['nll']:.6f} for the matched legacy objective "
             f"(Phase A minus legacy {delta:+.6f}; {abs(relative):.2f}% {'lower' if relative>0 else 'higher'} NLL). "
             f"Accuracy is {100*b['accuracy']:.3f}% versus {100*a['accuracy']:.3f}%.")
    # This report deliberately keeps promotion separate from a single short arm comparison.
    decision=("Keep both modes available and retain legacy as the default. The requested Phase A implementation "
              "and 262,144-target comparison are complete. The next decision is whether to strengthen future-input "
              "grounding before introducing owner-local gradient routing; no Phase B-D dynamics were added.")
    interpretation=(f"Against shuffled-future labels, the largest predictive advantage is {best} "
                    f"({100*gains[best]:.2f}% lower Huber); the smallest is {worst} ({100*gains[worst]:.2f}%). "
                    "Persistence repeats the preceding ordered source bundle. Shuffling rotates complete future "
                    "bundles across valid examples while retaining their internal source/time order. All latent "
                    "comparisons use the same checkpoint's fixed training statistics.")
    fixture=ha['grounding']['fixed_prefix']
    response=', '.join(f"{n} {r['changed_future_target_mse']:.5f}" for n,r in fixture.items())
    health=', '.join(f"{n} {v:.3f}" for n,v in b['message_std'].items())
    cover={name:sum(r['regional'][name]['pairs'] for r in latent['history'])/
                    sum(r['regional'][name]['possible'] for r in latent['history']) for name in ('L2','S2','L3')}
    training_seconds=[r['training_seconds'] for r in runs]
    timing=(f"Measured update time: legacy {training_seconds[0]/60:.2f} minutes; Phase A {training_seconds[1]/60:.2f} minutes. "
            "The arms ran concurrently, each with one CPU thread and the existing fused CPU kernels. Throughput "
            "is aggregate external byte targets per second, not single-stream generation speed. Peak RSS is process-wide.")
    # Tests are also preserved in the deliverable bundle.
    checks=("All 62 existing checks and seven focused Phase A checks pass. An independent real CLI run exactly "
            "matches uninterrupted weights, Adam state, recurrent state, normalizers and RNG after pause/resume. "
            "Changed objective, source or document manifest is rejected. Both final checkpoints exactly reproduce "
            "their full held-out results after reload; validation leaves normalizers unchanged.")
    init=("Both arms start from b4507b4's 1,048,576-target checkpoint and retain its shared weights and all 464 "
          "saved Adam states at age 768. Two disjoint B8/T256 calibration windows observe 4,096 training bytes "
          "without any weight update. New predictors start with fresh Adam state. At each endpoint, active shared "
          "states are age 896; the candidate's new heads are age 128 and retained legacy heads stay at 768. "
          "Each arm ends at 896 lifetime optimizer updates and 1,310,720 optimizer target exposures.")
    weights=(f"Fixed Huber coefficients: L2 {cfg['l2_weight']:.8f}, S2 {cfg['s2_weight']:.8f}, L3 {cfg['l3_weight']:.8f}. "
             "Calibration targets 10% of the owning publisher's global-byte gradient norm, bounded to [0.005, 0.2]. "
             "Variance weight 0.01, covariance weight 0.001, standard deviation target 0.1; source normalizer "
             "EMA rate 0.05 and variance floor 0.01. These coefficients stay fixed throughout training.")
    coverage='; '.join(f"{n} {100*v:.2f}%" for n,v in cover.items())
    data=("Training uses 93 previously unused FineWeb documents, 262,144 targets per arm, B8/T256, joint AdamW "
          "0.001, betas (0.9, 0.95), matrix decay 0.01 and gradient clip 1. Validation is unchanged: 65,536 "
          "targets across 23 documents, with 128-byte evaluation windows. All prior training, validation and "
          "reserved documents are excluded by ID and full source-text hash. Calibration uses a separate nine-document "
          "8,192-target corpus; only its first 4,096 positions are observed.")
    mechanics=("L2 predicts four future A1 messages; S2 predicts four ordered A1/B1 pairs; L3 predicts four "
               "ordered L2/S2 pairs. Future labels are detached; the current representation and global byte path "
               "retain upstream gradients. Complete spans must stay in one document, batch row and forward window. "
               "The existing recurrent stream trajectory is unchanged; document masking applies to objective pairs. "
               "Live published messages receive variance and redundancy penalties. Sampling constructs no target "
               "records and runs no auxiliary heads. Training adds 83,200 parameters; inference remains 2,652,512.")
    hypotheses=("Changing future observations after an identical 32-byte prefix leaves predictor inputs exactly "
                "unchanged and changes the subsequent teacher bundles. Normalized target-response MSE: "+response+". "
                "This verifies sensitivity to future evidence; the shuffled-future comparison measures how much "
                "of that evidence the trained predictors anticipate.")
    gradient_stats={}
    for name in sorted(set().union(*(r['regional_gradient_norms'] for r in latent['history']))):
        values=[r['regional_gradient_norms'][name] for r in latent['history'] if name in r['regional_gradient_norms']]
        gradient_stats[name]=dict(mean=sum(values)/len(values),maximum=max(values),samples=len(values))
    result=dict(format='shoebox-hybrid-phase-a-comparison-1',delta_nll=delta,relative_nll_improvement_percent=relative,
                legacy=a,latent=b,protocol=latent['protocol'],calibration=calibration,
                training_pair_coverage=cover,gradient_norms=gradient_stats,
                audit_legacy=la,audit_latent=ha,decision=decision,
                checks=dict(existing=62,new=7),samples=dict(legacy=legacy['samples'],latent=latent['samples']))
    args.out.mkdir(parents=True,exist_ok=True)
    name='SHoEbox_Hybrid_Learning_Report'
    def md_table(headers,rows):
        return '| '+' | '.join(headers)+' |\n|'+ '|'.join('---' for _ in headers)+'|\n'+'\n'.join('| '+' | '.join(r)+' |' for r in rows)
    md=f'''# SHoEbox: hybrid learning Phase A

November 8, 2026. Six-region hierarchy; matched continuation from b4507b4.

{outcome}

{decision}

## What changed

{mechanics}

## Matched run

{data}

{init}

{weights}

## Full held-out result

Byte losses and swap penalties are nats/byte; lower ordinary NLL is better.

{md_table(['Measurement','Legacy','Phase A'],table_rows)}

{timing}

## Predictive grounding and message health

{md_table(['Region','Future Huber','Persistence','Shuffled future','Gain vs shuffled','Valid / eligible'],ground)}

{interpretation}

{hypotheses}

Mean within-window message standard deviations: {health}. Training pair coverage:
{coverage}. Coverage excludes incomplete future bundles and cross-document spans;
no target is borrowed from a later optimizer step. Per-region gradients, full
message-health histories and per-source losses accompany the checkpoints.

## Verification

{checks}

## Fixed-prompt samples

Same prompts, seed 731, temperature 0.8, 64 generated bytes.

### Legacy

```text
{chr(10).join(legacy['samples'])}
```

### Phase A

```text
{chr(10).join(latent['samples'])}
```

## Reproduction and handoff

Commands and API contracts: `docs/hybrid_learning.md`. Implementation retains
legacy objectives as the default. Calibration, exact CLI verification, endpoint
audits and report generation have dedicated scripts in `scripts/`.

The checkpoint ZIP contains the original seed, calibrated seed, both final
checkpoints, exact prepared corpora and manifests, cached FineWeb responses,
fixed coefficients, histories and verification records. Model source is published
on `JRowe47/SHoEbox`, branch `feat/hierarchy-pilot`; an attached patch contains
this implementation and documentation relative to b4507b4.

| Provenance | SHA-256 |
|---|---|
| Shared starting checkpoint | `{calibration['shared_origin_sha256']}` |
| Fresh training bytes | `{latent['protocol']['train_sha256']}` |
| Fixed validation bytes | `{latent['protocol']['validation_sha256']}` |
| Legacy endpoint | `{la['checkpoint_sha256']}` |
| Phase A endpoint | `{ha['checkpoint_sha256']}` |
| Runtime source | `{latent['protocol']['source_sha256']}` |
'''
    (args.out/(name+'.md')).write_text(md)
    (args.out/'SHoEbox_Hybrid_Learning_Results.json').write_text(json.dumps(result,indent=2)+'\n')

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,ax=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    for run,audit,color,label in zip(runs,audits,['#64748b','#087f8c'],['Legacy','Phase A']):
        series=[r for r in run['evaluation'] if not r['full']]+[dict(audit['monitoring'],updates=128)]
        ax[0].plot([r['updates'] for r in series],[r['nll'] for r in series],marker='o',color=color,label=label)
    ax[0].set(xlabel='Additional optimizer updates',ylabel='Nats / byte',title='Fixed 8,192-target monitor');ax[0].legend(frameon=False);ax[0].grid(alpha=.15)
    regions=list(b['regional'])
    x=list(range(len(regions)))
    for shift,key,label,color in [(-.25,'huber','Predictor','#087f8c'),(0,'persistence','Persistence','#d4a72c'),(.25,'shuffled_future','Shuffled future','#94a3b8')]:
        ax[1].bar([i+shift for i in x],[b['regional'][n][key] for n in regions],width=.24,label=label,color=color)
    ax[1].set(xticks=x,xticklabels=regions,ylabel='Normalized Huber',title='Full held-out latent targets');ax[1].legend(frameon=False,fontsize=8)
    fig.savefig(args.out/'SHoEbox_Hybrid_Learning_Curves.png',dpi=180);plt.close(fig)

    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=24,leading=29,textColor=colors.HexColor('#132d3a'),spaceAfter=13))
    styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9,textColor=colors.HexColor('#243744')))
    styles.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=8.4,leading=11,spaceAfter=7))
    styles.add(ParagraphStyle(name='CodeCustom',fontName='Courier',fontSize=8,leading=11,spaceAfter=8))
    story=[]
    def para(text,style='BodyCustom'):
        story.append(Paragraph(escape(text),styles[style]))
    def heading(text):
        story.append(Paragraph(text,styles['Heading2']))
    def table(headers,rows,widths):
        data=[[Paragraph(escape(str(c)),styles['SmallCustom']) for c in r] for r in [headers]+rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7f1f2')),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#087f8c')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dbe1e5')),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.append(t);story.append(Spacer(1,10))
    para('SHoEbox: hybrid learning','TitleCustom');para('Phase A | November 8, 2026','SmallCustom')
    para(outcome);para(decision)
    heading('Matched held-out result');table(['Measurement','Legacy','Phase A'],table_rows,[252,126,126])
    para(timing,'SmallCustom');heading('What changed');para(mechanics)
    story.append(PageBreak())
    heading('Grounding and representation health')
    table(['Region','Future','Persistence','Shuffled','Gain'],[r[:5] for r in ground],[56,90,110,110,138])
    para(interpretation);para(hypotheses);para('Mean message standard deviations: '+health+'.')
    para('Training pair coverage: '+coverage+'. Only complete, document-isolated future bundles are scored.')
    story.append(Image(str(args.out/'SHoEbox_Hybrid_Learning_Curves.png'),width=504,height=171.4))
    heading('Verification');para(checks)
    story.append(PageBreak())
    heading('Training protocol');para(data);para(init);para(weights)
    heading('Fixed-prompt samples');para('Seed 731, temperature 0.8; 64 generated bytes per prompt.','SmallCustom')
    for label,run in [('Legacy',legacy),('Phase A',latent)]:
        para(label,'SmallCustom')
        for sample in run['samples']:
            # Wrapping and escaping preserve samples without overflowing the page.
            para(sample.encode('ascii','backslashreplace').decode(), 'CodeCustom')
    heading('Reproduction');para('See docs/hybrid_learning.md for complete commands and contracts. The accompanying checkpoint archive includes seed and endpoint states, exact data, calibration, histories and verification. The downloadable Markdown report contains complete hashes; the code patch is relative to b4507b4.','SmallCustom')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(54,31,'SHoEbox | Hybrid learning Phase A');canvas.drawRightString(558,31,str(doc.page))
    SimpleDocTemplate(str(args.out/(name+'.pdf')),pagesize=(612,792),leftMargin=54,rightMargin=54,topMargin=42,bottomMargin=48).build(story,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps(dict(delta_nll=delta,outputs=[name+'.md',name+'.pdf','SHoEbox_Hybrid_Learning_Results.json'])))


if __name__=='__main__':
    main()
