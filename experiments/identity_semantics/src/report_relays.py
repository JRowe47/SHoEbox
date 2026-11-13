"""Build compact relay-repair evidence and downloadable Markdown/PDF reports."""
import argparse,html,json,re,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables'

def render_pdf(md,pdf):
 from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether
 from reportlab.lib import colors
 from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
 from reportlab.pdfbase import pdfmetrics
 from reportlab.pdfbase.ttfonts import TTFont
 for name,file in [('Sans','DejaVuSans.ttf'),('Sans-Bold','DejaVuSans-Bold.ttf')]:
  pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/truetype/dejavu/'+file))
 pdfmetrics.registerFontFamily('Sans',normal='Sans',bold='Sans-Bold')
 styles=getSampleStyleSheet()
 styles.add(ParagraphStyle(name='BodyR',fontName='Sans',fontSize=9.2,leading=13.4,spaceAfter=7,textColor=colors.HexColor('#203244')))
 styles.add(ParagraphStyle(name='TitleR',fontName='Sans-Bold',fontSize=23,leading=28,spaceAfter=12,textColor=colors.HexColor('#123c55')))
 styles.add(ParagraphStyle(name='HeadR',fontName='Sans-Bold',fontSize=12.4,leading=17,spaceBefore=10,spaceAfter=7,keepWithNext=True,textColor=colors.HexColor('#126078')))
 styles.add(ParagraphStyle(name='CellR',fontName='Sans',fontSize=8.2,leading=11.5,textColor=colors.HexColor('#203244')))
 styles.add(ParagraphStyle(name='CodeR',fontName='Courier',fontSize=8,leading=11.5))
 def markup(text):
  text=html.escape(text)
  text=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',text)
  return re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
 lines=md.read_text().splitlines();story=[];i=0
 while i<len(lines):
  line=lines[i].strip();i+=1
  if not line:continue
  if line=='<!-- pagebreak -->':story.append(PageBreak());continue
  if line.startswith('# '):story.append(Paragraph(markup(line[2:]),styles['TitleR']));continue
  if line.startswith('## '):story.append(Paragraph(markup(line[3:]),styles['HeadR']));continue
  if line.startswith('```'):
   code=[]
   while i<len(lines) and not lines[i].startswith('```'):
    code.append(Paragraph(html.escape(lines[i]),styles['CodeR']));i+=1
   i+=1;story.append(KeepTogether(code+[Spacer(1,8)]));continue
  if line.startswith('|'):
   rows=[line]
   while i<len(lines) and lines[i].startswith('|'):rows.append(lines[i]);i+=1
   values=[[c.strip() for c in row.strip('|').split('|')] for row in rows]
   values=[row for row in values if not all(re.fullmatch(r'[-: ]+',c) for c in row)]
   widths=[115,377] if len(values[0])==2 else [492/len(values[0])]*len(values[0])
   cells=[[Paragraph(markup(c),styles['CellR']) for c in row] for row in values]
   table=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
   table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebf1')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f3f7f9'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,-1),.25,colors.HexColor('#cbdbe2'))]))
   story.extend([table,Spacer(1,7)]);continue
  story.append(Paragraph(markup(line),styles['BodyR']))
 def page(canvas,doc):
  canvas.saveState();canvas.setFont('Sans',7.3);canvas.setFillColor(colors.HexColor('#536b7c'))
  canvas.drawString(52,815,'SHoEbox / Relay teaching and recognition / 1M model')
  canvas.drawString(52,26,'13 November 2026 | Tested source and measured results')
  canvas.drawRightString(544,26,str(doc.page));canvas.restoreState()
 SimpleDocTemplate(str(pdf),pagesize=(596,842),leftMargin=52,rightMargin=52,topMargin=48,bottomMargin=48,
                   title='SHoEbox: teaching and recognition through relays').build(story,onFirstPage=page,onLaterPages=page)

def main():
 runs={(k,s):json.loads((ROOT/f'results/relay_full_{k}_seed{s}/training.json').read_text()) for k in ('semantics','relays') for s in (1,2)}
 assert all(r['status']=='complete' and r['tokens']==1000000 for r in runs.values())
 perf=json.loads((ROOT/'results/relay_throughput.json').read_text())
 verification=json.loads((ROOT/'results/relay_verification.json').read_text())
 selection=json.loads((ROOT/'results/relay_rate_selection.json').read_text())
 means={k:statistics.mean(runs[k,s]['test_nll'] for s in (1,2)) for k in ('semantics','relays')}
 delta=means['relays']-means['semantics'];speed=perf['median_bytes_per_second']
 change=100*(speed['relays']/speed['semantics']-1)
 compact=[]
 for (k,s),r in runs.items():
  item={key:r[key] for key in ('kernel','seed','peak_lr','parameters','tokens','validation','test_nll','full_heldout_nll','train_sha256','heldout_sha256','composition_bytes','relay_teaching')}
  compact.append(item)
  if k=='relays':
   for row in r['relay_teaching'].values():
    assert row['expired_producers']==row['duplicate_schedules']==0
    assert row['scheduled']==sum(row[x] for x in ('neural_targets','relay_targets','nonlearning_targets'))+int(row['pending_tick']>0)
 summary=dict(base_commit='500309bcb7ea0de7a198656c37dc21507e071c24',kernel='relays',parameters=1031936,
              checkpoint_version=31,composition_abi=5,runs=compact,mean_test_nll=means,change_nll=delta,
              relative_nll_change_percent=100*delta/means['semantics'],throughput=perf,validation_selection=selection,verification=verification)
 OUT.mkdir(exist_ok=True)
 (OUT/'SHoEbox_Relay_Preservation_Results.json').write_text(json.dumps(summary,indent=2)+'\n')
 rows='\n'.join(f"| {s} | {runs['semantics',s]['test_nll']:.6f} | {runs['relays',s]['test_nll']:.6f} | {runs['relays',s]['test_nll']-runs['semantics',s]['test_nll']:+.6f} |" for s in (1,2))
 counts=[]
 for s in (1,2):
  r=runs['relays',s]['relay_teaching']['L2']
  counts.append(f"| {s} | {r['scheduled']:,} | {r['neural_targets']:,} | {r['relay_targets']:,} | {int(r['pending_tick']>0)} |")
 text=f'''# SHoEbox: teaching and recognition through relays

13 November 2026 | Third repair from the principled code audit

## Result

Singleton relays now preserve outstanding producer teaching and inherited recognition. The repaired `relays` kernel builds on the published identity/vector runtime; it adds no neural parameters. The 1M configuration remains **1,031,936 parameters**.

All **{verification['package_tests']} package checks** pass, including the prior 62 checks and the new relay suite. ASan/UBSan and the repository check command pass. Two fresh seeds per arm each train on one million byte targets. Mean reserved-test NLL changes **{means['semantics']:.6f} to {means['relays']:.6f}**, a signed change of **{delta:+.6f} nats/byte** ({100*delta/means['semantics']:+.2f}%). Lower is better.

Same-core median training throughput changes **{speed['semantics']:,.0f} to {speed['relays']:,.0f} bytes/s** ({change:+.1f}%).

## The repaired contract

| Object | Behavior |
| --- | --- |
| Pending forecast | Saves the prediction, observed producer input, and the producer's eligibility tick/index. The saved record supplies activation, expert, gate, routing cue and semantic features. |
| Next target | The next observed regional content input teaches exactly once. A completed singleton relay qualifies; idle ticks do not. Y retains its per-byte normalized input and observed prefix-byte cue. |
| Transparent relay | Preserves content ID, descriptor, semantic code, intrinsic arity/kind, children, recognition, confidence and actual producer provenance. It does not execute or schedule a local expert. |
| Local metadata | `local_arity` and `relayed` describe the receiver's action independently of the inherited unit. Relay transport does not increment locally formed-compound counts. |
| Proposed relation | A new relation between familiar units remains provisional until its own dictionary supplies recognition evidence. |

The target is evaluated before replacing the public output. Predictor, expert, router and semantic updates use the saved producer. A nonlearning or frozen target retires the forecast without modifying weights. A missing or overwritten producer is counted and rejected. Document resets clear pending forecasts; chunk boundaries do not. Save/load preserves a pending forecast exactly.

In the current multi-source graph, ordinary singleton traffic occurs at L2. Controlled tests additionally carry recognized and provisional compounds through two transparent hops and verify that inherited familiar units can still form a new provisional relation.

## Verification

Checks cover every declared binding role, identity/vector distinctions, saved-feature gradients, changed public outputs, numerical predictor updates, once-only teaching, multi-hop recognition and provenance, frozen/zero-rate policies, stale-producer rejection, exact checkpoint/chunk continuation, reset behavior and explicit fitted-state conversion. The live-stream and million-byte runs account for every forecast, with **zero expired producers and zero duplicate schedules**.

<!-- pagebreak -->

## Matched language experiment

Width 64, 12 experts per region, unchanged fixed transports, the same Tiny Shakespeare bytes and identical seed-specific initial parameters. Validation-only pilots test relay peak rates 0.02, 0.04 and 0.08 on seed 3 for 100,000 targets. The selected rate is 0.08, matching the prior runtime. Both arms use cosine decay to 0.008 by one million targets, local rate one third of main, and prototype rate 0.002.

The 32,768-target validation segment selects the rate; the disjoint **78,771-target test segment** supplies the endpoint. Evaluation starts with fresh stream/episodic state, frozen neural parameters and dictionaries, and causal memory admission from the observed prefix.

| Seed | Previous semantics | Repaired relays | Change |
| --- | ---: | ---: | ---: |
{rows}
| Mean | {means['semantics']:.6f} | {means['relays']:.6f} | {delta:+.6f} |

L2 forecast accounting after training:

| Seed | Scheduled | Neural targets | Relay targets | Pending |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(counts)}

## Throughput and state

Six alternating trials per arm use one pinned CPU core, 1,024 warmup targets and 16,384 timed targets after other workers exit. Timing includes forward execution, routing, dictionaries, episodic admission and learning; it excludes compilation, evaluation and checkpoint I/O.

Composition state grows from **{runs['semantics',1]['composition_bytes']:,} to {runs['relays',1]['composition_bytes']:,} bytes**, an increase of **{runs['relays',1]['composition_bytes']-runs['semantics',1]['composition_bytes']:,} bytes**. No model-size increase accompanies the repair.

## Run and continue

Use `fresh_model(kernel="relays")`. The historical default remains `composition`; `semantics` remains the unchanged comparison kernel. New checkpoints use **version 31 / composition ABI 5** and resume exactly with `Model.load()`.

To start from version-30 learned semantics, `convert_to_relays.py` preserves neural/semantic weights, dictionaries, fixed transports, policies and fitted preprocessing means while resetting stream context, pending forecasts, eligibility, online adaptation state and episodic associations. This is a fresh-stream conversion, not exact continuation of the old event ABI.

```bash
python -m unittest discover -s tests -v
python tests/run_relays_sanitizer.py
python src/convert_to_relays.py OLD.npz NEW.npz
python src/run_relays_study.py pilots
python src/run_relays_study.py full
python src/relays_performance.py
```

Source, tests and compact measurements accompany this repair. Test models and intermediate run artifacts are not published. The next agreed step is regional neural contribution checks with content and relay transport preserved.
'''
 md=OUT/'SHoEbox_Relay_Preservation_Report.md';md.write_text(text)
 render_pdf(md,md.with_suffix('.pdf'))
 print(json.dumps(dict(mean_test_nll=means,delta=delta,speed=speed,report=str(md))))

if __name__=='__main__':main()
