"""Build the compact Action-7 Markdown/PDF report from recorded measurements."""
import hashlib,json,sys
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Preformatted
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from measure_current_baseline import source_manifest

def main():
 one=json.loads((ROOT/'results/action7_1m_measurements.json').read_text())
 five=json.loads((ROOT/'results/action7_5m_measurements.json').read_text())
 assert one['completed'] and five['completed']
 selected=next(r for r in one['summary'] if r['profile']=='selected' and r['workload']=='natural')
 manifest=source_manifest();manifest['baseline_revision']='085e0a60662aa8419bab7ea7f5db3e7d5c1f79c1'
 manifest['measurement_driver_sha256']=hashlib.sha256((ROOT/'src/measure_inference.py').read_bytes()).hexdigest()
 (ROOT/'results/action7_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 rows=[]
 for r in one['summary']:
  rows.append(['Reference' if r['profile']=='reference' else 'Selected',r['workload'].capitalize(),
       f"{r['frozen_bps']:,.0f}",f"{r['forward_bps']:,.0f}",f"{r['speedup']:.2f}x",
       f"{r['frozen_rss_mib']:.1f} / {r['forward_rss_mib']:.1f}"])
 q=five['summary'][0]
 scale=[['Selected','Natural',f"{q['frozen_bps']:,.0f}",f"{q['forward_bps']:,.0f}",f"{q['speedup']:.2f}x",f"{q['frozen_rss_mib']:.1f} / {q['forward_rss_mib']:.1f}"]]
 header=['Profile','Workload','Frozen B/s','Forward B/s','Speedup','RSS MiB old / new']
 outcome=(f"Action 7 is complete. The selected 1M profile processes natural text at {selected['forward_bps']:,.0f} B/s versus "
  f"{selected['frozen_bps']:,.0f} B/s with frozen training storage: {100*(selected['speedup']-1):.1f}% faster. "
  f"Process RSS drops from {selected['frozen_rss_mib']:.1f} to {selected['forward_rss_mib']:.1f} MiB ({selected['rss_reduction_percent']:.1f}% less). "
  "All matched output hashes and held-out losses are identical.")
 scope=("The new forward-only-v1 profile keeps all 18 regions, exact binding/source/occurrence semantics, learned content, "
  "MinKron recurrence, previous-byte feedback and paired episodic memory. Pattern assignments and lightweight-frame role sums "
  "move into immutable forward publications. Semantic derivative slabs, content/context eligibility, neural/temporal archives, "
  "reverse work and learning jobs are omitted; training-only collection and bottom temporal Jacobians are skipped.")
 protocol=(f"Host: {one['environment']['cpu_model']}; {one['environment']['compiler']}; "
  "production -O3 -march=native -std=c++17 -pthread, caller pinned to CPU 0. Three isolated subprocess trials per pair, "
  "with alternating runtime order. Each source receives 2,048 training targets at main/local LR .04/.01333, then a matching "
  "warm-adaptive frozen copy receives 1,024 warmup targets. Timed evaluation covers 8,192 targets in 1,024-byte chunks with "
  "logits export; generation covers 1,024 sampled bytes at temperature .8, including Python draw/run overhead and excluding "
  "full-logits export. Training, construction, compilation, inspection, hashing, evaluation copies and file IO are outside the timers. "
  "Tests, sanitizers and timing do not overlap.")
 data=("Natural text is the pinned 70ea7b0 README/STATUS fixture; its separate 1,024-target validation fixture is "
  "docs/cortical_fineweb.md. Repeated text and its distinct validation string use the Action-1 fixture. This is not a corpus-scale "
  "FineWeb qualification. The reference has 1,031,952 fitted parameters; the selected profile has 1,014,800 at 1M and "
  "4,981,776 at 5M, using rank-16 prediction and Y stride 8. Both retain learned frames. The selected source uses four-byte "
  "sparse epochs with one-epoch lag; no learning worker runs during inference.")
 memory=("RSS is measured after the source model has closed and the inference stream has warmed, then after timed execution. "
  "It includes Python, parameters, live state and allocator overhead. Native inventories report zero semantic derivative slab bytes, "
  "derivative-vector capacity, neural/context/temporal archives, live neural/context eligibility, reverse roots and pending bank work "
  "in forward-only runs. Fixed native compatibility buffers still occupy about 20.5 MiB; they are not a compact GPU layout.")
 quality=("For the natural-text 1M source, held-out NLL is 3.822702 for the reference and 3.792820 for the selected profile, "
  "unchanged between frozen and forward-only execution. The 5M selected source gives 3.776932 in both modes. Identical generated-byte "
  "hashes use matched RNG streams. These comparisons qualify inference values and cost, not a new learning-policy winner.")
 transition=("enter_inference() commits materialized slow-Y gradients, reverse epochs, directional replay and bank jobs, joins workers, "
  "then retires remaining eligibility and pending next-event teachers without inventing future targets. It preserves the observed prefix. "
  "Use finish_training() on the ordinary model for the matching frozen oracle. inference_copy(memory='continue') applies the transition "
  "to an independent snapshot; ordinary cold/warm evaluation copies retain the source's visible-parameter contract and do not commit original jobs.")
 policies=("cold_adaptive and cold_frozen start with empty episodes; warm_adaptive and warm_frozen copy the source store. "
  "Adaptive permits causal admission from the observed prefix, while frozen prevents writes. Query/recurrent state and clocks still evolve. "
  "All four policies are tested; paired episode contents, identities, publications and recall behavior match the frozen comparator.")
 checkpoint=("Version 47 saves fitted parameters and a dedicated forward-state payload, including current/pending publications, pattern/frame "
  "metadata, context/bottom memory, slow-Y values, clocks and memory-write policy. It omits derivative archives and learning jobs. "
  "Irregular chunks, snapshots and save/load preserve continuation. The sampler RNG remains caller-owned.")
 resume=("Training calls are rejected in Python and native run/document APIs until resume_training(reset_context=True). "
  "That resets the prefix, eligibility and epoch clocks, reconstructs bottom training accumulators from fitted parameters, and preserves "
  "learned weights, routing counts, dictionaries and episodic storage. No credit is claimed for earlier inference bytes. "
  "reset() and reset_context() alone leave inference active; learning workers can be enabled again after the explicit return.")
 validation=("106 tests passed at 1M: eight new inference tests covering multiple policies and lifecycle scenarios, 13 bank-epoch regressions, "
  "and 85 existing storage/indexing/scheduler/prediction/frame/paired-memory/default tests. Four additional inference checks passed at 5M. "
  "Coverage includes 16,385 forward continuation targets, lightweight frames, one/two context heads, ancestry/event policies, partial "
  "worker epochs, all memory write policies, autoregressive restart, no target leakage, frozen parameters and explicit training rejection/reset.")
 compatibility=("Old v40/v45/v46 training checkpoints import bit-for-bit. Against the previous 085e0a6 build, 1,025-target training continuation "
  "is exact for the full reference; the event/lightweight and selected variants have maximum logit drift 3.8147e-6 and fitted-weight drift "
  "below 4.8e-7, with exact structural identities and clocks. The arithmetic policy is unchanged; cross-build floating-point identity is "
  "not asserted for those variants. Within the new build, frozen versus forward-only continuation is bitwise exact.")
 sanitizer=("ASan and UBSan passed native reference/event/epoch/lightweight-frame runs, 8,193-step continuations, forward export/import, "
  "document drain, reset and resumed training, including worker entry. Leak detection was disabled for the host environment. "
  "Inspection is read-only at API boundaries; no profiling counters or timers were added to the byte loop.")
 remaining=("Next: Action 8, resident GPU backend and independent-stream batching; Action 9, longer 1M/5M/100M qualification, "
  "joint tuning and final profile selection. The selected repeated-text inference result of 16,154 B/s is separate from the still-open "
  "15,000 B/s single-stream training target. The measured Action-6 training profile remains the training recommendation.")
 code="""i = model.inference_copy(memory='warm_frozen')
i.run(prefix_bytes)  # NumPy uint8 array; mode='eval'
i.save('inference.npz')
i = Model.load('inference.npz')
i.resume_training(reset_context=True)"""
 def md_table(head,data):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(row)+' |' for row in data])
 sections=[('Implementation',scope),('Measurement protocol',protocol),('Data and profiles',data),('Memory',memory),('Quality',quality),
           ('Entry transition',transition),('Episodic memory',policies),('Checkpoint and continuation',checkpoint),('Return to training',resume),
           ('Validation',validation),('Historical compatibility',compatibility),('Sanitizers',sanitizer),('Remaining work',remaining)]
 source=f"Baseline: 085e0a60662aa8419bab7ea7f5db3e7d5c1f79c1. Runtime source SHA-256: {manifest['runtime_sha256']}."
 md=['# SHoEbox Action 7: dedicated forward-only runtime','14 November 2026 (America/Denver)',outcome,source,
     '## 1M comparison',md_table(header,rows),'## 5M natural-text cross-check',md_table(header,scale)]
 for title,body in sections:md+=['## '+title,body]
 md+=['## Use','```python\nfrom shoebox_reference import Model\n'+code+'\n```',
      '## Reproduce','See `INFERENCE_POLICY.md`, `src/measure_inference.py`, `tests/test_inference.py`, `tests/check_inference_legacy.py` and `tests/run_inference_sanitizer.py`. Compact raw evidence is in `results/action7_*`. Deliverables exclude datasets, trained checkpoints and compiled binaries.']
 dest=ROOT/'deliverables';dest.mkdir(exist_ok=True);stem=dest/'SHoEbox_Action7_Forward_Only';stem.with_suffix('.md').write_text('\n\n'.join(md)+'\n')
 fontdir=Path('/usr/share/fonts/truetype/dejavu')
 for name,file in [('ReportSans','DejaVuSans.ttf'),('ReportBold','DejaVuSans-Bold.ttf'),('ReportMono','DejaVuSansMono.ttf')]:
  pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
 styles=getSampleStyleSheet()
 styles['Title'].fontName=styles['Heading2'].fontName='ReportBold'
 styles.add(ParagraphStyle(name='Body7',fontName='ReportSans',fontSize=9,leading=12.5,spaceAfter=8))
 styles.add(ParagraphStyle(name='Small7',fontName='ReportSans',fontSize=7.5,leading=10,spaceAfter=5))
 story=[]
 def para(t,small=False):story.append(Paragraph(escape(t),styles['Small7' if small else 'Body7']))
 def heading(t):story.append(Paragraph(escape(t),styles['Heading2']))
 def table(rows):
  t=Table([[Paragraph(escape(v),styles['Small7']) for v in row] for row in [header]+rows],colWidths=[62,65,72,74,51,106],repeatRows=1)
  t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce9ef')),('VALIGN',(0,0),(-1,-1),'TOP'),
    ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#527080')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d6e0e5')),
    ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
  story.extend([t,Spacer(1,9)])
 story.append(Paragraph('SHoEbox / Action 7',styles['Title']));para('Dedicated forward-only runtime | 14 November 2026')
 para(outcome);para('Baseline 085e0a6 | Runtime hash '+manifest['runtime_sha256'][:16],True)
 heading('1M comparison');table(rows);heading('5M natural-text cross-check');table(scale);para(quality);para(memory)
 story.append(PageBreak());heading('Forward state and transitions');para(scope);para(transition);para(policies);para(checkpoint);para(resume)
 story.append(Preformatted('from shoebox_reference import Model\n'+code,ParagraphStyle(name='Code7',fontName='ReportMono',fontSize=8,leading=11)))
 story.append(PageBreak());heading('Measurement and validation');para(protocol);para(data);para(validation);para(compatibility);para(sanitizer)
 heading('Remaining work');para(remaining);para('Reproduce using INFERENCE_POLICY.md, src/measure_inference.py and tests/test_inference.py. Raw evidence and source hashes are under results/action7_*. No datasets, trained checkpoints or binaries are included.',True)
 def footer(canvas,doc):
  canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor('#526875'));canvas.drawString(44,25,'SHoEbox | Action 7 | 14 November 2026');canvas.drawRightString(A4[0]-44,25,str(doc.page))
 SimpleDocTemplate(str(stem.with_suffix('.pdf')),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=36,bottomMargin=42,title='SHoEbox Action 7: Forward-only runtime').build(story,onFirstPage=footer,onLaterPages=footer)
 (ROOT/'results/action7_summary.json').write_text(json.dumps(dict(outcome=outcome,one_m=one['summary'],five_m=five['summary'],tests_1m=106,tests_5m=4,baseline=manifest['baseline_revision'],runtime_sha256=manifest['runtime_sha256'],remaining=[8,9]),indent=2)+'\n')
 print(outcome)
if __name__=='__main__':main()
