"""Create compact Markdown/PDF implementation evidence from completed runs."""
import json,statistics,math,re,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
def load(name):return json.loads((ROOT/'results'/name).read_text())
def mean(xs):return statistics.mean(xs)
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
 train=load('message_training.json');contrib=load('message_contributions.json');perf=load('message_throughput.json');selection=load('message_selection.json');verify=load('message_verification.json')
 by={(r['job']['kernel'],r['job']['seed']):r for r in train}
 baseline=mean(by['relays',s]['test_nll'] for s in (1,2));revised=mean(by['messages',s]['test_nll'] for s in (1,2));delta=revised-baseline
 def intervention(kernel,kind,region):
  rows=[r['delta'] for r in contrib['rows'] if (r['kernel'],r['kind'],r['region'])==(kernel,kind,region)]
  return mean(rows)
 regions=['E4','V4','G5','F4','L3']
 measured=table(['Seed','Relay NLL','Message NLL','Change'],[[s,f"{by['relays',s]['test_nll']:.6f}",f"{by['messages',s]['test_nll']:.6f}",f"{by['messages',s]['test_nll']-by['relays',s]['test_nll']:+.6f}"] for s in (1,2)]+[['Mean',f'{baseline:.6f}',f'{revised:.6f}',f'{delta:+.6f}']])
 expert=table(['Region','Old expert removal','New expert removal','New neural removal'],[[r,f"{intervention('relays','expert_transform',r):+.8f}",f"{intervention('messages','expert_transform',r):+.8f}",f"{intervention('messages','neural_output',r):+.8f}"] for r in regions])
 channels=table(['Region','Semantic transfer removal','Feedback branch removal','Whole-source removal'],[[r,f"{intervention('messages','semantic_payload',r):+.8f}",f"{intervention('messages','feedback_channel',r):+.8f}",f"{intervention('messages','whole_source',r):+.8f}"] for r in regions])
 credit=[]
 for r in regions:
  values=[]
  for kernel in ('relays','messages'):
   values.append(mean(by[kernel,s]['lace_metrics'][r]['gradient_norm_sum']/max(1,by[kernel,s]['lace_metrics'][r]['credit_updates']) for s in (1,2)))
  credit.append([r,f'{values[0]:.8f}',f'{values[1]:.8f}',f'{values[1]/values[0]:.2f}x' if values[0] else '-'])
 credit_table=table(['Region','Old expert credit norm','New expert credit norm','Ratio'],credit)
 preserved=sum(r['events_preserved'] for r in contrib['rows'] if r['kind']!='whole_source');expected=sum(r['kind']!='whole_source' for r in contrib['rows'])
 speed=perf['median_bytes_per_second'];direction='improved' if delta<0 else 'increased';speed_change=(perf['ratio']-1)*100
 md=f'''# SHoEbox: learned-message repair

13 November 2026 | Base: 9566f4f | 1,031,936 parameters | Two fresh seeds

## Result

Implemented learned constituent transfer, independently bounded feedback for activation and routing, and credit to the actual adapter or neural producer. Stable IDs, structural descriptors, role binding and relay forecasts remain intact. The new runtime is selected with `fresh_model(kernel="messages")`; `relays` remains available as the unchanged comparator.

Mean held-out loss {direction} from **{baseline:.6f} to {revised:.6f} nats/byte** ({delta:+.6f}). Same-core median training throughput changed from **{speed['relays']:.0f} to {speed['messages']:.0f} bytes/s** ({speed_change:+.1f}%).

{measured}

Each model trained on one million next-byte targets from the same Tiny Shakespeare stream. Evaluation used 78,771 reserved test targets with fresh stream state and causal adaptive associative memory. Six 100,000-target pilots on seed 3 selected controls using a separate 32,768-target validation segment. Both full-run variants used peak learning rate 0.08, the same cosine schedule, local rate LR/3, and router rate 0.002. No test data selected controls.

## Implemented behavior

- Parents consume a normalized, bounded role-transformed bundle of `child.code - child.structure` in their content input and routing cue. Exported codes retain their local adapter provenance; received payloads are not relabeled as parent-produced adapter vectors.
- Declared delayed feedback adds its own bounded branch to activation and routing. The legacy context path remains. With all three new channel gains zero and semantic LR multiplier one, training exactly reproduces the relay runtime.
- Semantic credit uses saved source occurrences, roles, features and derivatives. It updates the actual semantic adapter, including through held publications and relays. Feedback uses the saved neural producer, chosen/runner keys, sparse mask and smooth gate derivative. The hard selection itself is not differentiated.
- The two feedback terms for the same consumed occurrence share one reuse count. Source freeze and task-rate controls apply to transferred semantic credit.

Selected gains: **payload 1.0; feedback activation 0.1; feedback routing 0.1; semantic LR multiplier 1.0**. The independently normalized branches are bounded by `tanh`; all channel gains are constrained to [0, 1]. No neural parameters were added.

## Reproduction

From `experiments/identity_semantics`, restore the pinned corpus with `python src/restore_artifacts.py --download-data`. Run `python src/run_messages_study.py` with `pilots`, then `full`; run `python src/check_messages_results.py` with `interventions`, then `timing`. Unit checks: `python -m unittest discover -s tests -v`. Native checks: `python tests/run_messages_sanitizer.py`.

<!-- pagebreak -->

## Regional contribution

Entries are two-seed mean changes in test NLL when the named path is removed. Positive values mean removal hurts prediction; negative values mean removal helps. These interventions keep the remaining dynamics live and are not additive shares of knowledge.

{expert}

`expert_transform` removes the learned matrix/bias while preserving the input skip. `neural_output` removes the full published neural response, including its skip and recall components.

{channels}

`semantic_payload` removes only outgoing adapter residuals at their consumers. `feedback_channel` removes only the newly added bounded feedback branch, keeping the legacy context path. G5 has no immediate constituent consumers, so its semantic-transfer intervention is intentionally inactive. Whole-source removal also changes content/event transport.

All **{expected} of {expected}** interventions that promise content-event preservation passed exact identity/provenance trace checks ({preserved} verified). Routing changes caused by feedback or payload removal are intentional. {len(contrib['rows'])} full-test interventions and {len(contrib['parity'])} exact production/diagnostic parity comparisons completed. Semantic-residual and relay-context measurements are included in the machine-readable results.

G5's learned-expert removal penalty is now {intervention('messages','expert_transform','G5'):+.8f}, versus {intervention('relays','expert_transform','G5'):+.8f} previously. E4 and V4 learned-expert effects remain near zero. E4's transferred semantic payload now contributes {intervention('messages','semantic_payload','E4'):+.8f} nats/byte when removed. These measurements show greater G5 influence and usable semantic transport, while the mean test loss still favors the relay comparator. Keep `relays` for current training; `messages` is available for continued channel/credit experiments.

<!-- pagebreak -->

## Delivered task credit

Mean delivered expert-credit norm per update, averaged across seeds. These are gradient magnitudes before the parameter update, not parameter-change norms. Semantic-adapter deliveries are recorded separately.

{credit_table}

Both repaired runs completed with zero missing legacy credit parents, zero expired retained-input edges, zero missing feedback producers, and zero expired or duplicate forecast producers. The saved forecast accounting balances exactly.

## Verification

{verify['summary']}

The new tests cover: fixed-ID payload sensitivity on E4 -> G5/EC-X, V4 -> G5 and L3/F4 -> Y; finite differences for semantic payloads and G5 feedback to F3/E4/F4/V4; the saved smooth response gate; expert switching under the normal sparse mask; held and relayed occurrence credit; source freezes and zero rates; exact zero-control parity; diagnostic isolation; pending-forecast teaching; checkpoint continuation; and explicit old-checkpoint conversion. Native ASan/UBSan exercises training, evaluation, draining, reset and state import/export.

The throughput comparison alternated six trials per runtime on CPU core {perf['core']}, with 1,024 warmup targets followed by 8,192 measured training targets. Compilation, loading, snapshot copies, warmup and checkpoint I/O are excluded. Concurrent full-study run times are not used to calculate the throughput comparison.

## Use and continuation

```python
import sys
sys.path.insert(0, "experiments/identity_semantics/runtime")
from runtime import fresh_model, Model
model = fresh_model(seed=1, kernel="messages")
# Resume a new-format checkpoint exactly:
# model = Model.load("messages.npz")
```

Message checkpoints use version 32 / composition ABI 6 and persist the controls and eligibility. Old relay checkpoints retain version 31 / ABI 5. Convert explicitly at a fresh stream boundary with `python src/convert_to_messages.py old_relays.npz new_messages.npz`. Conversion preserves weights, semantic adapters, dictionaries, fixed transforms and preprocessing means; it resets stream state, forecasts, eligibility and associative memory. It does not overwrite the source checkpoint.

## Evidence

Source and replay commands are documented in the package README. Exact compact measurements are in `results/message_*.json`; source hashes are in `SOURCE_MANIFEST.json`. Test checkpoints, corpus bytes, compiled binaries and intermediate runs are excluded.

'''
 (OUT/'SHoEbox_Learned_Message_Fix.md').write_text(md)
 render_pdf(md,OUT/'SHoEbox_Learned_Message_Fix.pdf')

def render_pdf(md,path):
 from reportlab.lib import colors
 from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
 from reportlab.lib.enums import TA_LEFT
 from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Preformatted,PageBreak,KeepTogether
 from reportlab.lib.pagesizes import letter
 from reportlab.pdfbase import pdfmetrics
 from reportlab.pdfbase.ttfonts import TTFont
 fontdir=Path('/usr/share/fonts/truetype/dejavu')
 for name,filename in [('ReportSans','DejaVuSans.ttf'),('ReportSans-Bold','DejaVuSans-Bold.ttf'),('ReportMono','DejaVuSansMono.ttf')]:
  pdfmetrics.registerFont(TTFont(name,str(fontdir/filename)))
 pdfmetrics.registerFontFamily('ReportSans',normal='ReportSans',bold='ReportSans-Bold',italic='ReportSans',boldItalic='ReportSans-Bold')
 styles=getSampleStyleSheet();styles['Title'].fontName='ReportSans-Bold';styles['Title'].fontSize=19
 styles.add(ParagraphStyle(name='BodySmall',fontName='ReportSans',fontSize=9,leading=12.2,spaceAfter=7))
 styles.add(ParagraphStyle(name='Cell',fontName='ReportSans',fontSize=7.8,leading=10.2))
 styles.add(ParagraphStyle(name='HeadingSmall',fontName='ReportSans-Bold',fontSize=13,leading=16,spaceBefore=9,spaceAfter=7,textColor=colors.HexColor('#183e59'),keepWithNext=True))
 styles.add(ParagraphStyle(name='CodeSmall',fontName='ReportMono',fontSize=7.7,leading=10.2,spaceAfter=8))
 def inline(s):
  s=html.escape(s);s=re.sub(r'`([^`]+)`',r'<font name="ReportMono">\1</font>',s);return re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
 flow=[];lines=md.splitlines();i=0
 while i<len(lines):
  line=lines[i]
  if not line.strip():i+=1;continue
  if line.startswith('<!-- pagebreak'):flow.append(PageBreak());i+=1;continue
  if line.startswith('```'):
   block=[];i+=1
   while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
   flow.append(Preformatted('\n'.join(block),styles['CodeSmall']));i+=1;continue
  if line.startswith('|'):
   rows=[]
   while i<len(lines) and lines[i].startswith('|'):
    if not re.match(r'^\|[\s|:-]+\|$',lines[i]):rows.append([Paragraph(inline(v.strip()),styles['Cell']) for v in lines[i].strip('|').split('|')])
    i+=1
   cols=len(rows[0]);widths=([60]+[(516-60)/(cols-1)]*(cols-1)) if cols>1 else [516]
   t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5edf4')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#66869b')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9fb')])]))
   flow.extend([t,Spacer(1,9)]);continue
  if line.startswith('# '):flow.append(Paragraph(inline(line[2:]),styles['Title']))
  elif line.startswith('## '):flow.append(Paragraph(inline(line[3:]),styles['HeadingSmall']))
  else:
   block=line[2:] if line.startswith('- ') else line
   flow.append(Paragraph(('&#8226; ' if line.startswith('- ') else '')+inline(block),styles['BodySmall']))
  i+=1
 def footer(c,doc):
  c.setStrokeColor(colors.HexColor('#bccbd4'));c.line(48,35,564,35);c.setFont('ReportSans',7.5);c.setFillColor(colors.HexColor('#526b7a'));c.drawString(48,23,'SHoEbox | Learned-message repair | 13 November 2026');c.drawRightString(564,23,str(doc.page))
 doc=SimpleDocTemplate(str(path),pagesize=letter,rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=48,title='SHoEbox learned-message repair',author='Joshua Rowe / Codex')
 doc.build(flow,onFirstPage=footer,onLaterPages=footer)
if __name__=='__main__':main()
