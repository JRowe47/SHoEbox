"""Render the fifth repair and compact verification evidence as Markdown/PDF."""
from pathlib import Path
import html,json
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
data=json.loads((ROOT/'results/paired_memory_validation.json').read_text())
timing=json.loads((ROOT/'results/paired_memory_timing.json').read_text())
checks=json.loads((ROOT/'results/paired_memory_checks.json').read_text())
old=data['runs']['independent_transitions'];new=data['runs']['paired_episodes'];stats=new['paired']
before=timing['runs']['independent_transitions'];after=timing['runs']['paired_episodes']
change=100*(after['median_bytes_per_second']/before['median_bytes_per_second']-1)
intro=('Fix 5 is implemented and enabled in fresh reference models. EC-C and EC-X now share complete content/context episodes and one retrieval decision. '
       'Both learned return ports receive members of the same immutable episode. The neural parameter count remains 1,031,952; episodic records are additional memory state.')
sections=[
('Asynchronous observations, explicit pairing','After a regional pass, the latest completed EC-C observation pairs with the latest completed EC-X observation. Both must exist in the current document stream. A new occurrence tuple is considered once; a held member retains its original sequence, tick and primary boundaries. Stored values are detached observed EC inputs, excluding recurrent state and recalled returns.'),
('One causal retrieval transaction','The first EC request in a byte or drain step queries with its current observed input and the latest completed opposite-side observation. It locks one episode ID, similarity gate and both return vectors for that step. Later requests reuse this pair. If the first request lacks the opposite side, both returns remain empty. Admission happens after the regional pass, before target scoring.'),
('Joint addresses and whole-episode replacement','Joint similarity averages the two key cosine scores; a signed joint projection supplies sparse bucket probes. Defaults allow 8,192 complete pairs. Novelty, outcome surprise, salience-weighted reservoir priority and shared utility govern admission through stable/adaptive bucket namespaces. Each episode occupies one row. Reinforcement updates metadata without blending values. Replacement assigns a new ID to both members; there is no FIFO, sliding window or age expiry.'),
('Credit and document boundaries','The existing EC return matrices remain separately learned. Either port can update shared episode utility. Delayed credit checks the saved slot and immutable ID; retired episodes still train the return matrices from their saved values but cannot teach a replacement’s utility. Clearing memory preserves the serial allocator. Stream reset clears pending observations and advances the epoch while preserving stored episodes.'),
('Checkpoints and evaluation','Checkpoint v39 adds a versioned paired-memory policy and sidecar; composition ABI 10 and existing raw neural/state layouts remain unchanged. Snapshots and fingerprints include episodes, asynchronous observations and cached read identity. Imports validate the sidecar before replacing live memory. Warm evaluation copies episodes into a fresh pairing stream. Frozen memory blocks admission, reinforcement, competition and utility changes.'),
('Historical continuation and explicit upgrade','Versions 35-38 retain their independent transition stores. An actual v38 checkpoint from base commit 3d17b4e reproduced 129 continuation bytes, logits, weights, composition, neural ownership, content eligibility and pattern state bit-for-bit. To upgrade explicitly: reset_context(), clear_memory(), configure_paired_memory(True). Policy changes require a fresh stream and empty memory.')]
verification=(f"{checks['regression_tests']} distinct regression tests passed, including 15 new paired-memory tests; all 15 also passed at 64 experts. "
              'ASan/UBSan passed training, draining, memory import/export, resets and a live semantic DAG above 16,384 entries. '
              'Current checkpoints, snapshots and chunked training continue exactly; production and diagnostic logits match.')
acceptance=('Controlled cases cover content-first/context-first writes, held-member provenance, same-step coalescing, read-before-write causality, context disambiguation at identical content, transaction locking, immutable reinforcement, replacement and stale-credit rejection, learned port updates, document isolation, frozen memory, target independence and malformed-state rejection.')
protocol=(f"Two fresh 1M reference models, seed 7, one pinned CPU, {timing['warmup_targets']:,} warmup targets. "
          f"Each of {timing['repeats']} timed rounds trains {timing['targets_per_round']:,} targets from an exact copy of the same warmed state, with alternating order. "
          'Only the paired-memory policy differs. Compilation, snapshot copying and warmup are excluded. LR is 0.04; local LR is 0.04/3.')
rows=[['Median bytes/s',f"{before['median_bytes_per_second']:.2f}",f"{after['median_bytes_per_second']:.2f}"],
      ['Median bytes/CPU-second',f"{before['median_cpu_bytes_per_second']:.2f}",f"{after['median_cpu_bytes_per_second']:.2f}"],
      ['Training NLL after 3,072 targets',f"{old['rounds'][-1]['nll']:.6f}",f"{new['rounds'][-1]['nll']:.6f}"],
      ['Held-out NLL after 3,072 targets',f"{old['heldout']['nll']:.6f}",f"{new['heldout']['nll']:.6f}"]]
measured=(f"Median throughput changed {change:+.1f}% in the repeated-state measurement. "
          'The separate 3,072-target mechanism run uses repeated sentences and separate synthetic held-out prose. '
          'An initial evolving-state timing had substantial round-to-round variance; its raw measurements remain in the evidence file. '
          'All missing neural, content and feedback producers, overwritten credit, expired semantic handles and capacity-pruning counters remained zero.')
observed=(f"The mechanism run admitted {stats['admissions']:,} episodes, retained {stats['occupied']:,}, replaced {stats['replacements']:,}, and made {stats['hits']:,} joint recalls. "
          f"It delivered {stats['utility_credit']:,} utility updates and rejected {stats['retired_utility_handles']:,} retired utility handles without redirecting credit. "
          f"Content/context publications were {stats['content_publications']:,}/{stats['context_publications']:,}, exercising asynchronous pairing. "
          f"The occupied paired sidecar occupied {new['paired_snapshot_bytes']:,} uncompressed bytes.")
contribution=(f"The familiar-prefix frozen probe made {new['familiar']['paired']['hits']} joint recalls. Suppressing both returns changed NLL by "
              f"{new['familiar_recall_delta_nll']:+.9f} nats/byte and logits by RMS {new['familiar_recall_logit_rms']:.9f}. "
              'The separate held-out probe made no accepted recalls and had identical logits with returns suppressed. '
              'The measured downstream effect in this short run is confined to the familiar prefix.')
commands=['python -m unittest discover -s experiments/identity_semantics/tests -v',
          'SHOEBOX_EXPERTS=64 python -m unittest discover -s experiments/identity_semantics/tests -p test_paired_memory.py -v',
          'python experiments/identity_semantics/tests/run_completion_sanitizer.py',
          'python experiments/identity_semantics/src/validate_paired_memory.py',
          'python experiments/identity_semantics/src/validate_paired_memory.py --repeatable-timing --targets 512 --repeats 4 --output experiments/identity_semantics/results/paired_memory_timing.json']
def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(row)+' |' for row in rows])
md='# SHoEbox: paired content/context episodic memory\n\n13 November 2026 | Fifth recommended fix | Base: 3d17b4e\n\n'+intro+'\n\n'
for title,body in sections:md+='## '+title+'\n\n'+body+'\n\n'
md+='## Verification\n\n'+verification+'\n\n'+acceptance+'\n\n'
md+='## Measurements\n\n'+protocol+'\n\n'+mdtable(['Measure','Independent transitions','Paired episodes'],rows)+'\n\n'+measured+'\n\n'+observed+'\n\n'+contribution+'\n\n'
md+='## Reproduce\n\n```bash\n'+'\n'.join(commands)+'\n```\n\nEvidence: results/paired_memory_validation.json, paired_memory_timing.json, paired_memory_checks.json, paired_memory_legacy38.json and paired_memory_sanitizer.txt.\n'
(OUT/'SHoEbox_Paired_Episodic_Memory.md').write_text(md)
pdfmetrics.registerFont(TTFont('ReportSans','/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'))
pdfmetrics.registerFont(TTFont('ReportBold','/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCompact',fontName='ReportSans',fontSize=9.2,leading=12.2,spaceAfter=7,textColor=colors.HexColor('#253348')))
styles.add(ParagraphStyle(name='Label',fontName='ReportSans',fontSize=8,leading=10.3,spaceAfter=7,textColor=colors.HexColor('#58677c')))
for key in ('Title','Heading2'):styles[key].textColor=colors.HexColor('#172f4d');styles[key].fontName='ReportBold'
styles['Title'].fontSize=21;styles['Title'].leading=25;styles['Title'].alignment=0
styles['Heading2'].fontSize=11.5;styles['Heading2'].leading=14;styles['Heading2'].spaceBefore=7;styles['Heading2'].spaceAfter=4
story=[]
def para(text,style='BodyCompact'):story.append(Paragraph(html.escape(text),styles[style]))
def table(headers,rows):
    values=[[Paragraph(html.escape(v),styles['BodyCompact']) for v in row] for row in [headers]+rows]
    t=Table(values,colWidths=[211,144,144],repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f1f5')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#cad6e0')),('LEFTPADDING',(0,0),(-1,-1),6),
        ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),0)]))
    story.append(t);story.append(Spacer(1,8))
para('Paired episodic memory','Title');para('SHoEbox | 13 November 2026 | Recommendation 5 | Base 3d17b4e','Label');para(intro)
for title,body in sections:para(title,'Heading2');para(body)
story.append(PageBreak());para('Verification and measurements','Title');para(verification);para(acceptance,'Label')
para('Matched throughput','Heading2');para(protocol);table(['Measure','Independent transitions','Paired episodes'],rows);para(measured)
para('Observed memory behavior','Heading2');para(observed);para(contribution)
para('Reproduce','Heading2')
for command in commands:para(command,'Label')
def footer(canvas,doc):
    canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor('#647488'))
    canvas.drawString(48,27,'SHoEbox | Paired episodic memory | v39 / ABI 10');canvas.drawRightString(A4[0]-48,27,str(doc.page))
SimpleDocTemplate(str(OUT/'SHoEbox_Paired_Episodic_Memory.pdf'),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=36,bottomMargin=43,title='SHoEbox paired episodic memory',author='SHoEbox').build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT/'SHoEbox_Paired_Episodic_Memory.pdf')
