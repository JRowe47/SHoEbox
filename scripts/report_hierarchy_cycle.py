"""Render the matched regional-cycle experiment as a concise Markdown/PDF report."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--joint', type=Path, required=True)
    p.add_argument('--cycle', type=Path, required=True)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    joint, cycle, result = [json.loads(path.read_text()) for path in
                            (args.joint, args.cycle, args.comparison)]
    audits = [json.loads((path.parent/'checkpoint_audit.json').read_text()) for path in (args.joint, args.cycle)]
    if not all(audit['exact_score_reproduction'] for audit in audits):
        raise ValueError('Checkpoint reproduction is required before reporting.')
    args.out.mkdir(parents=True, exist_ok=True)
    name = 'SHoEbox_Regional_Training_Report'
    j, c = joint['evaluation'][-1], cycle['evaluation'][-1]
    delta = result['cycle_minus_joint_nll']
    outcome = ('The regional cycle improved held-out next-byte prediction over matched joint training.'
               if delta < 0 else
               'The regional cycle did not improve held-out next-byte prediction over matched joint training.')
    decision = ('Keep the cycle available as the next training candidate; this comparison measures the combined regimen.'
                if delta < 0 else
                'Keep joint AdamW as the default. The cycle remains an opt-in experiment; do not promote this schedule on this result.')
    metrics = [
        ('Next-byte NLL', j['nll'], c['nll']),
        ('Next-four-byte NLL', j['span_nll']['mid'], c['span_nll']['mid']),
        ('Next-sixteen-byte NLL', j['span_nll']['high'], c['span_nll']['high']),
        ('Decoder-summary swap delta', joint['upper_swap_delta_nll'], cycle['upper_swap_delta_nll']),
        ('Decoder + feedback swap delta', joint['upper_all_swap_delta_nll'], cycle['upper_all_swap_delta_nll'])]
    table = '\n'.join(f'| {label} | {a:.6f} | {b:.6f} |' for label, a, b in metrics)
    protocol = result['protocol']
    initial = result.get('initial_audit')
    initial_note = (f"The starting checkpoint scores {initial['ordinary']['nll']:.6f} nats/byte. "
                    f"Swapping all outward upper messages adds "
                    f"{initial['all_upper']['nll']-initial['ordinary']['nll']:.6f} nats/byte; "
                    f"feedback-only swapping adds {initial['feedback_only']['nll']-initial['ordinary']['nll']:.6f}."
                    if initial else '')
    md = f'''# SHoEbox regional training cycle

November 8, 2026. Six-region predictive hierarchy; 3,050,848 parameters.

{outcome} Cycle minus joint: **{delta:+.6f} nats/byte**. {decision}

## Matched result

Both arms start from the same 256-update checkpoint, including Adam moments,
runtime state and corpus position. Each completes **{result['additional_updates']:,} additional updates /
{result['additional_targets']:,} additional byte targets**: {result['lifetime_updates']:,} lifetime updates and
{result['lifetime_targets']:,} lifetime target exposures. Held-out evaluation covers **{result['heldout_targets']:,}
targets across 23 separate documents**. All prediction losses are nats/byte; lower is better.

| Measurement | Joint AdamW | Regional cycle |
|---|---:|---:|
{table}

Swap deltas are intervention NLL minus ordinary NLL; positive means mismatched
upper messages worsen prediction. Decoder-only swapping leaves feedback and
producer trajectories intact. The combined intervention also swaps top-down
contexts into A1/B1/L2/S2 and recomputes their trajectories from fresh states.

{initial_note}

## Implemented regimen

| Phase | Updates per cycle | Learning parameters |
|---|---:|---|
| Upper development | 32 | L2/S2/L3, future heads, Y and nonlocal decoder |
| Lower alignment | 32 | Byte embedding and A1/B1 |
| Joint learning | 64 | All; fast producers and local projection use 0.5x learning rate |

Both future objectives must improve by 2% on their training EMA before lower
alignment is enabled. The existing checkpoint already meets this condition.
Every fourth upper-phase chunk scales the decoder's direct local slot to 0.5;
producer inputs remain intact. The endpoint uses every route at full strength.
Executed phases: {result['phases']}; attenuated chunks: {result['attenuated_updates']}.

Freeze/thaw preserves Adam moments and stops both momentum updates and weight
decay for frozen parameters. States continue evolving, and gradients pass
through frozen modules to their trainable inputs. No additional replay or
regional execution dropping is introduced. Joint training remains the default.

## Validation and reproduction

**59 checks pass**, including frozen parameter/moment invariance, credit in both
directions through frozen regions, producer-preserving decoder attenuation,
feedback intervention causality, exact optimizer migration and exact resume.
A separate six-update command-line pause/resume run reproduces model, optimizer,
state, RNG, corpus position and controller exactly across all three phases.
Both final checkpoints exactly reproduce their complete held-out byte and span
scores after reload; parameters and runtime states are finite.

Training uses B8/T128, AdamW 0.001 with betas (0.9, 0.95), 0.01 matrix weight decay,
gradient clipping at 1, and unchanged 0.2/0.1 future-loss weights. There are
194,560 unique training targets; the corpus repeats. Runs share a CPU host and
use one CPU thread each. Observed aggregate training throughput:
{joint['training_bytes_per_second']:.2f} bytes/s joint and {cycle['training_bytes_per_second']:.2f} bytes/s cycle.
These timings are not an isolated hardware benchmark.

This is one matched continuation, with two schedule cycles. Rotation, fast-group
learning rates and local-route attenuation change together. Per-component effects
are not separated. GradNorm and adaptive phase lengths are not implemented here.

Reproduction commands and parameter roles: `docs/regional_training.md`.
Machine-readable comparison: `docs/regional_training_results.json`.
Source digest: `{protocol['source_sha256']}`.
Initial checkpoint SHA-256: `{protocol['initial_checkpoint_sha256']}`.
'''
    (args.out/(name+'.md')).write_text(md)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 2.45), dpi=180)
    colors = ('#3569ad', '#c06a27')
    for run, audit, label, color in zip((joint, cycle), audits, ('Joint AdamW', 'Regional cycle'), colors):
        rows = [r for r in run['evaluation'] if not r['full']]+[audit['monitoring_endpoint']]
        ax.plot([r['updates'] for r in rows], [r['nll'] for r in rows], marker='o', ms=3,
                label=label, color=color, linewidth=1.7)
    for start in (0, 128):
        ax.axvspan(start, start+32, color='#c06a27', alpha=.07)
        ax.axvspan(start+32, start+64, color='#3569ad', alpha=.06)
    ax.set(xlabel='Additional optimizer updates', ylabel='Validation NLL',
           title='Fixed 8,192-target monitoring window; full endpoint scores are in the table')
    ax.set_xlim(0, result['additional_updates'])
    ax.grid(alpha=.18); ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    curve = args.out/'SHoEbox_Regional_Training_Curve.png'
    fig.savefig(curve); plt.close(fig)

    from reportlab.lib import colors as rc
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    fonts = Path(matplotlib.get_data_path())/'fonts/ttf'
    pdfmetrics.registerFont(TTFont('ReportSans', str(fonts/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('ReportSansBold', str(fonts/'DejaVuSans-Bold.ttf')))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodySmall', fontName='ReportSans', fontSize=9.5, leading=13,
                             spaceAfter=8, textColor=rc.HexColor('#283644')))
    styles.add(ParagraphStyle(name='SmallNote', fontName='ReportSans', fontSize=8, leading=11,
                             spaceAfter=6, textColor=rc.HexColor('#526174')))
    styles['Title'].fontSize=22; styles['Title'].leading=25; styles['Title'].alignment=TA_LEFT
    styles['Title'].fontName='ReportSansBold'
    styles['Title'].textColor=rc.HexColor('#16384d')
    styles['Heading2'].fontSize=12; styles['Heading2'].textColor=rc.HexColor('#16384d')
    styles['Heading2'].fontName='ReportSansBold'
    def para(text, style='BodySmall'):
        return Paragraph(escape(text), styles[style])
    def make_table(rows, widths):
        t=Table([[para(str(cell),'SmallNote') for cell in row] for row in rows], colWidths=widths)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),rc.HexColor('#e7eff4')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[rc.white,rc.HexColor('#f5f7f9')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),
            ('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),
            ('BOTTOMPADDING',(0,0),(-1,-1),3)]))
        return t
    flow=[para('SHoEbox regional training cycle','Title'),
          para('MATCHED CONTINUATION  |  NOVEMBER 8, 2026','SmallNote'),
          para(outcome+f' Cycle minus joint: {delta:+.6f} nats/byte.'),
          para(f"Each arm: {result['additional_updates']} additional updates / {result['additional_targets']:,} byte targets. "
               f"Same starting weights, optimizer and state. Evaluation: {result['heldout_targets']:,} targets across 23 held-out documents."),
          make_table([['Measurement (nats/byte)','Joint AdamW','Regional cycle']]+
                     [[label,f'{a:.6f}',f'{b:.6f}'] for label,a,b in metrics],[282,111,111]),Spacer(1,10),
          para('Positive swap deltas mean mismatched upper messages worsen prediction. The combined intervention includes top-down feedback; decoder-only swapping leaves producer trajectories intact.','SmallNote'),
          para(initial_note, 'SmallNote'),
          Image(str(curve),width=504,height=154.35),Spacer(1,8),para(decision),PageBreak(),
          para('What changed and what was checked','Title'),
          make_table([['Phase','Updates','Learning parameters'],
                      ['Upper development','32','L2/S2/L3, future heads, Y and nonlocal decoder'],
                      ['Lower alignment','32','Byte embedding and A1/B1'],
                      ['Joint learning','64','All; fast producers and local projection at half learning rate']],
                     [130,55,319]),Spacer(1,12),
          para('Readiness and decoder access','Heading2'),
          para('Both future objectives must improve by 2% on their training EMA before lower alignment begins. '
               'The starting checkpoint meets this condition. Every fourth upper-phase chunk halves the direct local decoder slot; '
               'inputs to producer regions remain intact. Evaluation restores the full route.'),
          para(f"Executed phases: {result['phases']}. Attenuated chunks: {result['attenuated_updates']}."),
          para('State and gradient semantics','Heading2'),
          para('Frozen parameters keep their Adam moments and do not undergo weight decay. Recurrent states keep evolving, '
               'and gradients pass through frozen modules to trainable inputs. No additional replay or regional execution dropping is added.'),
          para('Verification','Heading2'),
          para('All 59 checks pass. A separate command-line split run exactly reproduces uninterrupted weights, optimizer, '
               'runtime state, RNG, corpus cursor and controller across upper, lower and joint phases. '
               'Both final checkpoints exactly reproduce all full held-out byte and span scores after reload.'),
          para('Protocol and interpretation','Heading2'),
          para('B8/T128; AdamW 0.001, betas (0.9, 0.95), matrix decay 0.01, clipping at 1. '
               'Future-loss weights remain 0.2/0.1. The 194,560 unique training targets repeat. '
               'This is one matched continuation with two cycles; phase rotation, regional rates and attenuation change together.'),
          para(f"Aggregate training throughput: {joint['training_bytes_per_second']:.2f} bytes/s joint; "
               f"{cycle['training_bytes_per_second']:.2f} bytes/s cycle. Runs shared a CPU host with one thread each; these are not isolated hardware timings.",'SmallNote'),
          para('Joint AdamW remains the default. GradNorm and adaptive phase lengths are outside this implementation. '
               'Commands: docs/regional_training.md. Full protocol and checksums are included in the Markdown report.','SmallNote')]
    def footer(canvas, doc):
        canvas.setFont('ReportSans',8); canvas.setFillColor(rc.HexColor('#657386'))
        canvas.drawString(54,30,'SHoEbox | Regional training experiment')
        canvas.drawRightString(558,30,str(doc.page))
    SimpleDocTemplate(str(args.out/(name+'.pdf')),pagesize=letter,leftMargin=54,rightMargin=54,
                      topMargin=42,bottomMargin=48).build(flow,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps(dict(markdown=str(args.out/(name+'.md')),pdf=str(args.out/(name+'.pdf')),curve=str(curve))))


if __name__ == '__main__':
    main()
