"""Render the delayed-recall repair report from committed measurements."""
import json
from pathlib import Path
import re
from statistics import mean
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, Preformatted

HERE = Path(__file__).parent


def table(rows):
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"]*len(rows[0])) + " |"]
                     + ["| " + " | ".join(row) + " |" for row in rows[1:]])


def main():
    evidence = json.loads((HERE / "results/delayed_recall.json").read_text())
    groups = {p:[x for x in evidence["language"] if x["memory_profile"] == p]
              for p in ("addressed_v3", "addressed")}
    old, new = ({k:mean(x[k] for x in groups[p]) for k in ("test_nll", "bytes_per_second", "teacher_forced_inference_bps", "sample_generation_bps")}
                for p in groups)
    recall = [x for x in evidence["delayed"] if x["profile"] == "addressed"]
    previous = [x for x in evidence["delayed"] if x["profile"] == "addressed_v3"]
    task_rows = [["Distractor gap", "Previous v3", "Repaired v4"]]
    for gap in (16,48,96,192,768,4096):
        means = [mean(next(s["byte_accuracy"] for s in r["scores"] if s["gap"] == gap) for r in rows)
                 for rows in (previous, recall)]
        task_rows.append([str(gap), *[f"{100*x:.0f}%" for x in means]])
    ablations = [["Frozen read control", "Answer accuracy (gap 192)"]]
    for mode in ("no_read", "recency_only", "no_fixed_values", "no_learned_values"):
        accuracy = mean(next(s["accuracy"] for s in r["ablations"][mode] if s["gap"] == 192) for r in recall)
        ablations.append([mode, f"{100*accuracy:.1f}%"])
    language = [["Seed", "v3 test NLL", "v4 test NLL", "v3 bytes/s", "v4 bytes/s"]]
    for seed in (7,17,29):
        a,b = [next(x for x in groups[p] if x["seed"] == seed) for p in groups]
        language.append([str(seed), f"{a['test_nll']:.5f}", f"{b['test_nll']:.5f}", f"{a['bytes_per_second']:,.0f}", f"{b['bytes_per_second']:,.0f}"])
    language.append(["Mean", f"{old['test_nll']:.5f}", f"{new['test_nll']:.5f}", f"{old['bytes_per_second']:,.0f}", f"{new['bytes_per_second']:,.0f}"])
    broader = [["Task / seed", "Gap 16", "Gap 192", "Gap 768"]]
    for row in evidence["extra_tasks"]:
        name = "Two cues" if row["task"] == "two_cues" else "Varied distractors"
        broader.append([f"{name} / {row['seed']}", *[f"{100*next(x['byte_accuracy'] for x in row['scores'] if x['gap']==gap):.1f}%" for gap in (16,192,768)]])
    continuous = mean(x["continuous_byte_accuracy"] for x in recall)
    delta = new["test_nll"] - old["test_nll"]
    speed = 100*(new["bytes_per_second"]/old["bytes_per_second"]-1)
    md = f'''# SHoEbox delayed-recall repair

14 November 2026 | Base: `51030d6` | Native policy: version 4

## Result

The original delayed AB/BA answer failure is repaired: full 256-byte-vocabulary answer accuracy improves from 50% to 100% across seeds 7, 17 and 29 at the same 6,000-example training budget. The repaired model passes all eight prefix shifts and repeated-distractor gaps through 4,096 bytes. Removing either retained retrieval or learned query selection returns this task to chance.

This is a repair of retained-unit selection and delivery. Varied changing distractors remain unresolved at this budget, and original-writer credit still stops at the local derivative boundary. The measured language cost is {delta:+.5f} nats/byte and {speed:.1f}% training throughput.

## What was missing and what changed

**Retained identity.** Each scale now keeps at most 16 ordered pairs of consecutive completed contents. The byte-scale pairs overlap, preserving a cue that crosses a four-byte grouping boundary. Fixed order-sensitive sketches provide addresses; values contain both learned content features and fixed features. Consecutive identical pairs do not overwrite retained units or advance the occurrence clock. A returning pair refreshes its occurrence; a new pair evicts the least recent slot when full.

**Present-question selection.** After every observed byte, a learned query from that byte and its fast contextual state reads every scale's retained units. Slow memory can therefore respond before its next expert/composition event. Attention combines fixed-key similarity with an occurrence-recency bias. All records contain observed input only; the next-byte target is unavailable to admission and querying.

**Direct delivery and local learning.** Retrieved values contribute directly to the existing shared binary byte decoder. This avoids forcing their effect through stale cached slow states and another tanh. The answer error teaches query selection and retained-value decoder coefficients online. Its query-state derivative enters the existing fast local block. The stored pair content is detached: this does not backpropagate into a retired writer or across parent/child boundaries.

There are no special cases for AB, BA, x, question marks, answer digits or episode boundaries in the runtime. Training still uses ordinary next-byte targets. Existing compact MinKron/addressed matrices, sparse expert routing, native content composition and local auxiliary objectives remain active. Retained recall is enabled by default, separately from the optional exact-payload association API.

<!-- pagebreak -->

## Matched delayed-answer evidence

Both versions train on the same continuous 6,000-example stream per seed, with gaps 16/48/96 and explicit prefix lengths 0-3. Frozen evaluation uses balanced counterfactual AB/BA pairs with identical suffixes, prefix lengths 0-7, and the gaps below. Each cell aggregates 48 answer trials (16 per seed). Explicit prefix lengths 4-7 are absent from training, although concatenation covers all native grouping phases.

{table(task_rows)}

These are full-vocabulary argmax answers, not only a forced choice between 0 and 1. A separate continuous frozen evaluation on 256 shuffled cases per seed also reaches {100*continuous:.0f}%; training/evaluation reset mismatch does not explain the result.

{table(ablations)}

The learned and fixed payload halves each support the simple task when the other half is removed. Selection is essential: recency alone does not recover the answer. This separates useful stored information from successfully selecting and delivering it.

## Language and throughput cost

Fresh nominal-1M models train on the same first 1,000,000 Tiny Shakespeare targets. The final contiguous 10% supplies 111,539 held-out targets. A single worker is pinned to CPU 0; execution order alternates by seed. Compilation, evaluation and checkpoint I/O are excluded from training timing; terminal local updates are included.

{table(language)}

Mean held-out loss changes by {delta:+.5f} nats/byte. Training retains {100*new['bytes_per_second']/old['bytes_per_second']:.1f}% of v3 throughput. Teacher-forced inference averages {new['teacher_forced_inference_bps']:,.0f} versus {old['teacher_forced_inference_bps']:,.0f} bytes/s; sampled generation averages {new['sample_generation_bps']:,.0f} versus {old['sample_generation_bps']:,.0f} bytes/s. The repair establishes delayed-answer capability, not a language-quality improvement.

<!-- pagebreak -->

## Broader boundary and remaining work

Two additional fresh-model tasks use 6,000 examples per seed. The two-cue task presents AB/BA and CD/DC and uses either ? or ! to request the corresponding answer. The varied-distractor task replaces repeated x with random bytes from x/y/z. These are development diagnostics, with 64 and 16 trials per cell respectively; they are not claimed as passed gates.

{table(broader)}

Pair changes advance the recency clock even when the alphabet is small; enough distinct content can also evict an earlier cue. Thus a very long repeated gap and a long changing sequence are different retention demands. Selective protection under interference, richer reusable composition, and local reinforcement of retired originating representations remain outstanding. This repair does not change the fact that upward publication carries current content rather than recalled context or forecasts.

## Validation and lifecycle

Passed 2,060 native checks and 17 Python tests, including independent numerical derivatives of query selection and decoder delivery, read causality, bounded slots, pair order, off-boundary slow-memory querying, a learned delayed-answer regression, and exact partial-block checkpoint continuation. AddressSanitizer and UndefinedBehaviorSanitizer pass. LeakSanitizer is explicitly disabled on this ptrace-managed host.

New models use checkpoint version 4. Version-3 native files are copied byte-for-byte from the base revision; versions 1 and 2 also remain unchanged. Loading old checkpoints selects the saved historical backend. Cold context clears causal retained pairs while preserving explicit persistent bindings; warm copies and checkpoints retain pair records and current read caches. Frozen inference admits newly observed pairs but never changes fitted weights.

The four-scale v4 model has 999,615 fitted scalars and expert banks [98,49,29,20], versus v3's 997,183 scalars and [104,52,31,21]. Retained queries/decoder coefficients cost 53,120 fitted scalars; the budget allocator compensates with fewer experts. The four retained streams occupy 32,352 bytes, excluding read caches. Reads scan at most 16 slots per scale and do not execute another expert. The minimum four-scale budget is 205,503 scalars; use 250,000 for small tests.

## Reproduction

```bash
python multiscale/tests/run_all.py --no-leak-check
python -m multiscale.compare_delayed_recall --data input.txt --out /tmp/recall --extra-tasks
python -m multiscale.build_delayed_report
```

Measurements: `results/delayed_recall.json`. Verification: `results/delayed_recall_verification.txt`. Core changes: `recall.hpp`, `native.cpp`, `native.hpp`, `model.py`. Historical controls: `addressed_v3/`. Source, tests and compact evidence are committed; datasets, trained checkpoints, binaries and temporary experiments are excluded.

Hardware: AMD EPYC 9V74; one CPU worker. Python {evidence['python']}, NumPy {evidence['numpy']}; C++17, -O3, -march=native, -ffp-contract=off. Dataset SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
'''
    (HERE / "DELAYED_RECALL_REPAIR.md").write_text(md)
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    for name, file in (("Report", "DejaVuSans.ttf"), ("ReportBold", "DejaVuSans-Bold.ttf"), ("ReportMono", "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(font_dir/file)))
    pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold", italic="Report", boldItalic="ReportBold")
    styles = {
        "body": ParagraphStyle("body", fontName="Report", fontSize=9.1, leading=12.0, textColor=colors.HexColor("#253348"), spaceAfter=5),
        "h1": ParagraphStyle("h1", fontName="ReportBold", fontSize=23, leading=28, textColor=colors.HexColor("#17334d"), spaceAfter=12),
        "h2": ParagraphStyle("h2", fontName="ReportBold", fontSize=12, leading=16, textColor=colors.HexColor("#11666e"), spaceBefore=9, spaceAfter=5),
        "cell": ParagraphStyle("cell", fontName="Report", fontSize=8.5, leading=11.5),
        "code": ParagraphStyle("code", fontName="ReportMono", fontSize=6.4, leading=9, spaceAfter=9),
    }
    def text(s):
        s = escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        return re.sub(r"`([^`]+)`", r'<font name="ReportMono" size="8">\1</font>', s)
    story = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
        elif line.startswith("# "):
            story.append(Paragraph(text(line[2:]), styles["h1"]))
        elif line.startswith("## "):
            story.append(Paragraph(text(line[3:]), styles["h2"]))
        elif line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i]); i += 1
            story.append(Preformatted("\n".join(code), styles["code"]))
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [s.strip() for s in lines[i].strip("|").split("|")]
                if not all(s == "---" for s in cells):
                    rows.append([Paragraph(text(s), styles["cell"]) for s in cells])
                i += 1
            n = len(rows[0])
            widths = {2:[310,190], 3:[200,150,150], 4:[200,100,100,100], 5:[60,110,110,110,110]}[n]
            t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dcecef")),
                ("VALIGN", (0,0), (-1,-1), "TOP"), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f3f6f8")]),
                ("LINEBELOW", (0,0), (-1,0), .6, colors.HexColor("#82afb7")),
                ("LEFTPADDING", (0,0), (-1,-1), 7), ("RIGHTPADDING", (0,0), (-1,-1), 7),
                ("TOPPADDING", (0,0), (-1,-1), 4.5), ("BOTTOMPADDING", (0,0), (-1,-1), 4.5)]))
            story.extend([t, Spacer(1,8)])
            continue
        else:
            story.append(Paragraph(text(line), styles["body"]))
        i += 1
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#dce5ec")); canvas.line(48,37,564,37)
        canvas.setFont("Report",7.5); canvas.setFillColor(colors.HexColor("#637286"))
        canvas.drawString(48,24,"SHoEbox | Delayed recall | 14 November 2026")
        canvas.drawRightString(564,24,str(doc.page))
    doc = SimpleDocTemplate(str(HERE / "DELAYED_RECALL_REPAIR.pdf"), pagesize=(612,792),
        rightMargin=56, leftMargin=56, topMargin=42, bottomMargin=48,
        title="SHoEbox delayed-recall repair", author="SHoEbox development")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
