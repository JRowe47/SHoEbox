"""Rebuild the addressed-memory report from committed measurement JSON."""
from pathlib import Path
from statistics import mean
import hashlib
import json
import re
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer, Preformatted

HERE = Path(__file__).resolve().parent


def main():
    evidence = json.loads((HERE / "results/addressed_memory.json").read_text())
    # Older evidence used the then-current alias "addressed" for version 2.
    for row in evidence["language"]:
        if row["memory_profile"] == "addressed_v2": row["memory_profile"] = "addressed"
    groups = {p:[r for r in evidence["language"] if r["memory_profile"] == p] for p in ("legacy", "addressed")}
    summary = {p:{k:mean(r[k] for r in rows) for k in ("test_nll", "training_seconds", "bytes_per_second", "teacher_forced_inference_bps")}
               for p,rows in groups.items()}
    old, new = summary["legacy"], summary["addressed"]
    comparison = "\n".join(f"| {p.title()} | {groups[p][0]['parameters']:,} | {s['test_nll']:.5f} | {s['training_seconds']:.2f} | {s['bytes_per_second']:,.0f} |"
                           for p,s in summary.items())
    per_seed = "\n".join(f"| {seed} | " + " | ".join(f"{next(r for r in groups[p] if r['seed']==seed)['test_nll']:.5f}" for p in groups) + " |"
                         for seed in (7,17,29))
    deltas = {mode:mean(r["memory_interventions"][mode]-r["intervention_baseline_nll"] for r in groups["addressed"])
              for mode in ("no_read", "no_time", "no_content")}
    interventions = "\n".join(f"| {mode} | {delta:+.6f} |" for mode,delta in deltas.items())
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout="constrained")
    for profile,color in (("legacy","#475569"),("addressed","#0e7490")):
        rows=groups[profile]
        y=np.array([[c["nll"] for c in r["curve"][1:]] for r in rows])
        counts=np.array([c["targets"] for c in rows[0]["curve"][1:]])/1e6
        times=np.array([[c["seconds"] for c in r["curve"][1:]] for r in rows]).mean(0)
        for ax,x in zip(axes,(counts,times)):
            ax.plot(x,y.mean(0),color=color,label=profile.title(),lw=2)
            ax.fill_between(x,y.min(0),y.max(0),color=color,alpha=.1)
    axes[0].set(xlabel="Training targets (millions)",ylabel="Held-out nats/byte",title="Learning per byte")
    axes[1].set(xlabel="Training seconds",ylabel="Held-out nats/byte",title="Learning per second")
    for ax in axes:ax.grid(alpha=.15);ax.legend(frameon=False)
    chart=HERE / "results/addressed_memory_curves.png"
    fig.savefig(chart,dpi=160);plt.close(fig)
    text=f"""# SHoEbox: addressed multiscale memory

Implementation and matched comparison | 14 November 2026 | Base: d319c3e

The rebuilt multiscale runtime now combines fixed content/time write addresses, learned read queries, native rotary clocks and local constituent reconstruction. New models enable the complete path by default; version-1 checkpoints continue through an unchanged legacy backend.

Across three matched million-target runs, mean language loss is essentially unchanged: **{new['test_nll']:.5f} versus {old['test_nll']:.5f} nats/byte**. Training retains **{100*new['bytes_per_second']/old['bytes_per_second']:.1f}%** of the original throughput. The new memory contributes to frozen language predictions, while the ordered delayed-cue task remains at **50% accuracy**.

## Implemented behavior

| Component | Behavior |
| --- | --- |
| Local feature learning | Reconstruct one rotating child role from the same canonical encoding used by composition; mean squared error weight 0.05. |
| Native prediction | Retain the existing causal next-unit contrastive objective, weight 0.1. Both objectives train the selected local expert. |
| Stable write addresses | Fixed raw-byte sketches and ordered child composition; separate current, previous and temporal address channels. |
| Learned reads | Queries depend on current learned content, previous contextual state and fixed clock phase; values combine learned and fixed features. |
| Native clocks | Eight sine/cosine pairs per slow scale; periods 4-256 native events; fixed rotations and phase renormalization. |
| Long-lived summaries | A 32-by-16 matrix at each 4/16/64-byte scale, with fixed row half-lives of 32-2,048 native events. |
| Local cost | Eight-event derivative blocks remain; no additional expert execution or cross-level backward traversal. |

Canonical content remains independent of time and recurrent context. Time enters the separate memory address. Every read precedes its write, and stored write coordinates do not move as the encoder learns. The clocks are Fourier/rotary features for associative memory, rather than transformer RoPE applied to content.

The three active address matrices total 6 KiB, excluding normalizers, sketches and derivative rings. New query/read/reconstruction parameters add 9,728 scalars per slow level. The budget allocator compensates with fewer experts: [104, 52, 31, 21], for 997,183 fitted scalars. The original four-scale profile has 997,567. The 8-by-8 MinKron recurrence remains at every level; the new memory runs only at slow events. Inference omits the reconstruction decoder.

## Matched language results

Fresh four-scale models train once on the first 1,000,000 Tiny Shakespeare next-byte targets. The final contiguous 10% supplies 111,539 held-out targets. Seeds are 7, 17 and 29; execution order alternates legacy/addressed between seeds. All results below are means from new runs on the same pinned CPU.

| Profile | Fitted scalars | Test nats/byte | Training seconds | Training bytes/s |
| --- | ---: | ---: | ---: | ---: |
{comparison}

Mean loss improves by only {old['test_nll']-new['test_nll']:.6f} nats/byte, with mixed per-seed effects. Training takes {100*(new['training_seconds']/old['training_seconds']-1):.1f}% longer; throughput falls {100*(1-new['bytes_per_second']/old['bytes_per_second']):.1f}%. Teacher-forced inference averages {new['teacher_forced_inference_bps']:,.0f} versus {old['teacher_forced_inference_bps']:,.0f} bytes/s. These runs do not establish an overall language-quality gain.

| Seed | Legacy test NLL | Addressed test NLL |
| --- | ---: | ---: |
{per_seed}

![Matched learning curves](results/addressed_memory_curves.png)

Curves use a fixed 16,384-target held-out subset every 100,000 training targets; shading is the seed range. Full-test figures in the tables use the entire held-out split after terminal local updates. Initialization, compilation, evaluation and checkpoint IO are excluded from training timing; terminal local updates are included.

## Contribution and delayed recall

Frozen interventions remove address channels throughout a cold evaluation prefix. Positive NLL changes mean that removal worsens prediction on the fixed 16,384-target subset.

| Removed component | Mean NLL change |
| --- | ---: |
{interventions}

The content-addressed memory supplies most of the measured contribution. Temporal addressing has little effect in this short language run. These interventions measure reliance within a fitted model; the retrained comparison above measures the total architectural change.

The distant-cue test supplies AB or BA, identical distractors, then a binary query. Each model trains on 6,000 examples with gaps 16/48/96 and grouping shifts 0-3. Evaluation uses gaps 16/48/96/192 and shifts 0-7. Both profiles score 50% at every gap for all three seeds. The implementation therefore supplies the addressing and local learning mechanisms, but has not resolved the original delayed-recall failure.

## Verification and compatibility

The suite passes **1,347 native checks and 12 Python tests**, plus AddressSanitizer and UndefinedBehaviorSanitizer. Checks cover independent numerical derivatives for queries, values and carried matrices; reconstruction derivatives for all four roles; million-step clock phase stability; read-before-write causality; stable write coordinates under changed encoders; exact local block gradients; selected-bank updates; the minimum parameter budget; and checkpoint continuation at partial-block and group boundaries. LeakSanitizer was explicitly disabled on this ptrace-managed host.

Version-1 native sources are byte-identical to the base commit. Loading an older checkpoint automatically selects that backend and preserves continuation exactly. Fast-only models retain the original prediction arithmetic. Version 2 saves phases, clocks, stable sketches, normalization state, learned queries, reconstruction parameters and pending local records. Reopening all six benchmark checkpoints under the final source reproduces their full-test losses exactly. Minimum-budget allocation now reserves two experts for every level.

Effective capabilities are available through `stats()`. Frozen `no_read`, `no_time` and `no_content` controls are exposed through `evaluate(..., memory=...)`. Reconstruction can be independently ablated with `configure(reconstruction_weight=0)`. The existing cortical runtime and explicit episode-store policy are unchanged.

## Reproduction

```bash
python multiscale/tests/run_all.py --no-leak-check
python -m multiscale.compare_memory --data input.txt --out /tmp/shoe-memory
python -m multiscale.build_address_report
```

Use `Model(memory_profile="legacy")` for a fresh original-runtime control; new `Model()` instances use the addressed profile. Training and resume are also supported through `python -m multiscale.cli`. Resume chooses the saved policy automatically.

Hardware: {evidence['cpu_model']}, one worker pinned to CPU {evidence['cpu']}. Python {evidence['python']}; NumPy {evidence['numpy']}; {evidence['compiler']}. Native build: C++17, -O3, -march=native, -ffp-contract=off. Local peak learning rate 0.08, shared head 0.16, route attraction 0.002; 1% warmup and cosine decay to 10% for language runs. No tuning was performed against these comparison results.

Dataset SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`. Compact measurements, curves and source hashes accompany this report. Datasets, trained checkpoints and compiled binaries are excluded from the source package.

The fixed-address/learned-read separation adapts the design principle discussed in [Streaming Recurrent Memory](https://arxiv.org/html/2610.34833v1). This implementation uses its own compact state, fixed retention schedule and existing SHoE local learning; it does not import the full SRM architecture or its training machinery.
"""
    (HERE / "ADDRESSED_MEMORY_REPORT.md").write_text(text)
    render_pdf(text, HERE / "ADDRESSED_MEMORY_REPORT.pdf", chart)
    files=["address_memory.hpp","native.cpp","native.hpp","model.py","cli.py","benchmark.py","dependencies.py",
           "compare_memory.py","tests/native_tests.cpp","tests/test_model.py","legacy/native.cpp","legacy/native.hpp"]
    manifest={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in files}
    (HERE / "results/addressed_source_hashes.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


def render_pdf(text, path, chart):
    import matplotlib.font_manager as fm
    root=Path(fm.findfont("DejaVu Sans")).parent
    for name,file in (("AddressSans","DejaVuSans.ttf"),("AddressBold","DejaVuSans-Bold.ttf"),("AddressMono","DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(name,str(root/file)))
    pdfmetrics.registerFontFamily("AddressSans",normal="AddressSans",bold="AddressBold",italic="AddressSans",boldItalic="AddressBold")
    body=ParagraphStyle("Body",fontName="AddressSans",fontSize=9,leading=13,spaceAfter=8,textColor=colors.HexColor("#24364a"))
    h1=ParagraphStyle("Title",parent=body,fontName="AddressBold",fontSize=21,leading=25,spaceAfter=14)
    h2=ParagraphStyle("Heading",parent=body,fontName="AddressBold",fontSize=12,leading=16,spaceBefore=12,spaceAfter=7,keepWithNext=True,textColor=colors.HexColor("#0e7490"))
    cell=ParagraphStyle("Cell",parent=body,fontSize=8,leading=11,spaceAfter=0)
    code=ParagraphStyle("Code",fontName="AddressMono",fontSize=6.8,leading=10,spaceAfter=10,backColor=colors.HexColor("#f1f5f9"),borderPadding=6)
    def inline(s):
        s=escape(s)
        s=re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",r'<link href="\2" color="#0e7490">\1</link>',s)
        s=re.sub(r"\*\*(.*?)\*\*",r"<b>\1</b>",s)
        return re.sub(r"`([^`]+)`",r'<font name="AddressMono">\1</font>',s)
    lines=text.splitlines();i=0;story=[]
    while i<len(lines):
        line=lines[i];i+=1
        if not line.strip():continue
        if line.startswith("# "):story.append(Paragraph(inline(line[2:]),h1));continue
        if line.startswith("## "):story.append(Paragraph(inline(line[3:]),h2));continue
        if line.startswith("!["):story.append(Image(str(chart),width=515,height=180.25));continue
        if line.startswith("```"):
            rows=[]
            while i<len(lines) and not lines[i].startswith("```"):rows.append(lines[i]);i+=1
            i+=1;story.append(Preformatted("\n".join(rows),code));continue
        if line.startswith("|"):
            rows=[line]
            while i<len(lines) and lines[i].startswith("|"):rows.append(lines[i]);i+=1
            rows=[[c.strip() for c in row.strip("|").split("|")] for row in rows if not re.match(r"^\|[ :|\-]+\|$",row)]
            n=len(rows[0]);widths=[140,375] if n==2 else [515/n]*n
            table=Table([[Paragraph(inline(c),cell) for c in row] for row in rows],colWidths=widths,repeatRows=1,hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e0eff2")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f6f8fa")]),("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
            story.extend([table,Spacer(1,8)]);continue
        paragraph=[line]
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(("#","|","```","![")):
            paragraph.append(lines[i]);i+=1
        story.append(Paragraph(inline(" ".join(paragraph)),body))
    def footer(canvas,doc):
        canvas.setFont("AddressSans",8);canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(40,25,"SHoEbox | Addressed multiscale memory | 14 November 2026")
        canvas.drawRightString(555,25,str(doc.page))
    SimpleDocTemplate(str(path),pagesize=(595,842),leftMargin=40,rightMargin=40,topMargin=40,bottomMargin=50,
                      title="SHoEbox: addressed multiscale memory",author="Joshua Rowe").build(story,onFirstPage=footer,onLaterPages=footer)


if __name__=="__main__":main()
