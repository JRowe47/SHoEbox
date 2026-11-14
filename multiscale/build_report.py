"""Build the compact Markdown/PDF report and learning-curve figure from JSON."""
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
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, Preformatted, KeepTogether

HERE=Path(__file__).resolve().parent


def main():
    data=json.loads((HERE/"results/language.json").read_text())
    capacity=json.loads((HERE/"results/capacity_5m.json").read_text())[0]
    memory=json.loads((HERE/"results/association_language.json").read_text())[0]
    history=json.loads((HERE/"results/historical_comparison.json").read_text())
    summaries=[]
    for level in [1,2,4]:
        rows=[r for r in data if r["levels"]==level]
        summaries.append(dict(level=level,parameters=rows[0]["parameters"],nll=mean(r["test_nll"] for r in rows),
                              seconds=mean(r["training_seconds"] for r in rows),bps=mean(r["bytes_per_second"] for r in rows),
                              threshold=mean(next(c["seconds"] for c in r["curve"] if c["nll"]<=2.1) for r in rows)))
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout="constrained")
    palette={1:"#334155",2:"#059669",4:"#7c3aed"}
    for level in [1,2,4]:
        rows=[r for r in data if r["levels"]==level]
        x=np.array([c["targets"] for c in rows[0]["curve"][1:]])/1e6
        ys=np.array([[c["nll"] for c in r["curve"][1:]] for r in rows])
        times=np.array([[c["seconds"] for c in r["curve"][1:]] for r in rows])
        label={1:"Fast only",2:"Two scales",4:"Four scales"}[level]
        axes[0].plot(x,ys.mean(0),label=label,color=palette[level],lw=2)
        axes[0].fill_between(x,ys.min(0),ys.max(0),color=palette[level],alpha=.12)
        axes[1].plot(times.mean(0),ys.mean(0),label=label,color=palette[level],lw=2)
    axes[0].set(xlabel="Training targets (millions)",ylabel="Held-out nats/byte",title="Learning per byte")
    axes[1].set(xlabel="Training seconds",ylabel="Held-out nats/byte",title="Learning per second")
    for ax in axes:ax.grid(alpha=.15);ax.legend(frameon=False)
    fig.suptitle("Multiscale SHoE | three matched seeds | 1M fitted-parameter budget",fontsize=13)
    fig.savefig(HERE/"results/learning_curves.png",dpi=170);plt.close(fig)
    table="\n".join(f"| {s['level']} | {s['parameters']:,} | {s['nll']:.5f} | {s['seconds']:.2f} | {s['bps']:,.0f} | {s['threshold']:.2f} |" for s in summaries)
    seed_table="\n".join(f"| {r['seed']} | {r['levels']} | {r['test_nll']:.5f} | {r['bytes_per_second']:,.0f} |" for r in data)
    deltas=[]
    for level in [1,2,3]:
        rows=[r for r in data if r["levels"]==4]
        d=mean(r["interventions"][str(level)]["mask"]-r["intervention_baseline_nll"] for r in rows)
        q=mean(r["interventions"][str(level)]["shuffle"]-r["intervention_baseline_nll"] for r in rows)
        deltas.append(f"| Level {level} | +{d:.5f} | +{q:.5f} |")
    historical="\n".join(f"| {r['model']} | {r['parameters']:,} | {r['full_val_nll']:.5f} | {r['bytes_per_second']:,.0f} |" for r in history)
    text=f"""# SHoE multiscale rebuild: implementation and results

14 November 2026 | Base repository: JRowe47/SHoEbox, 5dc8166

The independent fixed-rate rebuild is implemented. Four scales average **1.97753 nats/byte at 64,746 training bytes/s** across three million-target Tiny Shakespeare runs. The deeper-dependency acceptance gate is **not passed**: the ordered distant-cue task remains at 50% accuracy. Learned boundaries and promotion as the selected generalist profile remain gated.

## What was delivered

The new `shoe_multiscale` entry point exposes a native C++17 runtime with Python/NumPy bindings. It supports one, two and four scales, width 64, top-one experts, ordered child composition, separate canonical content and recurrent context, compact 8x8 MinKron-style state, one shared binary byte decoder, cached slow gain/bias conditioning, native contrastive forecasting and eight-event local reverse sweeps.

The runtime has no cortical regional scheduler, publication DAG, replay machinery, historical parameter versions or cross-region credit traversal. Child values and entering local states are detached boundaries. Only selected expert weights receive updates. The existing `shoebox_reference` implementation is retained unchanged.

Versioned, checksummed checkpoints preserve pending blocks and partial groups exactly. Frozen inference, generation, explicit payload binding/retrieval, bounded indexed routing, CLI commands and reproducible benchmark/test runners are included. Models, datasets and compiled binaries are excluded from the published source package.

## Language results

All new models train on exactly the first 1,000,000 next-byte targets, once, in corpus order. Seeds are 7, 17 and 29. The final contiguous 10% is held out: 111,539 targets. Parameter budgets include all fitted routers, projections, prediction heads, embeddings and decoder parameters. Reported rows are three-seed means.

| Scales | Parameters | Test nats/byte | Train seconds | Train bytes/s | Seconds to 2.10 |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

The four-scale model improves mean held-out loss by {summaries[0]['nll']-summaries[2]['nll']:.5f} nats/byte over its retrained matched-budget fast-only control. It takes {summaries[2]['seconds']/summaries[0]['seconds']:.2f} times as long. At the measured 2.10 threshold, the fast-only control arrives sooner. These results support better learning per byte, not better learning per second from adding scales.

The 2.10 column uses the first checkpoint at or below 2.10 on the fixed 16,384-target curve subset, sampled every 100,000 targets. It is not an interpolated or full-test threshold. Full-test losses above use all held-out targets after terminal local updates. Curve points use the fitted weights present at each observation before a terminal drain.

![Learning curves](results/learning_curves.png)

The figure shows means after 100,000 targets; shading is the seed range. Untrained losses near 5.55 are omitted from its plotting range. Both axes use the same data and models.

| Seed | Scales | Full-test nats/byte | Training bytes/s |
| --- | ---: | ---: | ---: |
{seed_table}

## Historical comparison

The prior 14 November comparison used the same corpus, split, target order and AMD EPYC 9V74 host, seed 7. Those runs are reused as historical measurements; they were not rerun or extended to three seeds here. Their two-CPU placement differs from this rebuild's single native worker pinned to CPU 0.

| Historical model | Parameters | Test nats/byte | Training bytes/s |
| --- | ---: | ---: | ---: |
{historical}

The new four-scale seed-7 result is 1.97440 nats/byte, versus 1.97691 for the prior SHoEbox run. Mean new training throughput is approximately {summaries[2]['bps']/history[0]['bytes_per_second']:.1f} times that prior throughput. Generation and teacher-forced inference are measured separately in the raw JSON; neither is used as training throughput.

## Does the hierarchy contribute?

On the frozen four-scale models, masking a slow level worsens the 16,384-target held-out loss. A fixed coordinate permutation of that level's cached conditioning vectors also worsens loss. Other levels, events and learned weights are left intact. The baseline is recomputed after terminal updates for each model.

| Intervention | Mean mask NLL increase | Mean conditioning-shuffle increase |
| --- | ---: | ---: |
{chr(10).join(deltas)}

These tests demonstrate use of the slow conditioning. The retrained one/two/four-scale controls establish the small improvement per byte. Neither test establishes a learned long-distance dependency.

The separate ordered-cue task presents AB or BA, then an identical run of distractors, a query, and a cue-dependent binary answer. Both cues contain the same inventory; order carries the answer. Training mixes 16/48/96-byte gaps and shifts 0-3. Evaluation includes those gaps, an unseen 192-byte gap, and shifts 0-7. Each model trains on 6,000 examples per seed, about 360,000 bytes. All nine model/seed combinations score **50%** at all four evaluated gaps. Masking slow levels does not reveal useful cue retrieval. The result is recorded as a failed gate, not converted into a passing architecture test by checking gradients or changed logits.

## Association store and capacity

The store uses 256 buckets with four ways, up to sixteen semantic candidates and at most two returned records. Query precedes automatic write; a value is admitted only after its whole four-byte group is observed. Exact caller-provided keys and payloads have independent checked handles. Tests verify novel binding, delayed exact binary retrieval, checkpoint preservation, collision eviction, stale-handle rejection and invalidation across memory resets. This is explicit payload recovery, not learned free-form string reproduction by the byte decoder.

A one-seed language run with automatic association recall enabled reaches {memory['test_nll']:.5f} nats/byte at {memory['bytes_per_second']:,.0f} bytes/s, compared with 1.97440 for the same-seed store-disabled model. Frozen cold evaluation uses no stored training episodes. This does not qualify enabling neural recall by default. The independent exact binding API remains available.

The 5M capacity check fits {capacity['parameters']:,} parameters across banks {capacity['experts']}. It executes the same 1.328125 selected experts per byte and completes one million targets at {capacity['bytes_per_second']:,.0f} bytes/s, with test loss {capacity['test_nll']:.5f}. Router tests bound comparisons to at most 48 per event in this supported range. Derived centroid paths update locally. The one-seed capacity check establishes execution and bounded routing, not a demonstrated quality gain from 5M capacity or qualification for 100M.

## Learning policy and verification

Each level freezes its parameters for eight native events. Output derivatives accumulate across each state's lifetime using the actual readout weights that made each prediction. One reverse sweep differentiates local expert, adapter, projection, recurrent-vector and MinKron matrix computations. Shared byte readout learning remains online. Task gradients average over served byte predictions; native gradients average over valid next-unit targets and are weighted by 0.1. The final cross-block forecast is truncated, avoiding retained old parameter versions. Targets and negatives are immutable detached emitted content vectors.

All language runs use local peak rate 0.08, readout peak rate 0.16, route attraction 0.002, 1% warmup and cosine decay to 10% of peak. Compiler flags are C++17, -O3, -march=native and -ffp-contract=off; floats are float32. Python is 3.12.14 and NumPy 2.3.5. Training timings include terminal local updates and exclude build/import, initialization, evaluation and checkpoint IO. No GPU is used.

The verification suite passes **387 native checks**, including independent double-precision finite differences for the MinKron read/write derivative, byte-block derivatives, slow composition/projection derivatives, scheduling, content/context separation, ordered composition, selected-bank updates, routing bounds and checkpoint continuation. **Eight Python tests** cover streaming phase boundaries, source-preserving evaluation, causality, interventions, exact payloads, reset invalidation, learning, generation and invalid inputs. **Eight existing cortical inference tests** also pass.

The native suite passes AddressSanitizer and UndefinedBehaviorSanitizer. LeakSanitizer cannot run under this host's ptrace-based execution, so leak scanning was explicitly disabled while ASan/UBSan remained active. No sanitizer finding is suppressed. Compiler warning checks and whitespace checks pass. Reopening saved seed-7 models under the final runtime reproduces all three full-test losses exactly.

## Build-plan status

| Plan item | Status |
| --- | --- |
| Fast SHoE core | Implemented and measured |
| Ordered composition with native and direct credit | Implemented; long-dependency gate failed |
| Four fixed-rate scales near 1M | Implemented and compared with retrained controls |
| Single bounded association store | Implemented; exact API verified; neural path remains off by default |
| Learned boundaries | Not implemented: prerequisite architecture gate did not pass |
| Bank capacity scaling | 5M runtime and routing exercised; larger-bank and batching qualification pending |

Held-out mixed prose/code selection, systematic familiar-part recombination, multiple-seed 5M comparisons and matched new runs of all historical baselines remain beyond this failed acceptance gate. The practical next development target is the local native objective and its retention of information needed after repeated irrelevant units. Adding learned boundaries before that works would bypass the supplied plan's stopping condition.

## Reproduction

Run from the repository root with a local copy of Tiny Shakespeare:

```bash
python multiscale/tests/run_all.py --no-leak-check
python -m multiscale.benchmark --data input.txt --out /tmp/shoe-1m
python -m multiscale.dependencies --out /tmp/shoe-dependencies
python -m multiscale.benchmark --data input.txt --out /tmp/shoe-5m --parameters 5000000 --levels 4 --seeds 7
```

On an ordinary host, omit `--no-leak-check` to run leak scanning as well. Dataset SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`. Full source, tests, individual measurements and samples are under `multiscale/`; `shoe_multiscale.py` is the public import. Rebuild this report from its committed JSON with `python -m multiscale.build_report` (Matplotlib and ReportLab required).
"""
    (HERE/"REBUILD_REPORT.md").write_text(text)
    # Embed actual fonts rather than relying on viewer substitutions for the
    # PDF base-14 names; substitutions can have different character advances.
    font_root=Path("/usr/share/fonts/truetype/dejavu")
    if not font_root.exists():
        import matplotlib.font_manager as fm
        font_root=Path(fm.findfont("DejaVu Sans")).parent
    pdfmetrics.registerFont(TTFont("RebuildSans",str(font_root/"DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("RebuildBold",str(font_root/"DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("RebuildMono",str(font_root/"DejaVuSansMono.ttf")))
    pdfmetrics.registerFontFamily("RebuildSans",normal="RebuildSans",bold="RebuildBold",italic="RebuildSans",boldItalic="RebuildBold")
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyCustom",fontName="RebuildSans",fontSize=9.1,leading=13,spaceAfter=8,textColor=colors.HexColor("#27364a")))
    styles.add(ParagraphStyle(name="H1Custom",fontName="RebuildBold",fontSize=21,leading=25,spaceAfter=14,textColor=colors.HexColor("#152840")))
    styles.add(ParagraphStyle(name="H2Custom",fontName="RebuildBold",fontSize=13,leading=17,spaceBefore=13,spaceAfter=7,textColor=colors.HexColor("#176d72")))
    styles.add(ParagraphStyle(name="TableCustom",fontName="RebuildSans",fontSize=7.6,leading=10,spaceAfter=0))
    story=[]
    def inline(s):
        s=escape(s)
        s=re.sub(r"\*\*(.*?)\*\*",r"<b>\1</b>",s)
        s=re.sub(r"`([^`]+)`",r'<font name="RebuildMono">\1</font>',s)
        return s
    lines=text.splitlines();i=0
    while i<len(lines):
        line=lines[i];i+=1
        if not line.strip():continue
        if line.startswith("# "):story.append(Paragraph(inline(line[2:]),styles["H1Custom"]));continue
        if line.startswith("## "):story.append(Paragraph(inline(line[3:]),styles["H2Custom"]));continue
        if line.startswith("!["):
            story.append(Image(str(HERE/"results/learning_curves.png"),width=515,height=192));continue
        if line.startswith("```"):
            code=[]
            while i<len(lines) and not lines[i].startswith("```"):code.append(lines[i]);i+=1
            i+=1
            story.append(Preformatted("\n".join(code),ParagraphStyle(name="Code",fontName="RebuildMono",fontSize=6.2,leading=9,backColor=colors.HexColor("#f1f5f9"),borderPadding=7,spaceAfter=10)));continue
        if line.startswith("|"):
            table_lines=[line]
            while i<len(lines) and lines[i].startswith("|"):table_lines.append(lines[i]);i+=1
            cells=[[c.strip() for c in row.strip("|").split("|")] for row in table_lines if not re.match(r"^\|[ :|\-]+\|$",row)]
            n=len(cells[0]);widths=([170,345] if n==2 else ([160,177,178] if n==3 else [515/n]*n))
            table=Table([[Paragraph(inline(c),styles["TableCustom"]) for c in row] for row in cells],colWidths=widths,repeatRows=1,hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e5eff3")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f7f9fb")]),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),("LINEBELOW",(0,0),(-1,0),.6,colors.HexColor("#bdd1da"))]))
            story.extend([table,Spacer(1,10)]);continue
        paragraph=[line]
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(("#","|","```","![")):
            paragraph.append(lines[i]);i+=1
        story.append(Paragraph(inline(" ".join(paragraph)),styles["BodyCustom"]))
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor("#d9e2e8"));canvas.line(40,37,555,37)
        canvas.setFont("RebuildSans",8);canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(40,25,"SHoE multiscale rebuild | 14 November 2026")
        canvas.drawRightString(555,25,str(doc.page))
    pdf=HERE/"REBUILD_REPORT.pdf"
    SimpleDocTemplate(str(pdf),pagesize=(595,842),rightMargin=40,leftMargin=40,topMargin=40,bottomMargin=50,
                      title="SHoE multiscale rebuild: implementation and results",author="Joshua Rowe").build(story,onFirstPage=footer,onLaterPages=footer)
    sources=["native.hpp","native.cpp","model.py","benchmark.py","dependencies.py","cli.py"]
    manifest={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in sources}
    (HERE/"results/source_hashes.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"report":str(pdf),"bytes":pdf.stat().st_size,"summary":summaries},indent=2))


if __name__=="__main__":main()
