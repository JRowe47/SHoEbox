"""Render the compact, committed EXP01 evidence as Markdown and a PNG."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .adapter import ROOT
from .run import ARMS,write

OUT=ROOT/'reports/exp01'
LABELS=['Native 500K','Native, matched core','Additive, frozen core','Bilinear, frozen core','Additive, joint','Bilinear, joint']


def main():
    training=json.loads((OUT/'train.json').read_text());audit=json.loads((OUT/'audit.json').read_text())
    probes=json.loads((OUT/'probes.json').read_text());copy=json.loads((OUT/'single_copy.json').read_text())
    benchmark=json.loads((OUT/'benchmark.json').read_text())
    rows=[];summaries={};probe_rows=[]
    for core in ('minimal','full'):
        ar=[x for x in audit['rows'] if x['core']==core];tr=[x for x in training['rows'] if x['core']==core]
        summaries[core]={}
        for arm in ARMS:
            exact=[x['arms'][arm]['test']['native_rollout']['native_exact'] for x in ar]
            pair=[x['arms'][arm]['query_pair_exact'] for x in ar]
            fitted=tr[0]['arms'][arm]['inventory']['parameters']
            speed=np.mean([x['arms'][arm]['bytes_per_cpu_second'] for x in tr])
            inference=next(x['bytes_per_cpu_second'] for x in benchmark['rows'] if x['core']==core and x['arm']==arm)
            result=dict(exact=float(np.mean(exact)),seed_range=[min(exact),max(exact)],query_pair_exact=float(np.mean(pair)),
                        fitted_parameters=fitted,training_bytes_per_cpu_second=float(speed),inference_bytes_per_cpu_second=inference)
            summaries[core][arm]=result
            rows.append(f"| {core} | {LABELS[list(ARMS).index(arm)]} | {100*result['exact']:.2f}% | {100*min(exact):.2f}–{100*max(exact):.2f}% | {100*result['query_pair_exact']:.2f}% | {fitted:,} |")
        selected=[probes[f'{core}_{seed}'] for seed in (1,7,19)]
        strict=[next(x for x in p['factorized'] if x['source']=='l3' and x['at']=='answer' and x['condition']=='standardized')['test_strict_binding_accuracy'] for p in selected]
        linear=[next(x for x in p['rows'] if x['source']=='l3' and x['at']=='answer' and x['condition']=='standardized' and x['label']=='queried_value')['test_accuracy'] for p in selected]
        cp=[r['test']['native_rollout']['native_exact'] for r in copy['rows'] if r['core']==core]
        summaries[core]['diagnostics']=dict(strict_factorized=float(np.mean(strict)),strict_seed_range=[min(strict),max(strict)],
            direct_linear=float(np.mean(linear)),single_copy=float(np.mean(cp)))
        probe_rows.append(f"| {core} | {100*np.mean(cp):.2f}% | {100*np.mean(linear):.2f}% | {100*np.mean(strict):.2f}% | {100*min(strict):.2f}–{100*max(strict):.2f}% |")
    write(OUT/'summary.json',summaries)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,7.2),gridspec_kw={'width_ratios':[1.2,1]})
    colors={'minimal':'#2474a5','full':'#919ba5'}
    for i,core in enumerate(('minimal','full')):
        y=np.arange(6)+(i-.5)*.30;values=[100*summaries[core][arm]['exact'] for arm in ARMS]
        axes[0].barh(y,values,height=.28,color=colors[core],label=core.capitalize())
        for yy,v in zip(y,values):axes[0].text(v+1,yy,f'{v:.1f}',va='center',fontsize=8)
    axes[0].set_yticks(np.arange(6),LABELS);axes[0].invert_yaxis();axes[0].set_xlim(0,108)
    axes[0].axvline(50,color='#c87932',ls='--',lw=1);axes[0].axvline(95,color='#5b8e63',ls=':',lw=1)
    axes[0].set_xlabel('Whole two-answer episodes correct (%)')
    axes[0].set_title('Online heads: native binding\nremains near the 50% shortcut',loc='left',fontsize=11,fontweight='bold')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.925),ncol=2,frameon=False)
    names=['Query identity','Both record keys¹','Both values¹','Direct linear answer','Factorized answer²']
    for i,core in enumerate(('minimal','full')):
        ps=[probes[f'{core}_{s}'] for s in (1,7,19)]
        def score(labels):
            return 100*np.mean([x['test_accuracy'] for p in ps for x in p['rows'] if x['source']=='l3' and x['at']=='answer' and x['condition']=='standardized' and x['label'] in labels])
        vals=[score(['query_identity']),score(['first_key','second_key']),score(['first_value','second_value']),
              100*summaries[core]['diagnostics']['direct_linear'],100*summaries[core]['diagnostics']['strict_factorized']]
        yy=np.arange(5)+(i-.5)*.3;axes[1].barh(yy,vals,height=.28,color=colors[core])
        for y,v in zip(yy,vals):axes[1].text(v+1,y,f'{v:.1f}',va='center',fontsize=8)
    axes[1].set_yticks(np.arange(5),names);axes[1].invert_yaxis();axes[1].set_xlim(0,116)
    axes[1].set_xlabel('First-query diagnostic accuracy (%)')
    axes[1].set_title('The 64-float L3 state\nsupports factorized decoding',loc='left',fontsize=11,fontweight='bold')
    for ax in axes:ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('SHoEbox EXP01 · Stored bindings are recoverable',fontsize=15,fontweight='bold',y=.975)
    fig.text(.04,.105,'Three seeds per core · 1,024 held-out presentations / 256 worlds · Explicit query framing',fontsize=9)
    fig.text(.04,.07,'¹ Mean individual-key/value accuracy.  ² Unique match of predicted query/key identities; separately supervised offline probes.',fontsize=9,color='#4a5055')
    fig.text(.04,.035,'Left and right panels measure different endpoints; the probe is not a native online capability.',fontsize=9,color='#4a5055')
    fig.subplots_adjust(left=.19,right=.94,top=.81,bottom=.24,wspace=.68)
    fig.savefig(OUT/'SHoEbox_EXP01_Results.png',dpi=180,facecolor='white');plt.close(fig)
    costs=[]
    for arm,label in zip(ARMS,LABELS):
        m=summaries['minimal'][arm];f=summaries['full'][arm]
        costs.append(f"| {label} | {m['training_bytes_per_cpu_second']:,.0f} | {f['training_bytes_per_cpu_second']:,.0f} | {m['inference_bytes_per_cpu_second']:,.0f} | {f['inference_bytes_per_cpu_second']:,.0f} |")
    total_cpu=sum(r['warmup']['cpu_seconds']+sum(x['cpu_seconds'] for x in r['arms'].values())+r['arms']['native']['warmup_cpu_seconds'] for r in training['rows'])
    report=f'''# SHoEbox EXP01 — query/readout diagnosis

**15 November 2026 · Baseline: `317fbcd023b9d9b3c2b359ba8f99f3dadbf46783` · Minimal v19 and full C.**

The first development item now has an implemented online readout experiment and a
completed diagnostic screen. **The minimal model retains enough information to
recover the first queried binding: a factorized L3 probe reaches
{100*summaries['minimal']['diagnostics']['strict_factorized']:.2f}% held-out strict accuracy.** Its native
two-answer decoder reaches {100*summaries['minimal']['native']['exact']:.2f}% exact, near the 50% input-order
shortcut. Neither of the generic online readouts solves that selection problem.

**Decision:** keep minimal v19 as the production default. Continue EXP01 with an
online factorized query/key/value readout and direct selection credit. The
current evidence does not call for a larger state, more expert capacity, or
extra reasoning passes on this fixture. EXP02 slot expansion remains gated.

![EXP01 results](SHoEbox_EXP01_Results.png)

## What was implemented

- A native C++ rank-16 query-conditioned residual on the existing 256-byte
  hierarchical decoder: `W[tanh(Uq+bq) * tanh(Vm+bm)] + b`. Here `q` contains
  current states at all four scales and `m` contains the four retained reads.
- A capacity-matched additive head with the same projections and output weights,
  plus native full-budget and reduced-bank controls.
- Frozen-core head fitting and joint fitting through SHoEbox's existing bounded
  local credit. Both learn online, one observed target at a time.
- First-answer representation probes, raw/centered/standardized controls,
  query-order counterfactuals, head/query/read lesions, exact continuation, and
  independent timing. Literature references appear beside the implementation.

The added head has **10,511 fitted parameters**, deducted from the bank budget,
**42,044 weight bytes**, and **2,748 scratch bytes**. It adds **zero persistent
sequence-state bytes**, zero optimizer moments, no replay, and no extra model
pass. The generic head is a research adapter; it is not enabled in the public
minimal core. Code: `experiments/query_readout/`.

## Locked native audit

All values below are native 256-byte autoregressive scores. Both answer bytes
must be right for exact episode credit; supplied query/framing bytes are not
generated. Query-pair credit requires success in both original and reversed
query order, with the same observed records. Each run uses the same cases, so
three seeds do not triple the number of independent test worlds.

| Core | Arm | Exact episodes | Seed range | Both query orders | Fitted parameters |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

No arm crosses the registered 95% easiest-binding gate or a 5-point practical
gain. No arm is promoted. Frozen head/query/read lesions and restricted-alphabet
scores are retained per seed in `audit.json`; they do not change this decision.

## Localization: value retention versus usable binding

The single-value control supplies one record and one query with the original
padding. Its source-world split is preserved, but concrete one-record mappings
can recur across those derived folds. It tests copying the sole supplied value,
not generalization to unseen one-key mappings. It receives 250K bytes per seed.

The representation diagnostic stops **before the first answer**; no earlier
answer or teacher-forced answer can supply the target. The selected source is
the final 64-float L3 state. All normalizers and ridge heads are fit on training
worlds only. Every conditioning arm uses ridge regularization 1 and an
unpenalized intercept. Raw and centered fits are mathematically equivalent;
standardization changes the feature metric while preserving the fitting method.

| Core | Single-value copy, native | Direct linear answer probe | Factorized strict first-query probe | Factorized seed range |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(probe_rows)}

The factorized probe decodes the query ID, both record key IDs, and both record
values separately. It matches the **predicted query** against **predicted keys**
and selects the corresponding **predicted value**. Strict success also requires
a unique match. It never uses the true query-to-record index to choose the
primary answer. A separate oracle-index score is retained as a diagnostic ceiling.
The five heads contain **9,620 fitted parameters** plus 128 normalization floats.

This gives a concrete localization: the slow representation can expose the
query, record addresses, and their corresponding values; generic final-answer
training does not learn to combine them reliably at this operating point. It
does **not** establish native online binding: the diagnostic uses intermediate
labels, a batch ridge fit, and a task-specific matching operation. That distinction
is the purpose of the experiment. The next task is to train this separation
online and recover native output accuracy.

Results for all 1,728 final diagnostic linear fits are retained in `probes.json`: six labels,
two observation points, eight sources, three conditioning choices, and six core/
seed checkpoints. The extra queried-value head is the direct linear control;
only five heads participate in factorized decoding. Sources include current
retained reads and a read-only address-matrix requery. The latter is explicitly
a diagnostic read, not the previously executed native addressed read.

## Literature reviewed and integration decisions

| Source | Relevant demonstrated mechanism | Decision for SHoEbox |
| --- | --- | --- |
| [R1] Kim et al., *Hadamard Product for Low-rank Bilinear Pooling*, ICLR 2017 | Project two inputs to a small shared dimension, multiply elementwise, and project to output; reduces the cost of full bilinear interactions. | Implemented as the rank-16 online candidate with an additive control. It did not solve this fixture. |
| [R2] Miller et al., *Key-Value Memory Networks for Directly Reading Documents*, EMNLP 2016 | Use different representations for memory addressing and returned content. | The factorized probe supports making query/key selection and value decoding explicit in the next online readout. Do not add a symbolic binding store to the scored native path. |
| [R3] Schlag et al., *Linear Transformers Are Secretly Fast Weight Programmers*, 2021 | Delta-rule fast weights update an association using its prediction residual rather than only accumulating outer products. | The current bounded memory already uses residual writes; replacing storage is not the first intervention when its contents are recoverable. |
| [R4] Yang et al., *Gated Delta Networks*, ICLR 2025 | Combine adaptive forgetting with targeted delta updates in recurrent associative memory. | Keep as the next retention/editing control if interference tests show write failures. Do not change write policy and readout together in this diagnostic. |
| [R5] Arora et al., *Zoology*, 2023 | Multi-query associative recall exposes failures that simpler recall tasks can hide; input-dependent access matters. | Keep independently ordered queries, whole episodes, and counterfactual queries. A high answer-token score alone is insufficient. |
| [R6] Poli et al., *Mechanistic Design and Scaling of Hybrid Architectures*, ICML 2024 | Small capability tasks and explicit parameter/state/compute comparisons guide architecture development. | Preserve separate accuracy, training exposure, state, and runtime accounting; qualify an online binding win before broad scaling. |

These papers motivate mechanisms and controls. The numerical SHoEbox results in
this report come from the new local experiments, not from those papers.

## Speed and exposure

| Arm | Minimal training bytes/CPU-s | Full training bytes/CPU-s | Minimal inference bytes/CPU-s | Full inference bytes/CPU-s |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(costs)}

Training rates cover the 500K-byte continuation, resets, and explicit retirement;
they exclude evaluation, compilation, and checkpoint I/O. Frozen-core rows fit
fewer parameters and therefore are not equivalent training algorithms. The
screen ran up to four processes; do not treat small timing differences as wins.
Inference is a separate one-process, pinned-CPU measurement: seed 7, three passes
of 32,768 observed bytes, full 256-byte prediction **at every byte**, with all
per-byte work inside C++. It measures teacher-fed streaming inference, not text
quality or sampling overhead. `benchmark.json` records the fixed stream hash and
all repetitions. The default public runtime executes none of the adapter code.

- **36 completed comparison branches:** six arms × two cores × three seeds.
- **250K warmup + 500K continuation bytes** per branch, a fixed recipe: default
  fast/slow/query/head rates and auxiliary weight 0.1; new-head rate 0.16.
- Shared reduced-core warmups avoid repeating identical work: **21M actual
  charged training bytes** for the formal screen, versus 27M logical exposure
  summed over its final branches. Formal training consumed **{total_cpu:.2f} CPU seconds**.
- Additional one-seed development pilot: **1.7M bytes**. Six single-value copy
  controls: **1.5M bytes**. Total new online training: **24.2M charged bytes**.
- Train/dev/test: 2,048/512/1,024 balanced presentations, grouped by concrete
  two-key world. There are 512/128/256 underlying worlds. No test score selects
  an online arm, learning rate, checkpoint, or training horizon.
- The factorized diagnostic was chosen on pilot development data, then locked
  before its held-out audit. Its intermediate labels and offline fit are
  accounted separately from online model training.

This first fixed-recipe screen is not the planned six-policy promotion sweep or
a 1M/5M confirmation campaign. Its result is sufficiently clear to move from
generic output heads to factorized selection, rather than extend this sweep.

## Validation and compatibility

- **88 Python tests:** 48 public runtime tests and 40 MAD/EXP00/query tests,
  including 11 new query-adapter tests.
- **1,992 new native checks** pass in optimized and ASan/UBSan builds. They cover
  finite-difference derivatives of both heads, no-adapter parity, head-only core
  freeze, and continuation through unfinished stream boundaries.
- The complete existing native runtime suite also passes optimized and sanitizer
  checks. LeakSanitizer is explicitly disabled on this managed host; ASan/UBSan
  remain active. CI runs with leak checking enabled.
- Future/ignored labels do not alter observation state; first-answer extraction
  cannot consume an answer; normalizers do not use validation examples.
- Public runtime files, default v19 policy, and historical checkpoint semantics
  are unchanged. No weights, datasets, or compiled binaries are committed.

## Next implementation: EXP01-B

1. Replace the generic bilinear output with separately learned query/key
   selection and value readout over the **existing** evidence. Start with a small
   fixed-rank projection and one read; preserve byte-derived inputs and bounded
   online learning. Use [R2] for address/value separation.
2. Add direct local supervision for address selection and value reconstruction
   from already observed constituents. Compare this credit with answer-only
   training at matched capacity and exposure. Keep explicit record IDs confined
   to declared diagnostic controls.
3. Require native two-answer accuracy ≥95%, no seed below 90%, correct query
   counterfactuals, and measured one-pass cost. Then run the planned LR controls,
   offset/delay audits, and language/copy regressions before default promotion.
4. Revisit [R4] writes or EXP02 slots only if that experiment cannot preserve or
   select the needed evidence. The present probe demonstrates adequate state for
   this two-key fixture, not arbitrary delay or larger binding sets.

## Primary references

- **R1:** Kim et al. [Hadamard Product for Low-rank Bilinear Pooling](https://arxiv.org/abs/1610.04325), ICLR 2017; equations 3–5.
- **R2:** Miller et al. [Key-Value Memory Networks for Directly Reading Documents](https://aclanthology.org/D16-1147/), EMNLP 2016; section 3.
- **R3:** Schlag, Irie, and Schmidhuber. [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174), 2021.
- **R4:** Yang, Kautz, and Hatamizadeh. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464), ICLR 2025.
- **R5:** Arora et al. [Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927), 2023.
- **R6:** Poli et al. [Mechanistic Design and Scaling of Hybrid Architectures](https://proceedings.mlr.press/v235/poli24a.html), ICML 2024.

Reproduction commands and input/gradient contracts:
[`experiments/query_readout/README.md`](../../experiments/query_readout/README.md).
Compact evidence: `plan.json`, `train.json`, `audit.json`, `probes.json`,
`single_copy.json`, `pilot.json`, `benchmark.json`, `verification.json`, and `summary.json`.
'''
    (OUT/'SHoEbox_EXP01_Query_Readout.md').write_text(report)


if __name__=='__main__':main()
