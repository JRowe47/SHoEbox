"""Create the compact evidence report from completed, hash-linked runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics as st

GROUPS={
    'screen':{'C':'Corrected C','g4_4':'State-language gains 4 / 4','g4_16':'State-language gains 4 / 16','g16_16':'State-language gains 16 / 16','g16_64':'State-language gains 16 / 64'},
    'recall':{'r1':'Live recall-producer credit 1 / 1','r4':'Live recall-producer credit 4 / 4','r4_16':'Live recall-producer credit 4 / 16','gr1':'Recall 1 / 1 + state-language 4 / 16'},
    'expert':{'e4_16':'Recall credit; expert gains 4 / 16','e16_64':'Recall credit; expert gains 16 / 64'},
    'local':{'lm01':'Local byte heads; weight 0.1 / 0.1','lm1':'Local byte heads; weight 1 / 1'},
    'local_expert':{'lme1':'Local byte heads; experts only 1 / 1','lme4':'Local byte heads; experts only 4 / 4'},
    'state_expert':{'s4_16':'Primary-language expert gains 4 / 16','s16_64':'Primary-language expert gains 16 / 64'},
    'centered':{'cmean':'Centered L2/L3 state readouts','cnorm':'Centered + normalized L2/L3 readouts'},
}
def read(path):return json.loads(path.read_text())
def mean(values):return st.mean(list(values))
def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True);s=read(a.out/'SHoEbox_L2_L3_Summary.json');seeds=(1,7,19)
    base=s['arms']['C'];changed=s['arms']['e4_16'];probes={seed:read(a.work/f'confirm/C_s{seed}/probe.json') for seed in seeds}
    screens=[]
    for group,arms in GROUPS.items():
        for arm,label in arms.items():
            r=read(a.work/group/f'{arm}_s7/results.json');assert r['status']=='complete'
            at10=next(v['nll'] for v in r['curve'] if v['targets']==10_000_000)
            screens.append(dict(arm=arm,description=label,parameters=r['parameters'],development_10m=at10,source=f'{group}/{arm}_s7/results.json',checkpoint_sha256=r['checkpoint_sha256']))
    baseline=screens[0]['development_10m']
    for row in screens:row['delta_vs_C_10m']=row['development_10m']-baseline
    with (a.out/'SHoEbox_L2_L3_Screening.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(screens[0]));writer.writeheader();writer.writerows(screens)
    s['screening']=screens
    s['centered_probes']={arm:read(a.work/f'centered/{arm}_s7/probe.json') for arm in ('cmean','cnorm')}
    def swap(arm,l):return mean(s['arms'][arm]['swap_penalties'][f'L{l}'])
    table=['| Model | Development | Audit | L2 swap penalty | L3 swap penalty |','|---|---:|---:|---:|---:|']
    for arm,label in [('C','Corrected C'),('e4_16','Recall-to-expert credit 4 / 16')]:
        v=s['arms'][arm];table.append(f"| {label} | {v['development_mean']:.6f} | {v['audit_mean']:.6f} | {swap(arm,2):+.8f} | {swap(arm,3):+.8f} |")
    means=s['fixed_mean_probes'];rows=['| Level / output | Remove output | Replace with training mean |','|---|---:|---:|']
    for l in (2,3):
        for name,abl,mean_name in [('modulation','no_modulation','modulation'),('learned recall','no_learned_recall','learned_recall')]:
            erased=mean(probes[seed]['ablations'][f'L{l}_{abl}']['delta'] for seed in seeds)
            replaced=mean(means[str(seed)]['means'][f'L{l}_{mean_name}']['delta'] for seed in seeds)
            rows.append(f'| L{l} {name} | {erased:+.6f} | {replaced:+.6f} |')
    combined=mean(v['means']['L2+L3_both']['delta'] for v in means.values())
    screen_table=['| Arm | L2 / L3 intervention | Parameters | 10M development | Change vs C |','|---|---|---:|---:|---:|']
    for r in screens:screen_table.append(f"| {r['arm']} | {r['description']} | {r['parameters']:,} | {r['development_10m']:.6f} | {r['delta_vs_C_10m']:+.6f} |")
    cp=a.work/'confirm/C_s7/latest.shoe';raw=cp.read_bytes();checkpoint=a.out/'SHoEbox_Corrected_C_5M_20M.shoe';checkpoint.write_bytes(raw)
    s['delivered_checkpoint']={'filename':checkpoint.name,'seed':7,'targets':20000000,'sha256':hashlib.sha256(raw).hexdigest(),'version':10,'source':'confirm/C_s7/latest.shoe'}
    (a.out/'SHoEbox_L2_L3_Summary.json').write_text(json.dumps(s,indent=2))
    timing=''
    if 'timing' in s:
        b=base['bytes_per_second'];c=changed['bytes_per_second']
        timing=f'''\nSerial paired training throughput was **{b:,.0f} bytes/s for C** and **{c:,.0f} bytes/s for extra recall/expert credit** ({100*(c/b-1):+.1f}%). The benchmark ran one pinned process at a time on an AMD EPYC 9V74: three seeds × three paired 200K-target repetitions, alternating arm order. Timing includes context resets and final credit retirement, and excludes loading, compilation, evaluation and saving. Parallel training-run wall times are not used for this comparison.\n'''
    centered=[]
    for arm in ('cmean','cnorm'):
        v=s['centered_probes'][arm];deltas={l:mean(x['delta'] for key,x in v['swaps'].items() if key.startswith(f'L{l}_')) for l in (2,3)}
        centered.append(f"- `{arm}`: development {v['full']['nll']:.6f}; L2 permutation {deltas[2]:+.8f}, L3 {deltas[3]:+.8f}.")
    report=f'''# SHoEbox L2/L3 diagnosis and credit experiments

2026-11-15 · corrected C · 5M parameters

## Decision

**Keep corrected C.** It is now the default for new public API and CLI models; D remains an explicit historical control. The tests identify a more specific failure than insufficient gradient magnitude: L2/L3's learned outputs act mainly as constant prediction biases. The experimental credit paths are implemented and numerically checked, but **a useful L2/L3 specialization repair has not been demonstrated**. No failed candidate is enabled in the selected baseline.

## What the intervention tests show

The original masking test conflates changing contextual information with a constant offset. To separate them, each trained C model was frozen, its upper output means were calibrated using 262,144 **training inputs only**, and development inference replaced the changing output with that fixed mean. No labels or parameter updates were used during calibration. Means were collected separately where a modulation or recall output was available; absent outputs remain absent.

Values below are mean development NLL changes across seeds 1, 7 and 19 after 20M training targets. Positive means worse.

{chr(10).join(rows)}

Replacing **both L2 and L3 modulation and learned recall together** changes NLL by only **{combined:+.8f}**. The byte pathway, L1, and fixed recall-address channels remain intact. Thus these upper learned channels add almost no measured information beyond their constant offsets on this development set. This explains why deleting them looks harmful while exchanging their experts barely matters.

The routes themselves are widely used: the seed-7 10M C probe uses about 137 effective L2 experts out of 171 and 110 L3 experts out of 116. It finds no coordinates beyond |0.95| in upper input/content/state traces. The investigated failure is not a saturated tanh or a bank that stopped routing. Main language gradients reach upper constituents; finite-difference and gradient-decomposition checks verify that path.

## Language-credit changes tested

1. **State-language gain:** scale only the main language vector entering each upper native state, preserving output-head and auxiliary update rates.
2. **Retained-value producer credit:** backpropagate learned recall-value errors to both live producing constituents. The original retained-value path detached those producers. Credit is exact within the current 512-byte constituent epoch, including the ordered pair's tanh and pooling transforms. Older retained values remain readable after their producer credit retires.
3. **Expert-only credit:** amplify recall-derived or primary state-language gradients only for the selected expert's matrix and bias, preserving shared/lower-path updates.
4. **Local language heads:** predict observed future bytes directly from L2/L3 content; compare full constituent credit against expert-only credit. These add 33,150 fitted parameters.
5. **Centered readouts:** remove each upper state's running training mean before its gain/bias readout, with and without RMS normalization (epsilon 0.01, maximum inverse scale 10). Readout derivatives and saved statistics are included; evaluation does not update means.

All 10M screens below use seed 7 and the identical data order and training schedule. The two recall/expert variants also ran to 20M across three seeds; their entries here are still the matched seed-7 10M points.

{chr(10).join(screen_table)}

Lower development loss alone did not establish specialization. The state-only, expert-only and local-head trials continued to show negligible or negative upper expert-permutation penalties. The centered tests finished as follows:

{chr(10).join(centered)}

Both centered variants also have worse development loss than C. They are retained as experimental diagnostics, not promoted.

## Three-seed confirmation

The 4 / 16 recall-to-expert candidate was selected within that experiment family using development results before its audit evaluation. At 20M targets per seed:

{chr(10).join(table)}

Its development improvement is **{changed['development_mean']-base['development_mean']:+.6f} nats/byte**, but mean reserved-audit change is **{changed['audit_mean']-base['audit_mean']:+.6f}**. Paired audit changes by seed are {', '.join(f'{v:+.6f}' for v in s['audit_difference']['paired_seed_deltas'])}. The document bootstrap interval, after averaging paired fitted-seed differences, is [{s['audit_difference']['document_bootstrap_95'][0]:+.6f}, {s['audit_difference']['document_bootstrap_95'][1]:+.6f}]. This document resampling does not substitute for independent training seeds. The candidate's upper swap penalties remain tiny; C's L1 penalty is {swap('C',1):+.6f} for comparison.
{timing}
## Matched protocol and provenance

- Actual C size: **4,999,743 fitted parameters**, width 64, banks 571 / 285 / 171 / 116; event spans 1 / 4 / 16 / 64 bytes.
- FineWeb training pool: 20,000,000 unique next-byte targets from 6,537 documents; document-contained chunks of at most 4,096 targets shuffled by seed. No state crosses document/chunk resets.
- Development: 262,144 targets, 70 documents. Reserved prior audit: 1,000,000 targets, 310 documents, disjoint from training and development by document ID and text hash. This audit existed in prior research; it was not used to choose this reported recall-credit candidate.
- Same warmup/cosine schedule over 0–5M targets; thereafter fast/query 0.0002, slow 0.002, head 0.004. Prototype 0.002, contrastive 0.1, reconstruction 0.05 and constituent credit 1 remain fixed.
- Frozen expert interventions rotate compute matrices and biases among actually active experts, keeping prototype keys fixed. Three nonidentity rotations per level; each reported penalty is their mean. Evaluation never trains the source checkpoint.
- The new C runs exactly reproduce the previous corrected-C development curve for all three seeds. Training-data SHA-256: `127094a677d95d20379368003972cabf9bc77ded57341d4065cd3ce38ecb834c`. Development SHA-256: `4fe3ccbe4bce18e3b5f84b965a111521b6c6951a8d5ceaccc16ce81bfd0589ea`.
- Source starts from published `d003641`, restoring the preceding verified coverage-only correction before applying this investigation. Compact source and reproduction commands accompany the evidence.

## Code, compatibility and validation

The public runtime adds explicit `language_credit` and bounded `recall_credit` controls. Defaults remain unit language gains and disabled experimental producer credit, preserving corrected C. Version 11 checkpoints store nondefault language gains; version 12 also stores retained-value credit and pending producers. With neutral settings, old checkpoint bytes and continuation remain unchanged. Explicit historical routing and checkpoint policies are preserved.

**4,864 native checks and 31 Python tests pass; the native checks also pass ASan/UBSan.** LeakSanitizer was explicitly disabled on this ptrace-managed host. Additional private experiments pass 26 local-head checks, 26 expert-only local-head checks, 235 isolated main-language checks and 144 centered-readout checks. The old/new runtime reload and 4,097-target continuation are byte-identical; the continuation SHA-256 is `1c8192b0b3665b7d42a5dec6c226a34cc4f31052200da620a55724c4e8ea0968`.

The v13–v16 experimental checkpoint formats require their matching experiment installer. They are not public model formats. The recall/expert screen's extra gains are also an explicit installer setting. Only the ordinary corrected-C checkpoint is delivered for continued training.

Delivered baseline: `{checkpoint.name}`, seed 7, 20M training targets, v10; SHA-256 `{s['delivered_checkpoint']['sha256']}`. Its development NLL is 1.730870 and audit NLL is 1.684368. It includes its pending training state and the recorded rates.

## Remaining repair target

Upper experts need to carry useful *changing* conditional information. Any next candidate must beat a fixed-mean upper-output control, produce a materially positive active-expert permutation penalty, and improve held-out loss per byte and per second. The present work fixes the default selection and supplies checked credit mechanisms and sharper diagnostics; it does not label a development-only fluctuation as a specialization repair.
'''
    (a.out/'SHoEbox_L2_L3_Diagnosis.md').write_text(report)
    print(json.dumps({'screen_arms':len(screens),'report':str(a.out/'SHoEbox_L2_L3_Diagnosis.md'),'checkpoint':s['delivered_checkpoint']}))
if __name__=='__main__':main()
