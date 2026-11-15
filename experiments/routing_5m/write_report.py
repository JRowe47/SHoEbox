import argparse,json,math
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--work',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--runtime',type=Path,default=Path(__file__).resolve().parents[2])
args=parser.parse_args();w=args.work;out=args.out;repo=args.runtime
s=json.load(open(out/'SHoEbox_5M_Routing_Summary.json'));s=s.get('results',s);bench=json.load(open(w/'serial_benchmark.json'));export=json.load(open(w/'export_verification.json'))
arms=['D','C','D_warmup_only','C_warmup_only'];labels={'D':'D, original balancing','C':'C: D predictive features removed','D_warmup_only':'D, coverage warmup only','C_warmup_only':'C, coverage warmup only'}
results={a:[json.load(open(w/f'{a}_s{k}/results.json')) for k in (1,7,19)] for a in arms}
probes={a:[json.load(open(w/f'{a}_s{k}/probe.json')) for k in (1,7,19)] for a in arms}
diff=s['D_warmup_only']['development_mean']-s['D']['development_mean'];audit_diff=s['D_warmup_only']['audit_mean']-s['D']['audit_mean']
counts=np.array([r['targets'] for r in probes['D'][0]['audit']['documents']]);delta=np.mean([[b['nll']-a['nll'] for a,b in zip(probes['D'][i]['audit']['documents'],probes['D_warmup_only'][i]['audit']['documents'])] for i in range(3)],axis=0)
assert sum(counts)==1000000
rng=np.random.default_rng(20261115);draws=rng.integers(0,len(counts),(10000,len(counts)));vals=(delta[draws]*counts[draws]).sum(1)/counts[draws].sum(1);ci=np.percentile(vals,[2.5,97.5])
for a in arms:
 s[a]['delta_development_vs_D']=s[a]['development_mean']-s['D']['development_mean'];s[a]['delta_audit_vs_D']=s[a]['audit_mean']-s['D']['audit_mean']
# Equal training-time comparison uses an explicitly estimated common budget.
common=min(20e6/s[a]['serial_median_bps'] for a in arms)
for a in arms:
 t=common*s[a]['serial_median_bps'];loss=np.mean([np.interp(t,[q['targets'] for q in r['curve']],[q['nll'] for q in r['curve']]) for r in results[a]])
 s[a]['estimated_equal_time']=dict(seconds=common,targets=t,development_nll=float(loss))
lines=['# SHoEbox 5M: routing diagnosis and targeted rollback','', '15 November 2026 · FineWeb · 5M-parameter expert banks','',
 '**Decision: retain D’s predictive routing; remove the continuing usage-bias term.** Reverting D’s predictive addition to C barely changes loss. Removing the mature balancing pressure improves D in all three training seeds and on the separate audit set. The supplied model uses this corrected policy.','',
 f'Twelve matched runs completed 20M training-byte targets each. D’s mean development loss falls from **{s["D"]["development_mean"]:.6f} to {s["D_warmup_only"]["development_mean"]:.6f} nats/byte** ({diff:+.6f}). Mean audit loss falls from **{s["D"]["audit_mean"]:.6f} to {s["D_warmup_only"]["audit_mean"]:.6f}** ({audit_diff:+.6f}). Lower is better.','',
 '## Matched confirmation','', '| Setting | Development mean ± seed SD | Audit mean ± seed SD | Serial training bytes/s |','|---|---:|---:|---:|']
for a in arms:lines.append(f'| {labels[a]} | {s[a]["development_mean"]:.6f} ± {s[a]["development_sd"]:.6f} | {s[a]["audit_mean"]:.6f} ± {s[a]["audit_sd"]:.6f} | {s[a]["serial_median_bps"]:,.0f} |')
lines += ['', '| Seed | D development | D warmup-only development | Audit difference, corrected − D |','|---:|---:|---:|---:|']
for i,k in enumerate((1,7,19)):lines.append(f'| {k} | {s["D"]["development_by_seed"][i]:.6f} | {s["D_warmup_only"]["development_by_seed"][i]:.6f} | {s["D_warmup_only"]["audit_by_seed"][i]-s["D"]["audit_by_seed"][i]:+.6f} |')
lines += ['',f'The audit improvement wins 3/3 paired seeds. Resampling the 310 audit documents after averaging paired per-document differences across those fitted seeds gives a 95% interval of [{ci[0]:+.6f}, {ci[1]:+.6f}] nats/byte. This interval describes document variation; seed variation is shown separately above.','',
 f'With the balancing correction, D is {s["C_warmup_only"]["development_mean"]-s["D_warmup_only"]["development_mean"]:.6f} below C on mean development loss and {s["C_warmup_only"]["audit_mean"]-s["D_warmup_only"]["audit_mean"]:.6f} below C on mean audit loss. The predictive addition is a much smaller and less consistent effect than the balancing correction; these results do not support blaming it for the plateau.','',
 '## What caused the interference','',
 'The inherited C/D rule checked every expert against a 256-selection window. In the fast bank, 571 experts imply only **0.448 expected selections per expert per window**. The supposed starvation threshold was 10% of that, **0.0448 selections**. With integer counts, it rewarded every expert absent from that short window. Subtracting the bank-wide mean made this a relative penalty on recently used experts, including useful frequent specialists.','',
 'The fix removes the residual starvation term as coverage warmup fades. It keeps centered normalized cues, exact search, prototype adaptation, the calibrated gate, predictive SDR features, and the original coverage term. The least-used expert still controls the fade toward 256 training selections; accumulated coverage bias then decays on a 256-selections-per-expert horizon.','',
 'At 5M training bytes, clearing saved bias from an already-trained D model slightly worsened loss (1.801979 → 1.803691). At 20M it helped by 0.006858 in seed 7, while the model trained with the corrected rule was 0.033571 better. The main gain here comes from the learning trajectory, not merely deleting bias at evaluation time.','',
 '## Screening and hyperparameters','',
 'All screen rows use seed 7, the same bank allocation, identical byte order, and 10M training targets. They retain the repaired recall and learning-credit implementation; A is a router ablation, not a destructive revert of the entire D promotion commit.','',
 '| Screen arm | Development NLL at 10M |','|---|---:|']
sl={'A':'Earlier uncentered router A','B':'Centered router B; no coverage balancing','C':'C','D':'D','D_no_balance':'D; all balancing disabled from initialization','D_warmup_only':'D; coverage warmup only','C_warmup_only':'C; coverage warmup only','D_L1_only':'D predictive features retained only at L1','D_high_rate':'D; 4× continuation rates after 5M'}
for a,label in sl.items():
 r=json.load(open(w/f'{a}_s7/results.json'));v=next(q['nll'] for q in r['curve'] if q['targets']==10000000);lines.append(f'| {label} | {v:.6f} |')
lines += ['', 'Removing all balancing from the start underperformed retaining initial coverage. Disabling D’s L2/L3 predictive additions also failed to reproduce the larger benefit of retiring mature balancing.','',
 'All four confirmation arms use the same 0–5M warmup/cosine schedule, then fast/query **0.0002**, slow **0.002**, and head **0.004**. Prototype attraction remains 0.002; contrastive and reconstruction weights remain 0.1 and 0.05. The higher-rate 5M-model screen was slightly better at 10M than ordinary D, so the earlier 100M learning-rate finding does not transfer mechanically to this model size. The matched confirmation isolates routing from that rate question.','',
 '## Learning efficiency and the remaining bottleneck','',
 'The corrected D model confirmed loss ≤1.75 after **12M, 12M, and 16M bytes** for seeds 1, 7, and 19: a mean **13.33M**. Confirmation requires three consecutive evaluations. Original D and C confirmed this target in **0/3 seeds** through 20M. Corrected C confirmed it at 17M, 12M, and 11M.','',
 f'The corrected D runtime changes serial throughput by {100*(s["D_warmup_only"]["serial_median_bps"]/s["D"]["serial_median_bps"]-1):+.2f}%. At a common estimated serial training budget of {common:.1f} seconds, interpolated mean losses are {s["D"]["estimated_equal_time"]["development_nll"]:.6f} for original D and {s["D_warmup_only"]["estimated_equal_time"]["development_nll"]:.6f} for corrected D. At the same budget, original C is {s["C"]["estimated_equal_time"]["development_nll"]:.6f} and corrected C is {s["C_warmup_only"]["estimated_equal_time"]["development_nll"]:.6f}. Corrected C is {100*(s["C_warmup_only"]["serial_median_bps"]/s["D_warmup_only"]["serial_median_bps"]-1):.2f}% faster than corrected D and slightly ahead in this estimated equal-time comparison. D has a small mean advantage per training byte; C remains a useful speed comparator. These time-normalized values use separately measured throughput and the byte curves; they are not direct stop-at-time experiments.','',
 '| Seed-7 expert swap | Original D NLL change | Corrected D NLL change |','|---|---:|---:|']
for level in (1,2,3):
 q=probes['D'][1]['frozen'];f=probes['D_warmup_only'][1]['frozen'];lines.append(f'| L{level} | {q[f"swap{level}"]["nll"]-q["full"]["nll"]:+.7f} | {f[f"swap{level}"]["nll"]-f["full"]["nll"]:+.7f} |')
late={a:np.mean([next(q['nll'] for q in r['curve'] if q['targets']==10000000)-r['curve'][-1]['nll'] for r in results[a]]) for a in ('D','D_warmup_only')}
lines += ['', f'The fix lowers the curve but does not eliminate slow late learning: from 10M to 20M, mean development loss improves by {late["D"]:.6f} for D and {late["D_warmup_only"]:.6f} for corrected D. L2/L3 expert swaps still have negligible effect, although masking those levels hurts prediction. The hierarchy contributes mainly through shared paths rather than useful expert-specific specialization.','',
 'The fresh 5M-model gradient audit finds L2 contrastive gradient energy about **11.35×** language-gradient energy in corrected D; L3 is not similarly dominated in this run. The next focused test is to scale **language-derived credit separately at L2/L3**, keeping auxiliary update rates fixed, and require expert-swap penalties to become meaningfully positive.','',
 '## Model and reproducibility','',
 f'- Supplied checkpoint: **{export["stats"]["parameters"]:,} parameters**, width 64, expert banks 571 / 285 / 171 / 116, seed 7, exactly 20,000,000 training targets.','- C controls have 4,999,743 parameters. D adds 12,288 shared predictor parameters while preserving the same expert banks.',
 f'- Supplied model development NLL: **{export["development_nll"]:.6f}**; audit NLL: **{export["audit_nll"]:.6f}**.',
 '- Each seed shuffles the same 20M unique next-byte targets in document-contained chunks of at most 4,096 targets. Each target occurs once per run. Context resets at chunk boundaries.',
 '- Development: 70 documents / 262,144 targets. Separate audit: 310 documents / 1,000,000 targets, disjoint from current training by document ID and full-text hash. This is the archived audit from the preceding diagnosis, scored here after training.',
 '- Twelve confirmation runs account for 240M targets; five additional screen-only arms account for 50M. Diagnostic and verification continuations are disposable and excluded from reported checkpoints.',
 '- Serial timing: one AMD EPYC 9V74 CPU core, three 200K-target repetitions per policy, paired by seed with alternating order, after training/probes/tests stopped. Original native code times legacy controls; corrected production code times warmup-only arms. Loading, compilation and checkpoint writes are excluded.',
 '- Raw training/evaluation/checkpoint times are recorded. Concurrent training and offline verification made those wall times descriptive; the independent serial comparison supplies the throughput figures.',
 '- Source baseline: local continuation/diagnostic commit `1ed7285`, descended from the D promotion `d003641`. The evidence archive contains the patch, complete small runtime, data manifests and splits, per-seed curves, per-document scores, and verification logs. It contains no extra trained snapshots.','',
 '## Implementation and verification','',
 'New implicit public models use routing D with the version-10 warmup-only policy. Saved v8/v9 models retain their historical behavior. Explicit C/D controls preserve v9 unless `balance_policy="warmup"` is specified. The supplied checkpoint already stores v10 and needs no experimental global flags.','',
 'Passed: **2,093 native numerical checks, 28 Python tests, 19 semantic-lace checks, and 56 routing checks**, plus ASan/UBSan for the native, lace, and routing suites. LeakSanitizer was explicitly disabled for this managed host. Two new test-fixture issues were corrected: an invalid semantic-lace/routing combination and excessive stack allocation under ASan. Production arithmetic was unchanged by those fixture repairs.','',
 'Neutral diagnostic controls match A/B/C/D exactly. The production policy matches the tested warmup-only loss and learned state exactly, while historical C/D continuation remains byte-identical. The exported model reloads to the recorded development and audit scores and exactly reproduces a 4,097-target continuation. Those verification targets were not saved into the model. The packaged runtime was extracted and compiled in a fresh cache, then reproduced the same continuation exactly. The packaged diagnostic builder and all screen-policy parity/continuation checks also passed.','',
 f'Checkpoint SHA-256: `{export["checkpoint_sha256"]}`.','',
 '![Primary comparison](SHoEbox_5M_Routing_Diagnosis.png)','']
(out/'SHoEbox_5M_Routing_Diagnosis.md').write_text('\n'.join(lines))
selection=dict(selected='D with v10 coverage-warmup-only balancing',model_parameters=5012031,training_targets_per_seed=20000000,seeds=[1,7,19],results=s,audit_paired_document_interval=ci.tolist(),audit_delta=float(audit_diff),development_delta=float(diff),throughput=bench['summary'],export={k:v for k,v in export.items() if k!='stats'},verification=json.load(open(w/'verification_summary.json')))
(repo/'multiscale/results/routing_5m_selection.json').write_text(json.dumps(selection,indent=2)+'\n');(out/'SHoEbox_5M_Routing_Summary.json').write_text(json.dumps(selection,indent=2)+'\n')
print(json.dumps({'development_delta':diff,'audit_delta':audit_diff,'audit_document_interval':ci.tolist(),'common_estimated_seconds':common,'late_improvements':late},indent=2))
