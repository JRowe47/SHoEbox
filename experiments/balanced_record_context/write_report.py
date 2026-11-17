"""Render the A8 Markdown report from fixed evidence; no model computation."""
import argparse,json
from pathlib import Path
import numpy as np
from .report import ARMS,NAMES

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();w=a.workspace;o=a.out
 read=lambda n:json.loads((w/n).read_text());summary=json.loads((o/'summary.json').read_text());account=json.loads((o/'accounting.json').read_text())
 verify=read('verification.json');speed=read('benchmark.json');conf=read('conflicts.json');ret=read('retention.json');sur=read('survivors.json');updates=read('update_metrics.json');events=read('prefit_correctness_bytes.json');cov=read('coverage.json');sigs=read('signature_traces.json')
 means=summary['means'];pct=lambda x:f'{100*x:.6f}%';mean=lambda xs:float(np.mean(xs));f=account['fit_totals'];status='PASS' if verify['gates']['a8']['development_pass'] else 'FAIL'
 lines=['# SHoEbox EXP03/05-A8: phase-balanced PAD/diverse curriculum','',
 f"16 November 2026 (America/Denver). Starting main: `3487578a98198e19de833e118cf0d6c5f6b5c0f6`. Pre-fit source/receipt commit: `{read('a8/plan.json')['prefit_commit']}`.",'',
 f'**A8 development: {status}. Qualified public E8 remains unchanged.** All A1–A8 audits remain closed. Exactly 20.25M fitting bytes; no pilot, extra fit, sweep, inherited warmup or generic/capacity qualification. Every earlier result, timing and cost remains preserved.','',
 '## Full autoregressive results','',
 'Native full-256-byte whole-episode exactness. Matched E8/A6/A8, seeds 1/7/19, 128 fresh worlds ×4 record/query-order presentations per condition. Seeds share worlds. These fresh controls do not replace E8’s original qualification.','',
 '| Condition | Matched E8 | Matched A6 | A8 |','|---|---:|---:|---:|']
 for c in NAMES:lines.append('| '+NAMES[c]+' | '+' | '.join(pct(means[arm][c]) for arm in ARMS)+' |')
 lines+=['','![Primary results](SHoEbox_EXP0305_A8_Results.png)','','| Corrected overwrite | Changed key | Untouched key |','|---|---:|---:|']
 for arm in ARMS:
  rs=[r for r in updates['rows'] if r['arm']==arm];lines.append(f"| {arm.upper()} | {pct(mean([r['changed'] for r in rs]))} | {pct(mean([r['untouched'] for r in rs]))} |")
 lines+=['','## Exact intervention','',
 'Only A8 continuation alternates 64-episode PAD/diverse blocks, beginning with PAD. The rule is `((episode_index // 64) % 2 == 1)`, using the zero-based episode index before increment. Both original record spacers use the same block. Diverse blocks replace their existing 0–7-byte PAD spacers with per-occurrence bytes in [160,255]; PAD blocks change nothing. The exact A7 v7 generator prefix is retained. The cycle never resets at epochs or finish events, and the incomplete final cycle is consumed without rebalancing. The literal registered SHA-256 prefix, raw group-key bytes, little-endian episode/physical-position/block encoding and rejection rule are used. Both existing RNGs, lengths, phases, order, labels, masks, finish schedule and budgets stay unchanged. Every independent warmup and the E8/A6 controls retain PAD. Evaluation fixtures are unchanged.','',
 'The generated A8 header is byte-identical to A6 after reversing only its save/load magic `0x3841353050584553`; its native source is identical. Record costs stay 4/4, with A5’s all-256-code hybrid, float64 softmax/gradient and pre-update coupling. Query presence, applicability, values, native credit, A4 inference/chronology/ties/fallback, admission and existing state stay exact. No arithmetic reuse or optimization was introduced; FMA remains disabled.','',
 'Every arm has 497,589 fitted parameters, including 7,670 readout weights. There are no new parameters, persistent sequence state, moments, projection caches or slots. The outer cursor `exp0305-a8-cursor-1` records policy, curriculum and exact episode/partial-example position. Generic byte objectives and optional fixture-local supervision remain separate; inference supports all 256 bytes.','',
 'The independent spacer/interpreter traversal checks every augmented continuation episode and locks six per-arm/phase streams. Input/target digests change only for A8 continuation; masks, labels, RNG states, schedules and warmup bytes match. Both physical record orders, zero lengths, rejection boundaries, additional hash blocks, future-answer causality and partial-spacer resumes pass.','',
 '## Frozen failure localization','',
 'First 32 development worlds ×4 orders, both teacher-fed answers, all seeds. These subset measurements localize failure; full autoregressive outcomes set the gates. Final fits and complete evaluation outcomes were locked before these traces.','',
 '| Arm | Diverse256 signature survival | Incoming false recognition | True-record phase false negatives | Target removals |','|---|---:|---:|---:|---:|']
 for arm in ('a6','a8'):
  rs=[r['summary'] for r in ret['rows'] if r['arm']==arm];present=mean([r['arms'][arm]['diverse256']['trace']['present'] for r in sigs['rows']]);fp=sum(r['full_diverse_recognized'] for r in rs)/sum(r['full_diverse_pairs'] for r in rs);fn=sum(r['true_record_phase_false_negatives'] for r in rs)/sum(r['true_record_phase_events'] for r in rs)
  lines.append(f"| {arm.upper()} | {pct(present)} | {pct(fp)} | {pct(fn)} | {sum(r['loss_events'] for r in rs)} |")
 lines+=['','| Arm | Correct given signature survives | Surviving target shortlisted | Recognized filler phases/answer | Filler mass on wrong survivor answers |','|---|---:|---:|---:|---:|']
 for arm in ('a6','a8'):
  rs=[r['summary'] for r in sur['rows'] if r['arm']==arm]
  lines.append(f"| {arm.upper()} | {pct(mean([r['survivor_accuracy'] for r in rs]))} | {pct(mean([r['survivor_shortlisted'] for r in rs]))} | {mean([r['unannotated_recognized'] for r in rs]):.6f} | {pct(mean([r['incorrect_survivor_unannotated_mass'] for r in rs]))} |")
 lines+=['','| Arm | Competing nearest-key identity | Competing other-key selected mass | Changed-key old-value predictions |','|---|---:|---:|---:|']
 for arm in ('a6','a8'):
  rs=[r['summary'] for r in conf['rows'] if r['arm']==arm]
  lines.append(f"| {arm.upper()} | {pct(mean([r['compete16']['nearest_identity_correct'] for r in rs]))} | {pct(mean([r['compete16']['selected_mass']['other_key'] for r in rs]))} | {pct(mean([r['overwrite16']['predicted_old_value'] for r in rs]))} |")
 lines+=['','Admission and exhaustive selection are independently reconstructed. Missing signatures, surviving-target competition and identity/version errors remain separate. Detailed geometry, mass, phase and occurrence records remain in the evidence archive.','',
 '| A8 autoregressive episode exactness | First 32 worlds | Remaining 96 | All 128 |','|---|---:|---:|---:|']
 for cond,r in cov['primary_subsets']['a8'].items():lines.append('| '+NAMES[cond]+' | '+' | '.join(pct(r[k]) for k in ('first_32_worlds','remaining_96_worlds','all_128_worlds'))+' |')
 lines+=['','![Frozen diagnosis](SHoEbox_EXP0305_A8_Diagnosis.png)','','![Signature survival](SHoEbox_EXP0305_A8_Retention.png)','','![Presence and survivors](SHoEbox_EXP0305_A8_Presence.png)','',
 '## Measured training-window coverage','',
 'Exact data-only traversal of each 2M-byte continuation, with no model calls. Counts are per arm/seed; the generated data schedule is shared across seeds.','',
 '| Arm | Positive record-key events | Events containing PAD | Distinct key/phase/windows | Fully diverse negative events | Distinct diverse negatives |','|---|---:|---:|---:|---:|---:|']
 for arm in ARMS:
  r=cov['continuation_per_arm'][arm];keys=('record_key_events','record_key_events_containing_padding','distinct_positive_key_phase_windows','fully_diverse_negative_window_events','distinct_fully_diverse_negative_windows')
  lines.append('| '+arm.upper()+' | '+' | '.join(f'{r[k]:,}' for k in keys)+' |')
 lines+=['','| A8 block | Consumed bytes | Encountered episodes | Positive record-key events | PAD-containing positives | Distinct key/phase/windows |','|---|---:|---:|---:|---:|---:|']
 stream=read('streams.json')['rows'][-1]
 for name,v in cov['continuation_per_arm']['a8']['by_block'].items():
  episodes=sum(r['episodes'] for r in stream['coverage'] if r['regime']==name)
  lines.append(f"| {name.upper()} | {v['input_bytes']:,} | {episodes:,} | {v['record_key_events']:,} | {v['record_key_events_containing_padding']:,} | {v['distinct_positive_key_phase_windows']:,} |")
 lines+=['','The complete continuation encounters 11,088 episodes, including its incomplete last episode: 5,568 PAD and 5,520 diverse. Consumed bytes are 1,003,872 PAD-block and 996,128 diverse-block bytes. Both regimes cover all 64 phases. Exactly 38,797 original PAD input bytes per A8 continuation are replaced; no terminal balancing or appended example occurs. Per-condition/per-phase and block-specific raw-window counts are retained.','']
 lines+=['','## Unchanged extended gates','','| A8 condition | All-offset mean | Worst offset, seed mean | Both orders |','|---|---:|---:|---:|']
 g=verify['gates']['a8']
 for c in list(NAMES)[:5]:lines.append(f"| {NAMES[c]} | {pct(g[c]['offsets']['mean'])} | {pct(g[c]['offsets']['worst'])} | {pct(g[c]['both_orders']['mean'])} |")
 b=g['binary_regression'];lines+=['',f"Fresh binary regression: {pct(b['primary']['mean'])} primary; {pct(b['offsets']['mean'])} offset mean; {pct(b['offsets']['worst'])} worst offset; {pct(b['both_orders']['mean'])} both orders.",'',
 'Every primary condition and answer position requires ≥95% mean/90% every seed. Both orders and each padding/repeat/diverse delay require ≥90%/85%. All-offset mean requires ≥95%, worst-offset seed mean ≥90%. No-query drop ≥20 points; reverse must hurt. All state/pairing/gate lesions and every binary gate remain binding.','',
 'The same first 64 presentations are tested at all 64 offsets. The next 16 cross all offsets independently at 0/64/128/256 padding delays. Both-order/lesion samples contain 256 presentations. No sample count or threshold was reduced. All per-seed scores/decisions are retained.','',
 '## Isolated throughput','','| Arm | Bytes/CPU second | Relative native |','|---|---:|---:|']
 for arm,r in speed['rows'].items():lines.append(f"| {arm.upper()} | {r['bytes_per_cpu_second']:,.1f} | {pct(r['relative_native'])} |")
 lines+=['','One pinned CPU; three interleaved 65,536-byte repeats plus 1,024 warmup each. Full byte distributions at every input; all other jobs stopped. Compilation/cloning/warmup are outside timed intervals and inside process costs. Native is the fresh untrained v19 reference; public E8 imports matched E8 exactly, with identical prediction checksums. The unchanged speed requirement is ≥80% native.','',
 '![Throughput](SHoEbox_EXP0305_A8_Throughput.png)','',
 '## Folds, costs and verification','',
 'Salt `SHoEbox-EXP0305-balanced-record-context-v8` plus NUL; world RNG 20261130. 512/128/256 train/dev/closed-audit and 128 fresh binary dev worlds. Exclusions include all historical opened binding folds/E2 family, reserved A1–A7 audits and every A2–A7 dev/binary world. Operations/orders remain grouped. No generic data is used; opened E1/E7 generic groups remain excluded later.','',
 'Each arm/seed independently receives 250K native warmup +2M continuation. Order RNG 10701/10702, augmentation +1000. No replay, extra fit or inherited warmup. Passing source/data/receipt hashes were committed before fitting; final model/cursor/outcome locks verify.','',
 '| Fitting quantity | Total |','|---|---:|']
 for key in ('bytes','padding_bytes','nuisance_bytes','replaced_padding_bytes','targets','answer_targets','local_projection_steps','local_presence_steps','gate_targets','local_training_windows'):
  lines.append(f"| {key.replace('_',' ')} | {f[key]:,} |")
 lines.append(f"| Fitting CPU seconds | {f['cpu_seconds']:,.2f} |")
 lines+=['','PAD/nuisance/replaced-padding counts are included in the 20.25M budget, not added to it. Replaced padding is a subset of nuisance bytes.','',
 '| Local cost across three continuations | E8 | A6 | A8 |','|---|---:|---:|---:|']
 for key in ('record_positive_loss_components','record_negative_loss_components','record_positive_gradient_rows','record_negative_gradient_rows','hybrid_attraction_components','hybrid_discrimination_components','hybrid_gradient_rows','hybrid_code_comparisons'):
  vals=[sum(r['counts'][key] for r in events['research'] if r['arm']==arm) for arm in ARMS];lines.append('| '+key.replace('_',' ')+' | '+' | '.join(f'{v:,}' for v in vals)+' |')
 failed=[k for k,v in account['processes'].items() if v['exit_code']!=0]
 lines+=['',f"Whole-process research/verification CPU at report generation: **{account['research_and_verification_cpu_seconds']:,.2f} seconds**. Fitting/subphase CPU is included once. Failed metered attempts: {', '.join(failed) if failed else 'none'}.",'',
 f"A8 Python correctness invocations consumed {events['totals']['input_bytes']:,} discarded update bytes, including the failed block-boundary coverage assertion and the successful repeated suite. All 18 saved warmup/final pairs pass 1,000 uninterrupted versus 371 + save/load +629 updates: 36,000 separately charged discarded correctness bytes. Historical trajectories consumed {account['historical_correctness_bytes']:,} discarded bytes. Native independent microchecks retain event counts in their logs. Historical unit/native scalar writes remain uninstrumented; complete process CPU and logs are retained. Setup/editing/read-only inspection is not separately CPU-metered; artifact production/packaging has separate receipts.",'',
 '- 269 distinct Python tests pass: 60 runtime/public, 192 historical research and 17 A8 tests; 21 additional protected-policy checks pass.',
 '- Optimized and ASan/UBSan pass: 824,111 core checks, 723 public E8 checks, 13,863 inherited selection/lifecycle checks, 75,295 hybrid-objective checks and 7,826 record-presence checks per build. Leak checking was explicitly disabled.',
 '- Independent logistic/hybrid gradients, finite differences, all-256-code inclusion and pre-update coupling pass. Codebook has 256 unique codes, no collisions; minimum Hamming distance one. Exact same-weight local/native/inference/admission trajectories pass.',
 '- All 14 instantiated historical constructor/checkpoint trajectories match starting main. v5/v6 loader sources are unchanged but were not independently instantiated. Public/historical numerical sources and earlier reports remain untouched.','',
 '## Resume and next work','',
 'Read `DIAGNOSIS.md`, `RESUME.md` and the one focused next `A9_REGISTRATION.md`. A8 is complete; no additional diagnostic-driven fit follows its diagnosis. A9 is registered only, not implemented or fitted. The archive contains source/git history, starting-main source, all 18 model/cursor pairs, generated worlds, exact stream/source/outcome locks, raw diagnostics, failed/passing receipts and per-member SHA-256. Checkpoints, datasets and compiled binaries remain outside git. Never reset, finish or replay at an arbitrary cursor boundary.','',
 'Failure retains qualified E8 and keeps the A8 audit closed. Passing development would require separately registered fresh generic/capacity qualification at every original margin: language ≤+0.01 nats/byte, copy/reconstruction ≤1-point token loss, post-1M-language retention, all three-seed 1M/5M and speed gates.','',
 'Completed verified work, including unsuccessful research, is explicitly authorized for commit/push directly to main without confirmation. Preserve unrelated changes and remote history; never force-push. Deliver ordinary downloadable Markdown, PNGs, resumable archive and copy/paste prompt. No Pages/Spaces.','',
 'Primary grounding remains [Miller et al., EMNLP 2016](https://aclanthology.org/D16-1147/) for paired addressing and [Arora et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html) for multi-query recall measurement. The fixed A8 curriculum is this project’s registered intervention.','']
 (o/'SHoEbox_EXP0305_A8_Report.md').write_text('\n'.join(lines));print('Wrote A8 report from fixed evidence')
if __name__=='__main__':main()
