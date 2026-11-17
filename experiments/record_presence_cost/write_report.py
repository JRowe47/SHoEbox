"""Render Markdown directly from immutable A6 results and accounting."""
import argparse,json
from pathlib import Path
import numpy as np
from .report import ARMS,NAMES

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();w=a.workspace;o=a.out
 read=lambda n:json.loads((w/n).read_text())
 summary=json.loads((o/'summary.json').read_text());account=json.loads((o/'accounting.json').read_text());verify=read('verification.json');ext=read('extended/results.json');speed=read('benchmark.json');conf=read('conflicts.json');ret=read('retention.json');sur=read('survivors.json');identity=read('identity_attribution.json');updates=read('update_metrics.json');events=read('prefit_correctness_bytes.json');coverage=read('coverage.json')
 means=summary['means'];pct=lambda x:f'{100*x:.4f}%';mean=lambda xs:float(np.mean(xs));f=account['fit_totals'];status='PASS' if verify['gates']['a6']['development_pass'] else 'FAIL'
 lines=['# SHoEbox EXP03/05-A6: fixed record-presence negative cost','',
 '16 November 2026 (America/Denver). Starting main: `803dc86e563ff048866023bb536830ce0f28ea98`. Pre-fit source/receipt lock: `1ddb724`.', '',
 f'**A6 development: {status}. Qualified public E8 remains unchanged.** All A1–A6 audits remain closed. No generic/capacity qualification, pilot or sweep was run. Every earlier result, failed attempt and cost is preserved.', '',
 '## Primary results','',
 'Full-256-byte autoregressive whole-episode exactness; 128 fresh development worlds ×4 order presentations per condition, seeds 1/7/19. Seeds share worlds. These newly fitted matched controls do not replace E8’s original qualification.', '',
 '| Condition | Matched E8 | Matched A5 | A6 |','|---|---:|---:|---:|']
 for c in NAMES:lines.append('| '+NAMES[c]+' | '+' | '.join(pct(means[arm][c]) for arm in ARMS)+' |')
 lines+=['','![Primary results](SHoEbox_EXP0305_A6_Results.png)','',
 '| Corrected overwrite | Changed key | Untouched key |','|---|---:|---:|']
 for arm in ARMS:
  rs=[r for r in updates['rows'] if r['arm']==arm];lines.append(f"| {arm.upper()} | {pct(mean([r['changed'] for r in rs]))} | {pct(mean([r['untouched'] for r in rs]))} |")
 lines+=['','## Exact implementation and verification','',
 'The four record-presence classifier errors are `(p-y)*4.f`, with the original float32 affine/sigmoid arithmetic, phase/event order, `rate*auxiliary`, normalized affine step and disabled FMA. Positive updates remain bit-identical; negative error is exactly fourfold at identical pre-update input/weights. Stable softplus is used only for reference losses.', '',
 'A6 inherits A5’s exact all-256-code hybrid, float64 softmax/gradients and pre-update coupling. Query presence, applicability, value reconstruction, native responsibility credit, A4 inference/chronology/ties/fallback, admission, prior and 16 slots remain unchanged. The source-isolation test reverses only the record-error line and magic to recover A5 exactly. Same-weight trajectories verify identical inference, admission and generic/native learning. No arithmetic reuse or vectorization was introduced.', '',
 'There are 497,589 fitted parameters per arm, including 7,670 readout weights. A6 adds zero parameters, persistent sequence state, optimizer moments, slots or projection caches. Its outer magic is `0x3641353050584553`, nested v19; all historical constructors/loaders retain their meanings. Generic byte objectives do not supply or learn local labels. Inference uses the complete byte alphabet.', '',
 'All 256 fixed codes remain present: zero collisions, minimum Hamming distance one; little-endian float32 SHA-256 `c8292df6bb4ea5e5a6c98daaf9796d008d03b0672107744637a13548a785559a`. Unchanged training-only codebook: 16,384 bytes; hybrid scratch 2,256 bytes plus 64-byte z. Existing selection scratch: 6,968 bytes; prediction scratch: 2,408 bytes. None is new persistent sequence state.', '',
 '- 237 distinct Python tests pass: 60 runtime/public, 165 historical/research, 12 A6. Seventeen additional bidirectional protected-policy/import checks pass.',
 '- Optimized and ASan/UBSan builds pass: 824,111 core checks; 723 public E8 checks; 13,863 inherited numerical/request/admission/exhaustive-selection checks; 75,295 hybrid checks; 7,826 A6 presence checks per build. LSan was explicitly disabled.',
 '- Independent Python logistic and hybrid references, long-double native references, finite differences, all phases/label masks, unchanged other updates, exact positive steps and fourfold negative errors pass. Maximum native logistic-gradient discrepancy: 2.81436e-7; hybrid: 5.88852e-8.',
 '- Fourteen historical constructor/checkpoint trajectories match starting main. Original v5/v6 loader sources are unchanged but were not separately instantiated.',
 '- All 18 actual warmup/final model/cursor pairs pass 1,000 uninterrupted versus 371 + save/load +629 updates exactly, including interrupted schedule boundaries. Those 36,000 discarded correctness bytes do not extend selected fits.', '',
 '## Separate memory and readout diagnosis','',
 'Frozen first-32-world ×4-order teacher-fed traces localize errors; they do not replace autoregressive gates. Incoming false recognition below is the maximum over four phase classifiers on fully diverse eight-byte pairs.', '',
 '| Arm | Diverse256 signature survival | Incoming false recognition | True-record phase false negatives | Removed target events |','|---|---:|---:|---:|---:|']
 for arm in ('a5','a6'):
  rs=[r['summary'] for r in ret['rows'] if r['arm']==arm];present=mean([r['arms'][arm]['traces']['diverse256']['trace']['present'] for r in ext['rows']]);fp=sum(r['full_diverse_recognized'] for r in rs)/sum(r['full_diverse_pairs'] for r in rs);fn=sum(r['true_record_phase_false_negatives'] for r in rs)/sum(r['true_record_phase_events'] for r in rs)
  lines.append(f"| {arm.upper()} | {pct(present)} | {pct(fp)} | {pct(fn)} | {sum(r['loss_events'] for r in rs)} |")
 lines+=['','| Arm | Correct given target survives | Surviving target recognized | Surviving target shortlisted | Recognized filler phases/answer | Filler mass on wrong survivor answers |','|---|---:|---:|---:|---:|---:|']
 for arm in ('a5','a6'):
  rs=[r['summary'] for r in sur['rows'] if r['arm']==arm]
  vals=[pct(mean([r[k] for r in rs])) for k in ('survivor_accuracy','survivor_recognized','survivor_shortlisted')]
  lines.append(f"| {arm.upper()} | {' | '.join(vals)} | {mean([r['unannotated_recognized'] for r in rs]):.4f} | {pct(mean([r['incorrect_survivor_unannotated_mass'] for r in rs]))} |")
 lines+=['','| Arm | Competing nearest-key identity | Competing unrelated-key mass | Changed-key old-value predictions |','|---|---:|---:|---:|']
 for arm in ('a5','a6'):
  rs=[r['summary'] for r in conf['rows'] if r['arm']==arm]
  lines.append(f"| {arm.upper()} | {pct(mean([r['compete16']['nearest_identity_correct'] for r in rs]))} | {pct(mean([r['compete16']['selected_mass']['other_key'] for r in rs]))} | {pct(mean([r['overwrite16']['predicted_old_value'] for r in rs]))} |")
 lines+=['','Every recognized A5/A6 selected pair and weight agrees with the external exhaustive calculation. Signature absence, presence rejection, surviving-target selection and identity/version ambiguity remain separate fields. Exact removal reasons, incoming/target logits, selected masses and all vectors are retained in the archive and summarized in `DIAGNOSIS.md`.', '',
 'The first-32-world diagnostic is easier than the full endpoint: A6 diverse256 autoregressive whole-episode exactness is 93.75% on those worlds, 81.25% on the remaining 96, and 84.375% overall. Teacher-fed answer accuracy of 96.875% is a different measure on that subset. It does not establish full-population signature survival or readout accuracy.', '',
 'Data-only traversal finds 26,255 of 28,856 positive record-key windows (90.9863%) contain PAD, with 5,997 distinct key/phase/raw-window combinations. The 9,963 fully diverse negative windows cover 3,252 unique windows. Competing-trace record-key reconstruction squared error is 3.288515 versus 0.158972 for queries; wrong nearest identities differ by 4–6 fixed-code bits, so they are not code collisions. This motivates the fixed A7 record-context curriculum registered below.', '',
 '![Memory and admission](SHoEbox_EXP0305_A6_Retention.png)','',
 '![Presence and surviving-target outcomes](SHoEbox_EXP0305_A6_Presence.png)','',
 '## Unchanged extended gates','',
 '| A6 condition | All-offset mean | Worst offset, seed mean | Both orders |','|---|---:|---:|---:|']
 g=verify['gates']['a6']
 for c in list(NAMES)[:5]:lines.append(f"| {NAMES[c]} | {pct(g[c]['offsets']['mean'])} | {pct(g[c]['offsets']['worst'])} | {pct(g[c]['both_orders']['mean'])} |")
 b=g['binary_regression'];lines+=['',f"Fresh binary canonical regression: {pct(b['primary']['mean'])} primary; {pct(b['offsets']['mean'])} offset mean; {pct(b['offsets']['worst'])} worst offset; {pct(b['both_orders']['mean'])} both orders.",'',
 'Primary and each position require ≥95% mean/90% every seed. Both orders and every padding/repeat/diverse delay require ≥90%/85%. All-offset mean requires ≥95%, worst-offset seed mean ≥90%. No-query drop ≥20 points; reverse must hurt. All state/pairing/gate lesions and every binary gate remain binding.', '',
 'Primary/long-filler conditions use 512 presentations. The same first 64 are evaluated at all 64 offsets. The next 16 are crossed with every offset at each independent 0/64/128/256 padding delay. Both-order and lesion samples use 256 presentations. Every per-seed decision is in `verification.json`; no sample or threshold was reduced.', '',
 '## Isolated throughput','',
 '| Arm | Bytes / CPU second | Relative native |','|---|---:|---:|']
 for arm,r in speed['rows'].items():lines.append(f"| {arm.upper()} | {r['bytes_per_cpu_second']:,.1f} | {pct(r['relative_native'])} |")
 lines+=['','One pinned CPU, three interleaved 65,536-byte repeats plus 1,024 warmup each, full 256-byte output every byte. All other local research jobs had finished. Compilation/cloning/warmup are outside timed intervals and inside process cost. Native is a fresh untrained v19 reference; public E8 imports matched E8 exactly and prediction checksums agree. The unchanged candidate gate is ≥80% native.', '',
 'The separate public-E8 import control measures 78.74% here, also below 80%, despite its exact prediction-checksum match. This timing is retained without retry or replacement. It does not overwrite the original qualified E8 measurement; no public runtime source changed.', '',
 '![Throughput](SHoEbox_EXP0305_A6_Throughput.png)','',
 '## Folds, locks and costs','',
 'Salt `SHoEbox-EXP0305-record-presence-cost-v6` plus NUL; world RNG 20261128. Counts: 512/128/256 train/dev/closed-audit worlds plus 128 fresh binary dev worlds. Exclusions cover the complete historical opened binding-fold union/E2-family, reserved A1–A5 audits and all A2–A5 dev/binary worlds. Operations/orders stay grouped. No generic data is used; opened E1/E7 generic groups remain excluded from later qualification.', '',
 'The pre-fit commit locks 967 source dependencies, passing receipts, exact folds and independent interpreter/shortcut results before fitting. Interpreter agreement covers 12,800 primary, 5,120 changed-key/order and 512 binary presentations. First/latest/constant repeated answers score zero; a fixed query-blind answer pair is bounded at 50% by distinct answers and both query orders. Causal labels never inspect future answers.', '',
 'Each arm/seed independently receives 250K native warmup +2M continuation, order RNG 8701/8702 and augmentation +1000. No pilot, inherited warmup, refit, extra control, replay or sweep. Final outcomes/checkpoints/cursors are locked before diagnosis.', '',
 '| Fitting quantity | Total |','|---|---:|']
 for label,key in [('Charged input bytes','bytes'),('Padding bytes, included above','padding_bytes'),('Native objective targets','targets'),('Answer targets, included above','answer_targets'),('Local projection-row updates','local_projection_steps'),('Local presence updates','local_presence_steps'),('Applicability updates','gate_targets'),('Local training windows','local_training_windows')]:lines.append(f"| {label} | {f[key]:,} |")
 lines.append(f"| Fitting CPU seconds | {f['cpu_seconds']:,.2f} |")
 lines+=['','| Per-arm local cost, three continuations | E8 | A5 | A6 |','|---|---:|---:|---:|']
 for key in ('record_positive_loss_components','record_negative_loss_components','record_positive_gradient_rows','record_negative_gradient_rows','hybrid_attraction_components','hybrid_discrimination_components','hybrid_gradient_rows','hybrid_code_comparisons'):
  vals=[sum(r['counts'][key] for r in events['research'] if r['arm']==arm) for arm in ARMS];lines.append('| '+key.replace('_',' ')+' | '+' | '.join(f'{v:,}' for v in vals)+' |')
 failed=[k for k,v in account['processes'].items() if v['exit_code']!=0]
 lines+=['',f"Whole-process research/verification CPU at report generation: **{account['research_and_verification_cpu_seconds']:,.2f} seconds**. This includes descendants, compilation, repeated controls and failed commands; fitting/subphase CPU is included, not added twice. Failed metered attempts: {', '.join(failed) if failed else 'none'}.",'',
 f"The A6 Python pre-fit tests consume {events['totals']['input_bytes']:,} discarded update bytes; historical trajectories consume {account['historical_correctness_bytes']:,}. Exact data-only cursor traversal verifies all fitting/local positive/negative loss-component and gradient-row counts without recomputing weights. Counts are loss/gradient events, not loss-value sums or scalar floating-point writes. Per-phase, query-presence, applicability, hybrid/code-comparison, cursor and microcheck counts are retained separately. Historical unit/native scalar writes are uninstrumented; their complete CPU and logs are preserved. Setup/editing/read-only inspection is not separately CPU-metered. Artifact generation and packaging have separate receipts.",'',
 '## Resume and next work','',
 'Read `RESUME.md`, `DIAGNOSIS.md` and [A7_REGISTRATION.md](A7_REGISTRATION.md) in full. A7 is registered, not implemented or fitted: replace only continuation training’s existing 0–7-byte record spacers with fixed per-occurrence diverse bytes, preserving lengths, masks, labels, order and budget. Keep A6 model arithmetic and all gates. The ordinary archive includes published source and git history, starting-main source, all 18 fitted model/cursor pairs, generated world manifests, raw traces, source/data/results locks, process receipts and per-member SHA-256. No checkpoint, dataset or compiled binary enters git. Never reset, finish or replay at an arbitrary saved cursor boundary.', '',
 'Failure retains E8 and the closed A6 audit. Passing development would require separately registered fresh generic/capacity qualification: language ≤+0.01 nats/byte, copy/reconstruction ≤1-point token loss, post-1M-language retention, all three-seed 1M/5M and speed gates. Partial improvement cannot override a failed gate.', '',
 'The user explicitly authorizes committing/pushing completed verified work, including failure, directly to main without further confirmation. Preserve unrelated changes and history; never force-push. Deliver downloadable Markdown, primary PNGs, resumable archive and a copy/paste prompt with this authorization.', '',
 'Primary grounding: [Elkan, IJCAI 2001](https://cseweb.ucsd.edu/~elkan/rescale.pdf) analyzes distinct misclassification costs. The factor four is this project’s prospective choice. A5/A4 addressing references and all historical findings remain preserved; no paper establishes A6’s empirical outcome.','']
 (o/'SHoEbox_EXP0305_A6_Report.md').write_text('\n'.join(lines))
 print('Wrote A6 report from locked evidence')
if __name__=='__main__':main()
