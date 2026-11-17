"""Render completed A9-F1 evidence without running a model."""
import argparse,json
from pathlib import Path
import numpy as np
from .report import ARMS,NAMES


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();w=a.workspace;o=a.out
    read=lambda n:json.loads((w/n).read_text())
    summary=json.loads((o/'summary.json').read_text());account=json.loads((o/'accounting.json').read_text())
    gates=read('verification.json')['gates'];g=gates['a9'];fit=read('a9/results.json')
    speed=read('benchmark.json');updates=read('update_metrics.json');events=read('prefit_correctness_bytes.json')
    conf=read('conflicts.json');ret=read('retention.json');sur=read('survivors.json');sigs=read('signature_traces.json');cov=read('coverage.json')
    allocation=read('f1_allocation.json');streams=read('streams.json');means=summary['means'];f=account['fit_totals']
    pct=lambda x:f'{100*x:.6f}%';mean=lambda xs:float(np.mean(xs))
    lines=[]
    def add(s=''):lines.extend(s.split('\n'))
    def table(headers,rows):
        add('| '+' | '.join(headers)+' |');add('|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|')
        for row in rows:add('| '+' | '.join(map(str,row))+' |')
        add()
    status='PASS' if g['development_pass'] else 'FAIL'
    add('# SHoEbox EXP03/05-A9-F1: selective record-key credit\n')
    add(f"16 November 2026 (America/Denver). Starting main: `fa54cbd1c8c90be756ea4da8cfef2ccb735ed183`. Pre-fit lock: `{read('a9/plan.json')['prefit_commit']}`.\n")
    add('The local pre-fit commit preceded fitting. Its identical tree was published as `7dc8b5c275d00b469f068d788064b63025833120` after terminal Git credentials were unavailable; both commits and the exact tree mapping are preserved.\n')
    add(f'**A9 development: {status}. Qualified public E8 remains unchanged.** The complete fixed comparison consumed exactly **20,250,000 fitting bytes**. All A1–A9 audits remain closed. No pilot, sweep, replay, inherited warmup, extra fit or generic/capacity qualification ran.\n')
    add('## Binary allocation repair\n')
    add('F1 changes only binary development allocation. Independent enumeration and seeded shuffle select 128 unique worlds from 336 eligible candidates, excluding all historical groups and all 896 primary worlds. Every primary identity and all six original stream/credit manifests are byte-identical to blocked A9. The original 96-world failure and all its receipts and costs remain preserved. Binary worlds cannot enter training or audit allocation.\n')
    table(['Lock','SHA-256'],[(name,digest) for name,digest in allocation['primary_ordered_sha256'].items()]+[('Binary ordered sample',allocation['binary_ordered_sha256']),('Complete original stream manifest',allocation['streams_sha256'])])
    add('The raw world file and full stream/credit ledgers are in the archive. The committed pre-fit receipt pins their hashes, exact selected identity hashes and order, source closure, independent proof, and passing controls before fitting.\n')
    add('## Full autoregressive outcomes\n')
    add('Native full-256-byte whole-episode exactness; matched E8/A8/A9, seeds 1/7/19, 128 fresh primary worlds ×4 record/query-order presentations per condition. Generated first answers enter the second-query state. Seeds share worlds. These fresh controls do not replace earlier E8 or A8 results.\n')
    table(['Condition','Matched E8','Matched A8','A9'],[[NAMES[c]]+[pct(means[arm][c]) for arm in ARMS] for c in NAMES])
    add('![Primary results](SHoEbox_EXP0305_A9_Results.png)\n')
    table(['Overwrite answer','E8','A8','A9'],[[label]+[pct(mean([r[key] for r in updates['rows'] if r['arm']==arm])) for arm in ARMS] for key,label in [('changed','Changed key'),('untouched','Untouched key')]])
    add('## Exact selective-credit intervention\n')
    add('A8 and A9 receive identical inputs, targets, six local labels, answer masks and alternating 64-episode PAD/diverse blocks. A9 alone suppresses positive record-key hybrid calls in diverse continuation blocks. Presence, value, query-key, applicability and native rules are unchanged. Each arm/seed has an independent 250K native-only PAD warmup and 2M continuation.\n')
    add('The flag is ephemeral training input derived from the exact episode cursor, separate from labels and masks. It never enters inference or model state. Generic calls cannot train local heads. Enabled credit is byte-identical to A8 after reversing only the envelope identity; isolated disabled-credit checks change only eligible record-key rows and counters. Omitted events perform no hidden hybrid gradients or code comparisons. FMA remains disabled.\n')
    add('A9 magic is `0x3941353050584553`, nested v19; cursor schema is `exp0305-a9-cursor-1`. Every arm has 497,589 fitted parameters, including 7,670 readout weights. No new slots, sequence state, optimizer moments, cache, decoder, codebook or inference parser was added.\n')
    continuation={r['arm']:r for r in streams['rows'] if r['phase']=='continuation'}
    table(['Per 2M continuation','E8','A8','A9'],[[label]+[f"{continuation[arm]['counts'][key]:,}" for arm in ARMS] for key,label in [('positive_record_keys','Eligible record-key events'),('record_key_executed','Executed record-key events'),('record_key_omitted','Omitted record-key events'),('local_projection_steps','Executed total projection rows')]])
    add('These are executed counts, reconciled against all fitted model inventories and exact cursors. The incomplete terminal episode is retained; there is no rebalancing or compensation for omitted credit.\n')
    add('## Every registered gate\n')
    table(['A9 condition','Offset mean','Worst-offset mean','Both orders'],[[NAMES[c],pct(g[c]['offsets']['mean']),pct(g[c]['offsets']['worst']),pct(g[c]['both_orders']['mean'])] for c in list(NAMES)[:5]])
    table(['Independent padding delay','Mean','Minimum seed'],[[d,pct(v['mean']),pct(v['minimum'])] for d,v in g['padding'].items()])
    b=g['binary_regression']
    table(['Fresh binary regression','Result'],[('Primary mean',pct(b['primary']['mean'])),('Primary minimum seed',pct(b['primary']['minimum'])),('Offset mean',pct(b['offsets']['mean'])),('Worst-offset mean',pct(b['offsets']['worst'])),('Both orders',pct(b['both_orders']['mean']))])
    add('Primary and each answer position require ≥95% mean/90% each seed. Both query orders and every delay require ≥90%/85%. Offsets require ≥95% overall mean and ≥90% worst-offset mean across seeds. No-query drop must be ≥20 points and reversed selection must hurt. All state/pairing/gate lesions and all binary position/padding/lesion gates were evaluated; complete individual decisions are in `verification.json`.\n')
    add('Each offset uses the same first 64 presentations. Each independent 0/64/128/256 padding delay uses the next 16 presentations ×64 offsets. Both-order/lesion samples have 256 presentations. Primary and long-filler conditions have 512. No sample or threshold was reduced.\n')
    add('## Isolated throughput\n')
    table(['Arm','Bytes / CPU second','Relative native'],[[arm.upper(),f"{r['bytes_per_cpu_second']:,.1f}",pct(r['relative_native'])] for arm,r in speed['rows'].items()])
    add('One pinned CPU; three interleaved 65,536-byte repeats per arm plus 1,024-byte warmups, with all other local research jobs finished. Full 256-byte output is produced at every input. Compilation, cloning and warmup are excluded from timed intervals and included in process costs. Native is an untrained full-budget v19 timing reference. Public/research E8 checksums agree exactly. The unchanged candidate gate is ≥80% native. Raw times/checksums are retained without result replacement.\n')
    add('![Throughput](SHoEbox_EXP0305_A9_Throughput.png)\n')
    add('## Frozen subset diagnosis\n')
    add('All fits, 18 model/cursor pairs and every full outcome, including timing, were locked before diagnosis. The following uses only the first 32 worlds ×4 orders, both teacher-fed answers and all seeds. These per-answer subset measurements do not replace full autoregressive outcomes.\n')
    diag=[]
    for arm in ('a8','a9'):
        rr=[r['summary'] for r in ret['rows'] if r['arm']==arm];ss=[r['summary'] for r in sur['rows'] if r['arm']==arm];cc=[r['summary'] for r in conf['rows'] if r['arm']==arm]
        diag.append([arm.upper(),pct(mean([r['arms'][arm]['diverse256']['trace']['present'] for r in sigs['rows']])),pct(sum(r['full_diverse_recognized'] for r in rr)/sum(r['full_diverse_pairs'] for r in rr)),sum(r['loss_events'] for r in rr),pct(mean([r['survivor_accuracy'] for r in ss])),pct(mean([r['compete16']['nearest_identity_correct'] for r in cc])),pct(mean([r['overwrite16']['predicted_old_value'] for r in cc]))])
    table(['Arm','Diverse256 signature survival','False recognition','Target removals','Survivor answer accuracy','Competing nearest identity','Changed-key old predictions'],diag)
    table(['A9 autoregressive episode exactness','First 32 worlds','Other 96','All 128'],[[NAMES[c]]+[pct(r[k]) for k in ('first_32_worlds','remaining_96_worlds','all_128_worlds')] for c,r in cov['primary_subsets']['a9'].items()])
    add('Matched A8/A9 geometry, query/record reconstruction errors, latest/old/unrelated/filler mass, target recognition, shortlist membership, true-record false negatives, signature survival at 16/64/256 bytes, and externally reconstructed admission/selection remain in raw and compact evidence. Data-only window coverage and credit counts retain block, condition, key and phase. See `DIAGNOSIS.md` for the measured interpretation and sole next preregistration.\n')
    add('![Frozen diagnosis](SHoEbox_EXP0305_A9_Diagnosis.png)\n\n![Signature survival](SHoEbox_EXP0305_A9_Retention.png)\n\n![Presence and survivors](SHoEbox_EXP0305_A9_Presence.png)\n')
    add('## Accounting and verification\n')
    table(['Fitting quantity','Total'],[[k.replace('_',' '),f'{f[k]:,}'] for k in ('bytes','padding_bytes','nuisance_bytes','replaced_padding_bytes','targets','answer_targets','local_projection_steps','local_presence_steps','gate_targets','local_training_windows')]+[['Fitting CPU seconds',f"{f['cpu_seconds']:,.3f}"]])
    add('PAD, nuisance and replaced-PAD bytes are included in fitting bytes, not added to them. Replaced PAD is a subset of nuisance input.\n')
    keys=('record_positive_gradient_rows','record_negative_gradient_rows','hybrid_attraction_components','hybrid_discrimination_components','hybrid_gradient_rows','hybrid_code_comparisons','hybrid_coordinate_distance_comparisons','omitted_hybrid_components','omitted_hybrid_code_comparisons','omitted_hybrid_coordinate_comparisons')
    table(['Across three continuations','E8','A8','A9'],[[k.replace('_',' ')]+[f"{sum(r['counts'][k] for r in events['research'] if r['arm']==arm):,}" for arm in ARMS] for k in keys])
    failed=[k for k,v in account['processes'].items() if v['exit_code']]
    add(f"Whole-process research/verification CPU at report creation: **{account['research_and_verification_cpu_seconds']:,.3f} seconds**. Fitting and other nested subphase timers are included once. Failed metered attempts: {', '.join(failed) if failed else 'none'}. Artifact generation, publication and packaging have separate receipts.\n")
    add(f"The original blocked A9 report's **1,025.614 CPU seconds** remains unchanged. Its complete preserved process receipts, including later delivery work, are separately retained in `prior_a9/`; no prior cost is reset or attributed to this new fit.\n")
    add('During report preparation, an initial syntax check found a duplicate keyword introduced by arm-name replacement. It executed no module, model, fit or inference. The failed source and correction receipt are preserved; that initial editor command was not separately CPU-metered. The failed terminal push changed no remote ref and is also retained.\n')
    add(f"Current A9 Python controls consumed **{events['totals']['input_bytes']:,} discarded update bytes**. All 18 actual saved pairs pass 1,000 uninterrupted versus 371 + save/load +629 updates, totaling **36,000 discarded correctness bytes**. Historical trajectory checks consumed **{account['historical_correctness_bytes']:,}**. Executed/omitted local events and native masked-reference comparisons have separate ledgers. Historical/native microtrajectory scalar writes remain uninstrumented; full CPU/logs are retained. Setup, editing and read-only inspection are not separately metered.\n")
    add('- 289 distinct Python tests pass: 60 public/runtime, 209 historical research and the complete 20-test A9 suite; 23 additional protected-policy rejection checks pass.\n- Optimized and ASan/UBSan controls pass: 824,111 core, 723 public E8, 13,863 inherited selection/lifecycle, 75,295 hybrid, 7,826 presence and 57,280 masked-credit checks per build. Leak checking was disabled.\n- Independent gradients, finite differences, all 256 unique codes, pre-update coupling, scalar A8 parity, causal labels, training-only flag isolation, 64-phase block boundaries and exact partial-spacer resumes pass.\n- Fourteen historical constructor/checkpoint trajectories match starting main. v5/v6 loaders remain unchanged but were not independently instantiated. Public/historical numerical sources and all A1–A8 results remain untouched.\n')
    add('## Resume and next work\n')
    add('The sole next registration is `A10_REGISTRATION.md`: exact row-parallel affine execution, with **zero research fitting bytes**. It requires bitwise scalar equivalence and ≥80% native throughput on fixed fresh timing worlds. A10 is not implemented or measured. A speed pass would leave A9\'s accuracy failure intact; further accuracy work requires a separate registration.\n')
    add('The archive contains published source and Git history, starting-main source, all 18 fitted model/cursor pairs, exact world/stream/credit manifests, raw outcomes/diagnostics, every failed/passing receipt, original blocked A9 evidence and member hashes. Checkpoints, datasets and compiled binaries stay outside Git. `RESUME.md` explains verification and exact cursor use. The completed fitting budget is exhausted; no checkpoint extension is authorized without a new registration.\n')
    add('Failure retains E8 and closed A1–A9 audits and leads to one focused preregistration, not an additional fit. Passing development would require separate fresh generic/capacity qualification with language ≤+0.01 nats/byte, copy/reconstruction ≤1-point token loss, post-1M-language retention, all three-seed 1M/5M checks and every original speed gate.\n')
    add('Completed verified work, including unsuccessful research, is explicitly authorized for direct main commit/push without confirmation. Preserve unrelated changes and remote history; never force-push. Deliver ordinary downloadable Markdown, primary PNGs, resumable archive and a copy/paste prompt preserving this authority. No Pages or Spaces.\n')
    (o/'SHoEbox_EXP0305_A9_F1_Report.md').write_text('\n'.join(lines))
    print('Wrote completed A9-F1 report from locked evidence')


if __name__=='__main__':main()
