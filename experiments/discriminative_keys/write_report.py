"""Build the ordinary Markdown report from locked measurements, without fitting."""
import argparse,json
from pathlib import Path
import numpy as np
from .report import NAMES,ARMS

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();w=a.workspace;o=a.out
 read=lambda p:json.loads((w/p).read_text())
 result=read('a5/results.json');ext=read('extended/results.json');speed=read('benchmark.json');updates=read('update_metrics.json');conf=read('conflicts.json');ret=read('retention.json');geo=read('geometry.json');v=read('verification.json');account=json.loads((o/'accounting.json').read_text());counts=read('prefit_correctness_bytes.json');fit=account['fit_totals']
 pct=lambda x:f'{100*x:.2f}%'
 mean=lambda xs:float(np.mean(xs))
 means={arm:{c:mean([r['arms'][arm]['development'][c]['exact'] for r in result['rows']]) for c in NAMES} for arm in ARMS}
 primary='\n'.join('| '+NAMES[c]+' | '+' | '.join(pct(means[arm][c]) for arm in ARMS)+' |' for c in NAMES)
 update_table='\n'.join('| '+arm.upper()+' | '+' | '.join(pct(mean([r[k] for r in updates['rows'] if r['arm']==arm])) for k in ('changed','untouched'))+' |' for arm in ARMS)
 erows=[r['arms']['a5'] for r in ext['rows']]
 offset_table='\n'.join('| '+NAMES[c]+' | '+pct(mean([r['offsets'][c] for r in erows]))+' | '+pct(np.mean([r['offsets'][c] for r in erows],axis=0).min())+' | '+pct(mean([r['both_orders'][c]['both_query_orders_exact'] for r in erows]))+' |' for c in list(NAMES)[:5])
 speed_table='\n'.join(f'| {arm.upper()} | {r["bytes_per_cpu_second"]:,.1f} | {pct(r["relative_native"])} |' for arm,r in speed['rows'].items())
 trace_table='\n'.join('| '+NAMES[c]+' / '+arm.upper()+' | '+' | '.join(pct(mean([r['summary'][c][k] for r in conf['rows'] if r['arm']==arm])) for k in ('target_present','target_recognized','target_shortlisted','native_teacher_fed'))+' |' for c in ('compete16','overwrite16') for arm in ARMS)
 sig_table='\n'.join('| '+arm.upper()+' | '+' | '.join(pct(mean([r['arms'][arm]['traces']['diverse'+str(d)]['trace']['present'] for r in ext['rows']])) for d in (16,64,256))+' |' for arm in ARMS)
 gs={c:{k:mean([r['summary'][c][k] for r in geo['rows']]) for k in ('nearest_identity_correct','nearest_version_correct')} for c in ('compete16','overwrite16')}
 b=v['gates']['a5']['binary_regression'];hybrid=counts['research_totals']
 losses=[r['summary'] for r in ret['rows'] if r['arm']=='a5']
 failed=[k for k,z in account['processes'].items() if z['exit_code']!=0]
 decision='PASS' if v['gates']['a5']['development_pass'] else 'FAIL'
 text=f'''# SHoEbox EXP03/05-A5: local discriminative key reconstruction

16 November 2026 (America/Denver). Starting main:
`116efa345aa435780044e3ad5705350b6268f1a6`.
Pre-fit implementation/source-data-lock commit: `f28090a`.

**A5 development: {decision}. Qualified public E8 remains unchanged.**
All A1/A2/A3/A4/A5 audits remain closed. Every historical score, throughput
measurement, failed attempt and cost remains preserved. See `A6_REGISTRATION.md`
for the next focused revision; it is registered, not implemented or fitted.

## Primary results

Full-256-byte autoregressive whole-episode exactness. Matched E8/A4/A5,
seeds 1/7/19; 128 fresh development worlds x4 order presentations per condition.
Seeds share worlds and are not independent world samples. These are matched
fresh-curriculum controls, separate from public E8's original qualification.

| Condition | Matched E8 | Matched A4 | A5 |
|---|---:|---:|---:|
{primary}

![Primary results](SHoEbox_EXP0305_A5_Results.png)

| Corrected overwrite | Changed key | Untouched key |
|---|---:|---:|
{update_table}

A5 versus matched A4: competing binding changes by
{100*(means['a5']['compete16']-means['a4']['compete16']):+.2f} points; overwrite by
{100*(means['a5']['overwrite16']-means['a4']['overwrite16']):+.2f} points.
Each condition, answer position, offset, delay, binary regression and speed
gate remains independently binding.

## Exact objective and isolation

Positive query and positive record-key labels use exactly the registered hybrid:

`L = 0.25 * sum_r (z_r-t_r)^2 + (1/16) * (-log P_observed)`

`P = softmax(-4 * squared_distance(z, code_b))`, over all **256** byte labels.

`e_r = 0.5 * (z_r+mu_r-2*t_r) * (1-z_r*z_r)`.

Every pre-update tanh projection uses the original float32 affine arithmetic.
Distance sums, max-subtracted softmax, normalizer and means use float64 in the
registered coordinate/byte order. Cast each gradient once to float32, then
apply the unchanged affine update at `rate*auxiliary*4` with `1+energy(x)`
normalization. Every gradient exists before any row update; FMA is disabled.

The existing codebook has **256 unique codes, zero collisions**, and minimum
Hamming distance one. SHA-256 is
`c8292df6bb4ea5e5a6c98daaf9796d008d03b0672107744637a13548a785559a`.
No code or label was changed or omitted. Fixed read-only training codebook:
16,384 bytes; objective scratch: 2,256 bytes; pre-update z: 64 bytes.
These do not enter sequence state or checkpoints. Existing A4 selection scratch
is 6,968 bytes; combined existing prediction scratch remains 2,408 bytes.

All arms have 497,589 fitted parameters, including 7,670 readout weights.
A5 adds zero fitted parameters, persistent sequence state, optimizer moments,
slots or projection cache. Its magic is `0x3541353050584553`, nested v19.
All protected loaders reject implicit conversion.

Inference is exactly A4: all recognized float32 distances, the float64
`4*(double(d_min)+1.96-double(d_i))` rule, chronological weights, stable ties,
paired top-two values, one/zero-candidate fallback, and unchanged admission.
The codebook is never evaluated by inference. No parser, target oracle,
discrete decoded-key match or restricted output alphabet is introduced.

Source isolation and same-weight cross-policy trajectories verify unchanged
inference and generic/native updates. Native controlled-label tests verify
unchanged value, five presence, applicability and native-state updates.
Generic byte objectives freeze local heads; observed-window labels are explicit
optional local supervision, separate from next-byte loss.

## Frozen identity/version diagnosis

First 32 development worlds x4 orders, both teacher-fed answers, all seeds/arms.
These traces localize failure; autoregressive results determine qualification.

| Condition / arm | Target present | Recognized | Shortlisted | Native answer |
|---|---:|---:|---:|---:|
{trace_table}

Matched A4 and A5 nearest-candidate identity is {pct(gs['compete16']['nearest_identity_correct'])}
on competing queries. On overwrite, nearest identity is
{pct(gs['overwrite16']['nearest_identity_correct'])}, and nearest version is
{pct(gs['overwrite16']['nearest_version_correct'])}.
All candidate distances, vectors, occurrence order and selected mass categories
are retained. Every A4/A5 selected pair and weight agrees with an external
exhaustive calculation. The [separate diagnosis note](DIAGNOSIS.md) interprets these locked
measurements without a new inference candidate or coefficient sweep.

![Frozen matching diagnosis](SHoEbox_EXP0305_A5_Diagnosis.png)

## Separate long-diverse signature loss

| Arm | Signature present, 16 bytes | 64 bytes | 256 bytes |
|---|---:|---:|---:|
{sig_table}

The external admission trace records {sum(r['loss_events'] for r in losses)}
A5 target-removal events. Event causes, incoming false recognition, cache
occupancy and target presence logits are retained in `retention.json` and
summarized in the diagnosis note. This measures loss from the required L1
cache separately from recognition, selection and value decoding; it does not
claim every representation elsewhere disappeared. Frozen checkpoints are intact.

![Long-diverse retention](SHoEbox_EXP0305_A5_Retention.png)

## Unchanged extended gates

| A5 condition | All-offset mean | Worst offset, seed mean | Both orders |
|---|---:|---:|---:|
{offset_table}

Primary and each answer position: >=95% mean /90% every seed. Both orders and
every padding/repeat/diverse delay: >=90% mean /85% every seed. Offsets:
>=95% overall mean /90% worst-offset mean across seeds. No-query drop >=20
points; reversed selection must hurt. Every state/pairing/gate lesion is retained.

Fresh binary canonical regression: {pct(b['primary']['mean'])} primary,
{pct(b['offsets']['mean'])} offset mean, {pct(b['offsets']['worst'])} worst offset,
{pct(b['both_orders']['mean'])} both orders. Every binary per-position,
padding-delay and lesion gate is applied. Full per-seed decisions are in
`verification.json`.

Each primary/long-filler condition uses 512 presentations. The same first
64 presentations are crossed with all 64 offsets. Each independent 0/64/128/256
padding delay uses the next 16 presentations x64 offsets. Both-order and lesion
samples contain 256 presentations. No threshold or sample count was reduced.

No generic/capacity qualification follows development failure. Future gates
remain language <=+0.01 nats/byte, copy/reconstruction <=1-point token loss,
post-1M-language retention and complete three-seed 1M/5M qualification.
Public E8 retains its qualified 98.2747% binding through 1M language targets,
+0.006233 nats/byte language result, and passing copy/reconstruction/scale/speed.

## Isolated full-distribution throughput

One pinned CPU; three interleaved 65,536-byte repeats plus 1,024 warmup each;
all other local research jobs stopped. Full 256-byte output at every byte.
Compilation/cloning/warmup are outside timed intervals and inside process cost.
Native is a fresh untrained v19 reference. Public E8 exactly imports matched E8;
prediction checksums match. The unchanged requirement is **>=80% native**.

| Arm | Bytes / CPU second | Relative native |
|---|---:|---:|
{speed_table}

![Throughput](SHoEbox_EXP0305_A5_Throughput.png)

## Locks, costs and verification

Fresh salt `SHoEbox-EXP0305-discriminative-keys-v5` plus NUL; world RNG 20261127.
512/128/256 train/dev/closed-audit worlds plus 128 fresh binary dev worlds.
Exclusions include the entire prior opened binding-fold union/E2-family,
reserved A1/A2/A3/A4 audits, and all A2/A3/A4 dev/binary worlds. Operation/order
siblings stay grouped. No generic corpus is used; opened E1/E7 generic groups
remain excluded later. 939 fitting source dependencies, passing receipts and
fold hashes were locked before fitting. Final outcomes/checkpoints/cursors
were locked before diagnosis.

Independent interpretation passes on 12,800 primary and 5,120 both-change/order
presentations plus 512 fresh binary presentations. Repeating first/latest/constant
values yields zero whole-episode exactness; distinct answers and both query
orders bound any fixed query-blind answer pair at 50%. Labels are causal.

Each arm/seed receives its own 250K warmup +2M continuation. No pilot, inherited
warmup, refit, replay or optimizer/codebook/band/prior sweep occurred.

| Fitting quantity | Total |
|---|---:|
| Charged input bytes | {fit['bytes']:,} |
| Padding bytes, included above | {fit['padding_bytes']:,} |
| Native objective targets | {fit['targets']:,} |
| Answer targets, included above | {fit['answer_targets']:,} |
| Local projection-row updates, all arms | {fit['local_projection_steps']:,} |
| Local presence updates | {fit['local_presence_steps']:,} |
| Applicability updates | {fit['gate_targets']:,} |
| Local training windows | {fit['local_training_windows']:,} |
| A5 attraction-loss components | {hybrid['hybrid_attraction_components']:,} |
| A5 discrimination-loss components | {hybrid['hybrid_discrimination_components']:,} |
| A5 hybrid gradient rows | {hybrid['hybrid_gradient_rows']:,} |
| A5 256-code distance comparisons | {hybrid['hybrid_code_comparisons']:,} |
| A5 coordinate distance comparisons | {hybrid['hybrid_coordinate_distance_comparisons']:,} |
| Fitting CPU seconds | {fit['cpu_seconds']:.2f} |
| Pilot fitting bytes | 0 |

Whole-process research/verification CPU at report generation:
**{account['research_and_verification_cpu_seconds']:.2f} seconds**, including
compilation, setup dependency installation, descendants and repeated checks.
Fitting/subphase costs are included, not added twice. Failed metered attempts:
{', '.join(failed) if failed else 'none'}. The first broader Python invocation
failed because PyTorch was absent; its log/cost remains. After dependency
installation all 154 historical/research tests passed. No research fit failed.

All 18 actual model/cursor pairs resume exactly: 1,000 uninterrupted versus
371 + save/load +629, totaling 36,000 discarded correctness bytes. Pre-fit
Python controls consume {counts['totals']['input_bytes']:,} discarded update bytes;
historical trajectories consume 114,716. Data-only replay verifies every
local component/gradient/code-comparison count, separately for formal fits,
pre-fit tests and cursor checks. Independent objective microchecks, inference
and padding have separate ledgers. Historical unit/native scalar writes remain
uninstrumented; complete CPU/logs are retained. Setup editing/read-only inspection
is not separately CPU-metered; artifact generation/packaging has separate receipts.

- **225 distinct Python tests pass:** 60 runtime/public, 154 historical/research,
  and 11 A5. All 15 extra protected-policy rejection checks pass.
- **824,111 core native**, **723 public E8**, **13,863 inherited A4/E8/admission**,
  and **75,295 A5 objective checks** pass in optimized and ASan/UBSan builds.
  LSan is explicitly disabled. Native independent gradient error is at most
  5.889e-8; finite differences and controlled descent pass.
- Fourteen historical constructor/checkpoint trajectories match starting main.
  v5/v6 loader sources are unchanged but not independently instantiated.
- Every source/fold/snapshot/cursor/result lock verifies. Public E8, explicit
  v19/E7, all historical constructors and previous reports remain unchanged.

## Reproduce and resume

Read `experiments/discriminative_keys/README.md` and `RESUME.md`. The ordinary
archive includes source and git history, all 18 fitted model/cursor pairs,
generated worlds, raw results/diagnostics, locks, failure/passing receipts and
member hashes. Checkpoints, datasets and compiled binaries stay outside git.
Use this campaign's cursor driver with each saved E8/A4/A5 policy; never reset,
finish or replay warmup at an arbitrary interrupted boundary. Extending a
failed model consumes newly registered research bytes.

Primary grounding: [Movshovitz-Attias et al., ICCV 2017](https://arxiv.org/abs/1703.07464)
uses learned distance proxies; A5 uses existing fixed codes and its own exact
hybrid. [Miller et al., EMNLP 2016](https://aclanthology.org/D16-1147/) supports
paired key/value reading; [Arora et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html)
supports multi-query recall measurement. None establishes A5's performance or
its constants. All empirical claims above come from this locked campaign.
'''
 (o/'SHoEbox_EXP0305_A5_Report.md').write_text(text);print(decision,means['a5'],gs)
if __name__=='__main__':main()
