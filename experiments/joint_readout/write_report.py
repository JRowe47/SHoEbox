"""Render the measured EXP01-E campaign as ordinary Markdown."""
import argparse,json
from pathlib import Path
import numpy as np
def read(p):return json.loads(p.read_text())
def pct(x):return f'{100*x:.2f}%'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();w=a.workspace;o=a.out;s=read(o/'summary.json');g=s['binding']['formal']['gated_all'];ledger=read(o/'accounting.json');t=ledger['total']
 arms=table(['500K arm','Native exact mean','Seed 1 /7 /19','All-offset mean'],[[k,pct(v['primary_mean']),' / '.join(map(pct,v['primary'])),pct(v['offset_mean'])] for k,v in s['binding']['formal'].items()])
 capacity=table(['Nominal','Actual native /joint parameters','Joint exact','Offset mean /worst','Relative inference'],[[c,' / '.join(str(read(w/f'exp01e-{c}/train.json')['rows'][0]['arms'][k]['inventory']['parameters']) for k in ['native_all','gated_all']),pct(s['binding'][c]['gated_all']['primary_mean']),pct(s['binding'][c]['gated_all']['offset_mean'])+' / '+pct(s['binding'][c]['gated_all']['worst_offset']),pct(s['throughput'][c])] for c in ['formal','1m','5m']])
 costs=table(['Campaign','Charged bytes','PAD bytes','Fit CPU seconds'],[[k,f"{r['bytes']:,}",f"{r['padding_bytes']:,}",f"{r['cpu_seconds']:.2f}"] for k,r in ledger['campaigns'].items()])
 reg=table(['E1 held-out audit','Native','D3 joint','E1 gated','Gated minus native'],[[task]+[f"{s['generic'][task][arm]['nll']:.6f}" if task=='language' else pct(s['generic'][task][arm]['token_accuracy']) for arm in ['native','d3_joint','gated']]+[f"{s['generic'][task]['gated']['nll']-s['generic'][task]['native']['nll']:+.6f} nats" if task=='language' else f"{100*(s['generic'][task]['gated']['token_accuracy']-s['generic'][task]['native']['token_accuracy']):+.3f} pp"] for task in ['language','selective_copy','compression']])
 e7=read(w/'exp01e7-regressions/train.json')['rows'];e7rows=[]
 for task in ['language','selective_copy','compression']:
  vals=[float(np.mean([r['curve'][-1]['dev_nll'] if task=='language' else r['curve'][-1]['dev']['token_accuracy'] for r in e7 if r['task']==task and r['arm']==arm])) for arm in ['native','d3_joint','local_gate']]
  e7rows.append([task]+([f'{v:.6f}' for v in vals] if task=='language' else list(map(pct,vals)))+[f'{vals[2]-vals[0]:+.6f} nats' if task=='language' else f'{100*(vals[2]-vals[0]):+.3f} pp'])
 e7table=table(['E7 fresh development','Native','D3 joint','E7 request gate','E7 minus native'],e7rows)
 speed=read(w/'exp01e-local-gate-benchmark.json')['rows']['E7']['relative']
 text=f'''# SHoEbox EXP01-E: joint readout and applicability qualification

16 November 2026. Starting main: `428fb0b0272d81b832a8a10263b44218b23cff3d`.
Production remains minimal corrected-C v19. No candidate is promoted.

## Findings

The E1 joint residual readout reaches **98.83% native two-answer exactness**
across three seeds, within its nominal 500K budget, at **86.76% native inference
throughput**. Its fresh all-offset mean is **94.11%, below the unchanged 95%
gate**, and binding falls to zero after generic language continuation. Frozen
D3 also scores 94.14% across offsets on this fresh fold. The earlier D3 audit
remains valid on its own registered worlds; it does not guarantee every new
world sample passes. None of these audits was used to tune a revision.

Seven preserved revisions isolate two different failures: freely fitted output
calibration drifts, then generic-loss applicability suppresses a still-correct
readout. **E7 fixes retention in a seed-7 development pilot**: 99.41% native
exactness before and after 1M language targets, 98.05% offset mean, 95.31% worst
offset, 99.22% both query orders, and 100% at all registered padding delays.
It uses explicit observed-request local supervision. Its fresh three-seed
generic qualification fails, so this is not a production or general-language
result. E8's focused cold-start gate revision is registered but not fitted.

## Design and supervision

All inference is causal, online, bounded, and uses the existing constituent
state. The readout preserves each key/value pair, selects the latest recognized
query, and mixes values only after addressing. No hand-coded inference parser,
oracle source index, external exact store, new slots or history tape is added.
The stable constituent features are fixed byte-derived channels; successful
readout training does not prove learned upper-summary specialization.

E1 jointly teaches native core, residual output and a 65-weight sigmoid using
combined byte cross-entropy. Discrete source selection and key/value decoding
also receive **fixture-specific observed-window labels**. The local projections
are not trained end-to-end through the discrete top-two selector. The no-local
control measures this dependence. Generic language/copy/reconstruction fits
construct no such labels. E6/E7 freeze local applicability when labels are absent.
Their classifier is learned from fixture supervision, not generic byte loss.

Primary literature retained in code and registrations:

- Miller et al., [Key-Value Memory Networks, EMNLP 2016](https://aclanthology.org/D16-1147/): paired addressing and value representations.
- See et al., [Pointer-Generator Networks, ACL 2017 §2.2](https://aclanthology.org/P17-1099/): learned generation/retrieval switching. These experiments mix byte logits or distributions, not source pointers.
- Arora et al., [Simple linear attention language models balance recall-throughput tradeoffs, ICML 2024](https://proceedings.mlr.press/v235/arora24a.html): measure recall alongside throughput/state.
- Jaderberg et al., [Decoupled Neural Interfaces, ICML 2017](https://proceedings.mlr.press/v70/jaderberg17a.html): local credit context; no synthetic-gradient model is implemented here.

## Registered comparisons and data integrity

E1 compares six arms at 500K, seeds 1/7/19. Each receives its own 250K native
warmup plus 2M continuation bytes. Episode order, 0–7-byte record spacing,
balanced phases and budgets are matched; long delays are not trained. Answer
masks and every-byte objectives are explicitly separate controls. Final
snapshots and source hashes are locked before the audit, with no audit-based
selection. Full 256-byte autoregressive answers feed subsequent queries.

E1 has fresh grouped 2,048 train /512 development /1,024 audit presentations,
excluding opened D2/D3 audit worlds. E2–E7 use another grouped split excluding
D2/D3/E1 audit worlds. Their binding audit remains **unopened**. Sibling query/
record orders stay together. The E1 audit reuses worlds across seeds; seeds are
not independent task samples. Offset checks use the same 64 presentations at
each phase; each delay uses the next 16 at all 64 phases, preserving D3 sizes.

Generic data has document-grouped FineWeb splits, UTF-8, document resets and
pinned document/content/byte hashes. E7 uses new grouped generic folds excluding
all opened E1 audit groups. Its development failure stops the new audit, which
remains unopened. All original gates and margins remain unchanged.

## E1 audit and ablations

{arms}

The primary mean threshold is 95%, every seed 90%; the same thresholds apply
to each answer position. E1 first-answer scores are {' / '.join(map(pct,g['first']))};
second-answer scores are {' / '.join(map(pct,g['second']))}. Both-query-order
exactness averages {pct(float(np.mean(g['counterfactual'])))} (required 90% mean,
85% each seed). The worst offset averaged across seeds is {pct(g['worst_offset'])}
(required 90%), but the all-offset mean fails 95%. Every seed is 100% at
independent +0/+64/+128/+256 delays.

On the matched 256-presentation intervention sample, unmodified exactness is
{pct(g['lesion_unmodified'])}; no-query {pct(g['lesions']['no_query'])},
no-state {pct(g['lesions']['no_state'])}, no-head {pct(g['lesions']['no_head'])},
uniform addressing {pct(g['lesions']['uniform'])}, reversed addressing 0%,
and unpaired values 0%. The no-query drop exceeds 20 percentage points and
reverse strictly hurts. Full raw per-seed/offset/delay results are preserved.

![Binding and boundary audit](SHoEbox_EXP01E_Binding.png)

## Generic regressions, retention and capacity

The independent E1 generic fits pass the registered mean margins: language
NLL increase at most 0.01 nats/byte; copy/reconstruction token-accuracy decline
at most one percentage point. These independent fits do not establish retained
binding. Continuing each E1 binding checkpoint on 1M generic targets produces
0% binding exactness in every seed, already by 250K targets.

{reg}

Reconstruction uses its separately budgeted final-L3 MLP, not native language
answers. Full-answer exactness and zero-bottleneck controls are in raw evidence.

{capacity}

Here `formal` means nominal 500K. All three capacities pass primary binding,
query-order and padding-delay gates, but all fail the 95% all-offset mean.
Capacity does not repair the measured boundary-generalization failure. The
1M/5M results qualify only the E1 candidate, not later pilot revisions.

![Learning by bytes and CPU time](SHoEbox_EXP01E_Learning.png)

## Preserved revisions and measured diagnosis

| Revision | Experiment | Development outcome |
| --- | --- | --- |
| E1 | Joint residual logits + generic-loss gate | Binding succeeds; fresh offset audit and language retention fail |
| E2 | Independent absolute logits with learned switch | Fresh binding 99.41%; after language 0%; full-alphabet calibration drifts |
| E3 | Exact probability mixture, transferred E2 weights | 36.52% before language, 0% after; conditional from-scratch fit not triggered |
| E4 | Calibrated eight-bit value distribution | 99.41% before, 0% after; forcing gate on restores 99.41% |
| E5 | Add three query/record-presence gate features | 99.41% before, 0% after; features do not prevent gate suppression |
| E6 | Locally supervise current query presence | Both fresh arms 50% before, 0% after; label targets the query rather than delayed answer request |
| E7 | Locally supervise observed answer-request marker | 99.41% before and after language; generic development qualification fails |

E3 is a failed **transferred-policy** pilot, not evidence against a mixture
trained from scratch. E4 removes 2,295 fitted output weights; generic value-bit
calibration leaves 7,670 fitted readout/gate weights. After E4 language training,
local query/value diagnostics and forced-gate native NLL are unchanged. Natural
answer gating falls from 0.52071 to 0.15078; restoring only the old intercept does
not fix it. This identifies gate plasticity in addition to the earlier output
calibration drift.

E6's gate equals its trained query recognizer exactly, but at a delayed answer
the observed prefix is padding plus ANSWER, not QUERY plus key. Its supplied
target is negative there and answer gating is 0.0000676. E7's sixth training-only
label marks the already observed request byte, without inspecting the answer
mask or future target. Prefix causality and identical generic-control trajectories
are checked. E7 preserves every development binding gate before and after
language; the new generic screen below prevents promotion.

{e7table}

E7 language worsens by 0.519175 nats/byte and copy declines 1.042 percentage
points, exceeding both margins. Its untrained, frozen 0.5 gate assigns half
the distribution to an untrained readout. Bypassing that gate after fitting
still gives 2.504123 mean development NLL: the mixture also weakened native
learning, so post-fit output rescaling alone is insufficient.
E8 registers a fixed cold prior of
1/256, then repeats development and full qualification; it also requires
measuring generic-text activation after local binding training. No E8 fit or
passing result is claimed. E7's conditional formal binding/1M/5M runners are
provided but were not launched after this prerequisite failed.

## Speed, parameters and state

E1 has 9,965 fitted adapter parameters, 499,884 actual combined at nominal 500K.
Frozen/joint D3 has 9,900 and 499,819. Native has 498,367. E4/E6/E7 use 7,670
adapter parameters and 497,589 combined; E5 uses 7,673 and 497,592. These small
undershoots reflect indivisible expert banks, not hidden extra parameters.
Additional persistent sequence state and optimizer moments are zero. The native
stream/retained structures reported by the adapter occupy 289,152 bytes; this
is **not** the complete native allocation. E1 adapter prediction scratch is
2,368 bytes; E7 is 2,408. The comparison wrapper reserves unused head storage
for the native control; allocated and fitted counts are separately recorded.

Isolated pinned-CPU timing uses all 256 outputs at every byte, 65,536 bytes,
three interleaved repetitions, with compilation/cloning excluded. E1 retains
86.76% native speed at 500K, 90.53% at 1M, 95.50% at 5M. E2 retains 87.15%;
E3/E4/E5 prototypes retain 73.88%/79.57%/79.40%, failing the 80% threshold.
E6/E7 cache eight repeated bit softplus values with exact numerical parity;
E7's isolated retained-pilot ratio is **{pct(speed)}**. This timing pass alone
does not qualify the failed generic candidate.

![Retention and isolated inference](SHoEbox_EXP01E_Retention_and_Speed.png)

## Complete fitting ledger and verification

**{t['bytes']:,} charged bytes**, including **{t['padding_bytes']:,} padding bytes**,
across {t['fits']} recorded fitting phases (warmups counted separately), use
**{t['cpu_seconds']:.2f} fitting CPU seconds**. Frozen evaluations, compilation,
checkpoint I/O and discarded correctness-test updates are outside that timer;
no unmeasured total experiment CPU time is implied. Parallel jobs' CPU seconds
are summed, not confused with wall time. All pilots and failed continuations
are included. Transferred weights are charged once at their original fit.

{costs}

Counters record {t['local_projection_steps']:,} local projection-vector steps,
{t['local_presence_steps']:,} presence-classifier steps, {t['gate_targets']:,}
applicability steps, and {t['decoder_updates']:,} reconstruction-MLP steps.
These are objective/vector events, not counts of scalar coefficient writes.
Internal native auxiliary work is included in measured CPU/bytes; its individual
scalar updates are not instrumented. Formal binding worker peaks are about
58 MiB (500K), 80 MiB (1M), and 206 MiB (5M), including Python/evaluation;
other phases did not record peak RSS. See `accounting.json` for every fit,
target count, actual parameter/state inventory and exact padding accounting.

All **160 Python tests** pass (48 runtime, 76 existing research/E1, 6 E2, 7 E3,
4 E4, 5 E5, 6 E6, 8 E7). New native suites pass **2,860 checks per optimized
and ASan/UBSan build**, alongside the existing native suite. Tests cover
normalized full-byte probabilities, gate derivatives, observed-prefix causality,
frozen local heads/gates, all partial phases, checkpoint-policy separation and
exact continuation. Eight actual trained checkpoint/cursor pairs also pass
1,000 uninterrupted updates versus 371 + save/load +629 updates byte-for-byte.
Those discarded correctness updates are not selected research fits.

Production runtime and historical D1/D2/D3 source and checkpoint formats remain
unchanged. Each E policy has its own magic. No dataset, trained model, compiled
binary or temporary snapshot is committed. Downloaded evidence includes selected
resumable checkpoints, both RNG states, remaining shuffled order, partial episode
positions, document boundaries and corpus hashes, plus a source/evidence manifest.

## Continue

Read `AMENDMENT_E8.md`, then execute that fixed cold-start revision. Preserve
the original thresholds and all unopened folds. Promote only after complete
three-seed binding/retention, generic, 1M/5M and engineering qualifications pass.
Direct verified main publication is already authorized, including unsuccessful
research results; preserve remote history and never force-push.
'''
 (o/'SHoEbox_EXP01E_Report.md').write_text(text);print('report written')
if __name__=='__main__':main()
