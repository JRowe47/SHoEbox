"""Report an infeasible preregistration without creating model outcomes."""
import json,sys,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import write,sha,ROOT,sources
from .credit import augmented

def main():
 w=Path(sys.argv[1]);out=ROOT/'reports/exp0305a9';ev=out/'evidence';ev.mkdir(exist_ok=True)
 read=lambda n:json.loads((w/n).read_text())
 feasibility=read('fold_feasibility.json');stream=read('streams.json');counts=read('python_counts_failed1.json')
 assert not feasibility['passed'] and counts['tests']==20 and not counts['passed']
 assert read('prefit_native_final_process.json')['exit_code']==0
 assert not (w/'prefit.json').exists() and not (w/'a9').exists()
 rows={r['arm']:r for r in stream['rows'] if r['phase']=='continuation'}
 correctness={p:augmented(v,p) for p,v in counts['updates'].items()}
 processes={p.stem:json.loads(p.read_text()) for p in w.glob('*_process.json') if '_artifact_' not in p.name}
 accounting=dict(research_fitting_bytes=0,planned_unexecuted_fitting_bytes=20250000,pilots=0,inherited_warmups=0,replay=0,
   successful_python_control_updates=correctness,successful_python_control_bytes=sum(v['input_bytes'] for v in correctness.values()),
   python_control_observe_calls=counts['observed_bytes'],python_control_explicit_observed_bytes=sum(counts['observed_bytes'].values()),rejected_python_calls=counts['rejected_calls'],
   historical_discarded_correctness_bytes=read('history.json')['correctness_only_discarded_input_bytes'],
   data_only_stream_bytes=sum(r['counts']['input_bytes'] for r in stream['rows']),data_only_streams=6,
   planned_per_continuation={a:augmented(r['counts'],a) for a,r in rows.items()},
   processes=processes,cpu_seconds=sum(x['cpu_seconds'] for x in processes.values()),
   native_mask_checks_per_build=dict(checks=57280,local_calls=1280,actual_hybrid_calls=512,reference_hybrid_calls=768,record_hybrid_executed=256,record_hybrid_omitted=256,actual_hybrid_gradient_rows=8192,reference_hybrid_gradient_rows=12288,actual_code_comparisons=131072,reference_code_comparisons=196608,builds=2),
   accounting_scope='Research fit/evaluation/throughput/diagnosis: zero. Data-only traversal is not fitting. Python run/observe events counted directly; full native calls and inherited objective/presence counts retained in logs. Historical/native microtrajectory scalar writes and implicit prediction calls are not exhaustively counted; complete process CPU is retained. Setup/editing/read-only inspection are not separately CPU-metered. Artifact generation and delivery are metered separately. No fitted checkpoint retained.')
 write(w/'accounting.json',accounting)
 compact=[]
 for r in stream['rows']:
  compact.append(dict(arm=r['arm'],phase=r['phase'],digests=r['digests'],counts=r['counts'],cursor_sha256=__import__('hashlib').sha256(json.dumps(r['cursor'],sort_keys=True).encode()).hexdigest(),episodes=r['cursor']['episodes'],coverage=r['coverage'],credit_event_rows=len(r['credit_counts'])))
 write(ev/'streams_summary.json',dict(passed=True,rows=compact,full_stream_manifest_sha256=sha(w/'streams.json'),research_fitting_bytes=0))
 for name in ('fold_feasibility.json','environment.json','codebook.json','accounting.json','python_counts_failed1.json','history.json','setup_notes.txt','test_research_initial.py'):
  shutil.copy2(w/name,ev/name)
 for path in sorted(w.glob('*_process.json')):shutil.copy2(path,ev/path.name)
 for path in sorted(w.glob('*.txt')):shutil.copy2(path,ev/path.name)
 source_hashes=sources()
 write(w/'blocked_source.lock.json',dict(status='BLOCKED_NO_RESEARCH_FIT',source_hashes=source_hashes,original_registration_sha256=sha(ROOT/'reports/exp0305a8/A9_REGISTRATION.md'),
   locked_external_files={n:sha(w/n) for n in ('streams.json','world_feasibility_data.json','fold_feasibility.json','python_counts_failed1.json','history.json','accounting.json')},
   research_fitting_bytes=0,passing_prefit_receipt_created=False,audit_answers_evaluated=False))
 shutil.copy2(w/'blocked_source.lock.json',ev/'blocked_source.lock.json')
 plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
 fig,ax=plt.subplots(figsize=(9,4.6));names=['Primary train','Primary development','Closed audit','Binary development'];req=[512,128,256,128];avail=[1791,641,612,96]
 xx=list(range(4));ax.bar([x-.18 for x in xx],req,.35,label='Required',color='#475569');ax.bar([x+.18 for x in xx],avail,.35,label='Eligible after exclusions',color=['#2563eb']*3+['#dc2626'])
 for x,n in zip(xx,avail):ax.text(x+.18,n+25,str(n),ha='center',fontsize=11)
 ax.set_xticks(xx,names);ax.set_ylabel('Distinct world groups');ax.set_ylim(0,2000);ax.legend(frameon=False);ax.set_title('A9 pre-fit blocker: 96 binary worlds available; 128 required',fontsize=14);fig.tight_layout();fig.savefig(out/'SHoEbox_A9_Fold_Availability.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(8.5,4.5));executed=[rows[a]['counts']['record_key_executed'] for a in ('a8','a9')];omitted=[rows[a]['counts']['record_key_omitted'] for a in ('a8','a9')]
 ax.bar(['A8 control','A9 candidate'],executed,label='Allowed record-key events',color='#2563eb');ax.bar(['A8 control','A9 candidate'],omitted,bottom=executed,label='Omitted record-key events',color='#f59e0b')
 for i,(e,o) in enumerate(zip(executed,omitted)):
  ax.text(i,e/2,f'{e:,}',ha='center',va='center',color='white',fontweight='bold')
  if o:ax.text(i,e+o/2,f'{o:,}',ha='center',va='center',fontweight='bold')
 ax.set_ylim(0,33000);ax.set_ylabel('Record-key events\nper planned 2M bytes');ax.set_title('A9 credit mask verified by data-only traversal',fontsize=14);ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.12));fig.text(.5,.01,'No research model fitting: these are stream counts, not executed campaign updates.',ha='center',fontsize=10);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'SHoEbox_A9_Credit_Mask.png',bbox_inches='tight');plt.close(fig)
 cpu=accounting['cpu_seconds'];control_bytes=accounting['successful_python_control_bytes'];historical=accounting['historical_discarded_correctness_bytes']
 report=f'''# SHoEbox EXP03/05-A9: implemented, blocked before fitting

16 November 2026 (America/Denver). Starting main:
`3c24b6623e8f03ae39f2d7d6266609bfcc9344ba`.

**A9 cannot satisfy its original pre-fit data requirements: 128 fresh binary
development worlds are required, but only 96 are eligible. Research fitting:
0 bytes.** Public E8 and every original A1–A8 result, timing, cost and closed
audit remain unchanged. No A9 capability or throughput result is claimed.

## The exact blocker

The original v9 salt plus NUL, RNG 20261201, 64-key pool and all mandatory
historical exclusions are preserved. Exhaustive independent enumeration of
canonical world JSON, SHA-256 fold assignment and the complete exclusion union
agrees exactly with the production generator and seeded ordering.

| Split | Required worlds | Eligible after exclusions | Status |
|---|---:|---:|---|
| Three-value training | 512 | 1,791 | Available |
| Three-value development | 128 | 641 | Available |
| Three-value closed audit | 256 | 612 | Available; no answers scored |
| Binary development | 128 | **96** | **Blocked: 32 missing** |

There are 4,032 possible binary worlds. The v9 development bucket contains 784
before exclusions and 96 afterward. Across all buckets, 491 binary worlds
remain eligible (303 train, 96 development, 92 audit). Excluding the 896 selected
primary worlds leaves 336 unused binary worlds, which supports the prospective
allocation repair below. None has been scored or used for a fit.

The full pre-fit suite retains its 128-world requirement and raises
`ValueError: only 96 dev worlds at pool 64`. The shortage was independently
confirmed and rechecked after cleaning a duplicate explicit exclusion in the
test. No salt, exclusion, seed, count or gate was altered. There is no passing
pre-fit receipt, and the fitting driver cannot start without that committed receipt.

![World availability](SHoEbox_A9_Fold_Availability.png)

## Implemented and verified change

A9 retains A8's alternating 64-episode PAD/diverse curriculum. It passes a
training-only `allow_record_key_update` flag, separate from the six labels and
answer mask. Every positive record-key event in a diverse continuation episode
skips its 16-row hybrid update. All positive record labels still train presence
and values. Query-key, applicability and native rules remain unchanged.

The flag is derived from the exact saved episode index. It is not a parameter,
model field, inference argument or saved model state. Standalone local calls
default to enabled credit; generic calls ignore it and cannot update local heads.
A9 has checkpoint magic `0x3941353050584553`, nested v19, and outer cursor
`exp0305-a9-cursor-1`. FMA is disabled. No slots, caches, moments, codebook,
matching coefficients or inference arithmetic were added or changed.

Enabled credit matches A8 byte for byte after reversing only the envelope
identity, including every-byte local/native updates, full-256 predictions,
admission and saved state. Independent native checks show disabled credit
changes only eligible record-key rows and the projection counter for an isolated
local call. Test-only hybrid-call instrumentation confirms omitted events
perform no hidden hybrid/code comparisons. Unaffected heads equal the original
A8 reference. Gradients, finite differences, all 256 codes, coupling, selection,
causality, occurrence/lifecycle and exact partial-block resumes pass.

## Exact stream and credit accounting

All six per-arm warmup/continuation streams were traversed without model
weights. A8/A9 input/target/label/mask digests match exactly; their continuation
credit digests differ. Both RNGs, order, episode positions, lengths, labels,
64 phases and finish schedule match. Boundaries 63/64/127/128, zero spacers,
hash rejection/additional blocks and partial-spacer/block resumes are checked.
Each continuation encounters 11,090 episodes, including its partial last episode.

| Per planned 2M continuation | E8 | A8 | A9 |
|---|---:|---:|---:|
| Eligible positive record-key events | 28,831 | 28,831 | 28,831 |
| Allowed record-key events | 28,831 | 28,831 | **14,524** |
| Omitted record-key events | 0 | 0 | **14,307** |
| Total local projection rows | 1,046,808 | 1,046,808 | **817,896** |

A9 omits 228,912 key-gradient rows, 3,662,592 code comparisons and 58,601,472
coordinate-distance comparisons per planned continuation. These are **data-only
planned counts**, not executed research updates. Actual discarded correctness
updates and their executed/omitted ledgers are reported separately.

![Credit accounting](SHoEbox_A9_Credit_Mask.png)

## Verification and cost

- 60 public/runtime and 209 historical research Python tests pass.
- A9's complete 20-test invocation has 19 passes and one data-feasibility error.
  The single affected grouping check was rerun and preserves the same failure.
- 23 protected-loader rejection checks and 14 historical checkpoint trajectories pass.
- Optimized and ASan/UBSan checks pass: 824,111 core, 723 public E8,
  13,863 inherited selection/lifecycle, 75,295 hybrid, 7,826 presence and
  **57,280 A9 masked-credit checks per build**. Leak checking was disabled.
- All 256 fixed codes remain unique; enabled A8 parity, generic/local separation
  and exact cursor correctness pass. v5/v6 loader sources are unchanged but
  were not independently instantiated, matching the historical check scope.

Research fitting, pilots, inherited warmups, replay, development inference,
audit scoring and qualification: **zero**. The planned 20.25M-byte campaign was
not launched. A9 Python controls consumed **{control_bytes:,} discarded update
bytes**; historical trajectories consumed **{historical:,}**. Direct Python
observe calls total 1,172. Six data-only stream traversals cover 6.75M bytes,
not model training. Every failed command and repeated control is retained.

Whole-process research/verification cost at report creation is **{cpu:.3f} CPU
seconds**, including descendants and compilation. Setup/editing/read-only
inspection is not separately metered. Historical/native microtrajectory scalar
writes and implicit prediction calls are not exhaustively counted; their full
CPU/logs remain. Artifact generation, final validation and packaging have
separate receipts. `accounting.json` records exact successful Python event
counts, native masked-reference calls and every process cost without double
counting subphase CPU.

The first public-native command used a nonexistent module and exited before
model work; the corrected command passed. Both logs/costs remain. The binary
shortage is the sole remaining pre-fit blocker. There are no fitted A9
checkpoints and no 18-pair fitted-cursor claim.

## Gates deliberately not executed

The full primary/position/offset/order/padding/repeat/diverse/lesion/binary
measurements and isolated **>=80% native throughput** require the registered
fitted snapshots. None was executed or replaced by an untrained-model estimate.
There are no full outcomes to lock or diagnose. Teacher-fed subset diagnosis,
post-1M-language retention and fresh generic/1M/5M capacity qualification also
remain unexecuted. Every original threshold and sample size remains binding.

## One focused next preregistration

`A9_F1_REGISTRATION.md` changes only binary development allocation: use the
128-world seeded sample from all eligible binary worlds outside the unchanged
896 selected primary groups, instead of requiring the exhausted v9 development
bucket. All historical exclusions remain; 336 candidates exist. The same
20261201 RNG and original enumeration order are fixed. No F1 identities were
selected, no F1 code was implemented and no fitting occurred.

A subsequent execution must read the original A9 registration and F1 together,
implement the allocation, pass every prerequisite and commit complete source/
world/stream/credit locks before the original exact 20.25M-byte comparison.
Failure retains E8 and closed audits; development success still requires separate
fresh generic/capacity qualification. The A9 mechanism has no empirical outcome
from this turn, so no numerical successor or gate relaxation is justified.

## Delivery and resume

`RESUME.md` and the archive preserve source/history, raw availability proof,
all six exact stream/cursor/credit manifests, failed/passing controls and CPU
receipts. The archive resumes **pre-fit development**, not a partially trained
campaign. Datasets and compiled binaries remain outside git; no trained model
exists. The verified publication and copy/paste prompt accompany the download.

The user's explicit authority to commit/push completed verified work directly
to main, including unsuccessful research, remains in force. Preserve unrelated
changes and remote history; never force-push. Deliver ordinary Markdown, PNGs,
a resumable archive and the next prompt; no Pages or Spaces.
'''
 (out/'SHoEbox_EXP0305_A9_Report.md').write_text(report)
 write(out/'summary.json',dict(status='blocked_before_fitting',required_binary_dev=128,available_binary_dev=96,research_fitting_bytes=0,a9_development_pass=None,throughput_measured=False,audits_opened=False,next_registration='A9_F1_REGISTRATION.md',cpu_seconds_at_report=cpu))
 print(json.dumps(dict(status='blocked_before_fitting',source_files=len(source_hashes),cpu_seconds=cpu,correctness_bytes=control_bytes),indent=2))
if __name__=='__main__':main()
