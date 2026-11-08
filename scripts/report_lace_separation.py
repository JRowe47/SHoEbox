"""Compact downloadable Markdown/PDF report from audited measured results."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.report_hippocampus import render_pdf,table


def write_report(s,out):
    m=s['measurements'];pct=lambda x:f'{100*x:.1f}%'
    labels={'parent':'Retained parent','A':'A: bypass','B':'B: learned head','C':'C: unique credit','D':'D: full proposal'}
    counts=lambda row,stem='pairs': [f"{row[f'{stem}_{g}']['both_correct']}/{row[f'{stem}_{g}']['conflicting_pairs']}" for g in (0,32,64)]
    decision='No arm passes promotion. Keep the retained strong parent as the working model.' if s['retained_parent'] else 'At least one arm passes the declared validation promotion gates; see the gate table.'
    text=f'''# SHoEbox: query-conditioned lace separation

November 7, 2026 (Mountain Time) | Completed four-arm pilot

{decision} All four arms complete the locked 32-update continuation from the same bounded-CA1 endpoint. The new mechanism is implemented behind default-off switches, with rank-four query-conditioned bias, helpful-unique reinforcement and selective inhibition of shared routes receiving opposing task credit.

## Fixed endpoint results

Both questions must be correct about the same unchanged history. The fresh final bank has sixteen templates, a full 4x4 value crossing, both source orders and 192 distinct-answer histories per delay. Each cell below is a count, not a switching rate.

'''
    text+=table(['Model','Both +0','Both +32','Both +64'],[[labels[a],*counts(m[a]['test'])] for a in labels])+'\n\n'
    text+=table(['Model','Single +64','Correction +64','Text NLL'],[[labels[a],pct(m[a]['test']['single_64']['accuracy']),pct(m[a]['test']['correction_64']['accuracy']),f"{m[a]['test']['language']['nll']:.6f}"] for a in labels])+'\n\n'
    text+='Retention uses 64 single-fact and 128 correction answers. Text NLL uses 1,024 natural-byte targets. These established cohorts remain unchanged from the preceding CA1 pass.\n\n'
    text+=s.get('assessment','')+'\n\n'

    text+='## Validation and promotion\n\n'
    text+=table(['Model','Both +0','Both +32','Both +64'],[[labels[a],*counts(m[a]['validation'])] for a in labels])+'\n\n'
    text+=table(['Arm','Paired','Single','Correction','Language'],[[a,*['Pass' if s['gates'][a][k] else 'Fail' for k in ('paired','single','correction','language')]] for a in 'ABCD'])+'\n\n'
    text+='The gates require at least 50% both-correct validation pairs at +0 and +32, no more than two percentage points of single-fact loss, three points of correction loss, or +0.05 natural-text NLL against the retained parent on matching cohorts. Validation retention uses 32 single-fact and 128 correction answers at each of +0/+32; text uses 256 targets.\n\n'
    text+='Initial and final validation are preserved. No intermediate checkpoint was selected. The final test stayed sealed until all four endpoints and source ledgers were complete.\n\n'
    text+='The native short grammar can repeat exact inputs despite fresh seeds and indices. New paired cohorts therefore add 24 held-out natural-text bytes before both writes. Retention gaps remain unchanged. Diagnostic seed/start is 2309/180000, validation 2411/200000 and final 2539/240000. Validation has eight templates and 96 distinct-answer histories per delay; final has sixteen and 192. All models, including the retained parent, see the same locked inputs.\n\n'
    audit=s['cohort_audit']
    text+=f"The byte-hash audit compares against {audit['previous_rendered_answers']:,} distinct rendered prior answers from reconstructed evaluation/training banks. It finds zero historical and cross-cohort exact-input intersections. Without the natural lead-ins, {audit['unprefixed_native_answers_repeating_history']} new native answers would repeat old inputs. Semantic fact combinations are intentionally reused and balanced. The historical audit correction changed no rendered evaluation input.\n\n"

    text+='## Mechanism and ablations\n\n'
    text+='A versus B measures the conditional head; B versus C measures unique credit; C versus D measures contentious-overlap inhibition. The tables report final weights at update 32 in every arm.\n\n'
    text+='Each node concatenates its existing local lace context with its dedicated regional cue. The bounded bias enters fresh scores before top-k. No question IDs, answer IDs, owner annotations or source positions enter this head. Ordinary gradients still train every upstream component. Extra credit uses the final selected-score gradient from the normal task backward and changes only the small head.\n\n'
    pre=s['preflight']['diagnostics']
    text+=table(['Initial diagnostic','Selected overlap','Input RMS difference','Updated / eligible'],[[f'+{g} bytes',pct(pre[str(g)]['mechanism']['hard_overlap']),f"{pre[str(g)]['mechanism']['input_rms_difference']:.5f}",f"{pre[str(g)]['mechanism']['updated_decisions']}/{pre[str(g)]['mechanism']['eligible_decisions']}"] for g in (0,32)])+'\n\n'
    text+='Inputs differ and opposing credit is available even with high hard-route overlap. The diagnostic nonzero head also changes answer logits when valid paired cues are swapped. Thus the preflight verifies a live causal path rather than relying on overlap alone.\n\n'
    text+=table(['Arm, training mean','Selected overlap','Soft L1','Updated / eligible','Bias RMS'],[[a,pct(s['training'][a]['mechanism']['means']['hard_overlap']),f"{s['training'][a]['mechanism']['means']['soft_l1']:.4f}",f"{s['training'][a]['mechanism']['sums']['updated_decisions']}/{s['training'][a]['mechanism']['sums']['eligible_decisions']}",f"{s['training'][a]['mechanism']['means']['bias_rms']:.4f}"] for a in 'ABCD'])+'\n\n'
    text+='Coverage counts aligned controller/round-pair decisions; each includes three lace layers. Same-target decisions receive zero extra correction in every arm. The added gradient never exceeds 25% of the ordinary head-gradient norm. Captures skip inactive rows and no-grad checkpoint forwards, and clear after use. Final diagnostic records separately preserve shared/unique contribution mass, opposing-credit counts, input separation, saturation and gradient norms.\n\n'
    text+=table(['Final +32 diagnostic','Selected overlap','Unique mass','Updated / eligible','Extra / ordinary'],[[a,pct(s['endpoint_mechanism'][a]['32']['mechanism']['hard_overlap']),pct(s['endpoint_mechanism'][a]['32']['mechanism']['unique_mass']),f"{s['endpoint_mechanism'][a]['32']['mechanism']['updated_decisions']}/168",f"{100*s['endpoint_mechanism'][a]['32']['mechanism']['extra_head_norm']/s['endpoint_mechanism'][a]['32']['mechanism']['ordinary_head_norm']:.3f}%"] for a in 'BCD'])+'\n\n'
    text+='## Fixed-weight controls\n\n'
    d=m['D']['test']
    text+=table(['D, distinct targets','Older correct','Newer correct','Query switches'],[[f'+{g} bytes',pct(d[f'pairs_{g}']['distinct_older_accuracy']),pct(d[f'pairs_{g}']['distinct_newer_accuracy']),f"{d[f'pairs_{g}']['query_switches']}/192"] for g in (0,32,64)])+'\n\n'
    text+='Switching counts are separate from correctness. Equal-answer controls and all per-case predictions are retained in the machine-readable evidence.\n\n'

    text+=table(['Same D weights','Both +0','Both +32','Both +64'],[['Intact',*counts(d)],*[[label,*counts(s['ablations'][key])] for label,key in [('Bias bypassed','bias_off'),('Paired cues swapped','cue_permuted'),('Readbacks blocked','blocked')]]])+'\n\n'
    text+='The cue swap changes only the head cue input between questions sharing a valid history. C versus D is the training-time inhibition comparison.\n\n'
    text+=table(['Unseen wording','Both +0','Both +32','Both +64'],[[labels[a],*counts(m[a]['test'],'unseen')] for a in labels])+'\n\n'
    text+='The separate predeclared wording changes `A x?` to `Value of A x?`; it is never used for training or endpoint selection. There are 192 distinct-answer histories and 64 equal-answer histories per delay in this bank as well.\n\n'

    text+='## Cost and recovery\n\n'
    t=s['training']['A']['totals']
    text+=f"Each arm consumes {t['memory_targets']:,} answer targets, {t['memory_bytes']:,} logical task bytes, {t['executed_positions']:,} executed batch-byte positions and {t['language_bytes']:,} natural-text targets. Executed positions include padded admissions; actual expert execution is counted separately. Source documents advance from 800 through 855, and the full per-update modes, gaps, targets and budgets match across arms.\n\n"
    text+=table(['Arm','Update time (min)','Peak RSS (MiB)','Bookkeeping (ms/pair update)','Task experts'],[[a,f"{s['training'][a]['seconds']/60:.2f}",f"{s['training'][a]['peak_mib']:.0f}",f"{1000*s['training'][a]['mechanism']['means']['seconds']:.2f}",f"{s['training'][a]['task_expert_executions']:,}"] for a in 'ABCD'])+'\n\n'
    text+='Bookkeeping averages paired updates and includes diagnostic measurement in A/B, whose extra credit is zero.\n\n'
    p=s['preflight'];bench=p['timing']
    text+=f"Installed models have {p['parameters']:,} unique parameters, including {p['head_parameters']:,} new head parameters (448 per node). Arm A installs but bypasses the same heads. Top-k and expert counts remain fixed; changed contribution mass is functional redistribution, not fewer expert executions.\n\n"
    text+=f"The matched initial timing probe records {bench['bypass']['forward_backward_seconds']:.3f} s for bypass, {bench['head']['forward_backward_seconds']:.3f} s for the head alone, {bench['instrumented']['forward_backward_seconds']:.3f} s with capture and {bench['credit']['forward_backward_seconds']:.3f} s with capture for the credit arm. Local credit application adds {1000*bench['credit']['mechanism']['seconds']:.2f} ms. Wall times are per process during concurrent CPU runs. Full training time includes language and task forward/backward, clipping and AdamW; evaluation and packaging are additional. Detailed inference timings and memory reads/refinements are recorded for each identical cohort.\n\n"
    text+=table(['Final +32 inference','Bank time (s)','Posting reads','Readbacks','Refinements'],[[a,f"{m[a]['test']['pairs_32']['seconds']:.1f}",*[f"{m[a]['test']['pairs_32']['memory_work'].get(k,0):,}" for k in ('posting_reads','readbacks','refinements')]] for a in 'ABCD'])+'\n\n'
    text+='Each inference bank contains 256 histories and both questions, including equal-answer controls; times include shared-history execution on concurrent CPU workers. These are fixed workload wall times. Raw records also preserve expert counts, executed positions, directory/candidate work and peak memory.\n\n'
    text+='## Reproduction and checks\n\n'
    text+='The 52 focused checks pass, with a subsequent ten-check rerun after integration. They cover zero/disabled parity, owner/layer/pair alignment, credit signs, useful-to-both and same-target protection, analytic derivative agreement, inactive rows, runtime/reference equivalence, recomputation and exact resume. A separate production-checkpoint audit compares a complete bypassed update with the unchanged continuation, including inherited optimizer moments and carried language state.\n\n'
    text+='All runs preserve LR .003, AdamW (.9,.95), weight decay .01, language weight .25, syntax weight .1 and global clip 1. They use the same 16-byte recomputation segments above 512 positions. The extra-credit path uses unscaled FP32. There is no second adversarial model, replay store, persistent inhibition trace, altered capsule or global optimizer change.\n\n'
    text+='The source archive includes implementation, tests, protocols, measured summaries and full prediction files. The checkpoint archive includes the strong parent, experimental starting endpoint, initial/8/16/24/32 snapshots, final weights, AdamW/RNG/carried stream state, exact corpus, cohort manifests, timing and checksums. Source and protocol guards remain enforced.\n\n'
    text+='Resume the identical arm with `python scripts/run_lace_separation.py --arm D --out artifacts/lace_separation/D --initial artifacts/lace_separation/initial.json --resume`. For a fresh reproduction, use an empty output directory without `--resume`. The full protocol and commands are in `docs/lace_separation.md`; the current working checkpoint and next development target are in `STATUS.md`.\n'
    text+=f"\nTraining source revision: [{s['training_source_revision'][:7]}](https://github.com/JRowe47/SHoEbox/commit/{s['training_source_revision']}). The completed audit checks all 38 evaluation sections against their checkpoint, cohort and evaluator hashes, and verifies that all 223 guarded source files still match the training revision.\n"
    out.mkdir(parents=True,exist_ok=True);path=out/'SHoEbox_Lace_Separation_Report.md';path.write_text(text)
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();s=json.loads((ROOT/'artifacts/lace_separation/summary.json').read_text())
    md=write_report(s,a.out)
    render_pdf(md,md.with_suffix('.pdf'),title='SHoEbox: Query-conditioned Lace Separation',
        footer_text='SHoEbox | Lace separation | November 7, 2026 (Mountain Time)',
        page_break_before=('Validation and promotion','Mechanism and ablations','Fixed-weight controls','Cost and recovery','Reproduction and checks'))
    print(md);print(md.with_suffix('.pdf'))

if __name__=='__main__':main()
