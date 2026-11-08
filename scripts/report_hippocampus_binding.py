"""Readable results for cue timing, expert reinstatement and bounded CA1."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_hippocampus import render_pdf, table


def paired_rows(rows, denominator):
    return [[label, *[f"{m[f'pairs_{g}']['both_correct']}/{denominator}"
                     for g in (0, 32, 64)]] for label, m in rows]


def retention_rows(rows):
    return [[label, f"{100*m['single_64']['accuracy']:.1f}%",
             f"{100*m['correction_64']['accuracy']:.1f}%",
             f"{m['language']['nll']:.6f}"] for label, m in rows]


def write_report(data, out):
    ca1 = data['ca1']
    old, new = (ca1['models'][n] for n in ('legacy', 'bounded'))
    placement = data['models']['event_expert']
    decision = ('The corrected endpoint passes every declared validation gate.' if new['promoted'] else
                'The corrected endpoint does not pass the declared validation gates; keep the original retained parent as the working model.')
    text = f'''# SHoEbox hippocampus: retrieval control repair

November 7, 2026 (Mountain Time)

{decision} This pass completes a 64-update cue/reinstatement experiment and two matched 16-update controller continuations. All three controls remain independently selectable, with legacy defaults preserved.

## The CA1 correction

The audit exposed a suppression loop: accept, null and refine compete in one softmax, but the legacy controller multiplies the refined return by the original accept probability. When refine wins, it suppresses the reread it requested.

The corrected bounded policy delivers `a*v + r*a_after*v_refined`. The reread gets a fresh terminal accept/null decision, and its contributor receipt uses actual combined acceptance. Legacy delivery is `a*((1-r)*v + r*v_refined)`. Null can still reject either branch. A disabled or exhausted refinement budget cannot consume probability through an unavailable action.

The fix adds no parameters, transport fetches or native graph rounds. It rereads the selected owner capsules already in the bounded query and reuses the learned comparator.

## Matched 16-update continuation

Both arms start from identical update-64 weights, AdamW state, RNG, carried language state, stream cursor and next document. Only CA1 policy differs. The fresh test bank uses seed 1699/start 160000 and 96 conflicting histories per delay; both questions about a history must be correct.

'''
    rows = [('Legacy CA1', old['measurements']), ('Bounded CA1', new['measurements'])]
    text += table(['Controller', 'Both +0', 'Both +32', 'Both +64'], paired_rows(rows, 96)) + '\n\n'
    text += table(['Controller', 'Single +64', 'Correction +64', 'Text NLL'], retention_rows(rows)) + '\n\n'
    text += 'Retention uses 64 single-fact and 128 correction answers. Text NLL uses 1,024 held-out natural bytes; lower is better. These tests do not select checkpoints.\n\n'
    immediate = new['measurements']['pairs_0']
    text += (f"The corrected arm still favors the newest fact: immediate newer-value accuracy is "
             f"{100*immediate['newer_accuracy']:.1f}%, older-value accuracy is "
             f"{100*immediate['older_accuracy']:.1f}%, and changing the question switches "
             f"{immediate['query_switches']}/96 conflicting-pair outputs. Query-conditioned "
             'selection remains the next development target.\n\n')
    controls = [('Intact bounded', new['measurements']),
                ('Readbacks blocked', ca1['controls']['bounded_blocked']),
                ('Legacy policy restored', ca1['controls']['bounded_legacy_policy'])]
    text += table(['Same final weights', 'Both +0', 'Both +32', 'Both +64'], paired_rows(controls, 96)) + '\n\n'

    text += '## Earlier placement experiment\n\n'
    text += '''The first change advances the regional cue GRU once per incoming byte instead of once per internal round. The second places recalled evidence after the regional HAT adapter, directly before expert routing and computation. Capsule interpretation remains with the owner. Global objectives and all-parameter training are unchanged.

This 64-update comparison starts from the original strong parent. Its separate held-out bank uses seed 1511/start 120000 and 192 conflicting histories per delay. It must not be pooled with the later 96-history bank.

'''
    names = [('Retained parent', 'parent'), ('Previous hippocampus', 'legacy'),
             ('Event cue + expert input', 'event_expert')]
    rows = [(label, data['models'][name]['measurements']) for label, name in names]
    text += table(['Model', 'Both +0', 'Both +32', 'Both +64'], paired_rows(rows, 192)) + '\n\n'
    text += table(['Model', 'Single +64', 'Correction +64', 'Text NLL'], retention_rows(rows)) + '\n\n'
    control_counts = lambda m: '/'.join(str(m[f'pairs_{g}']['both_correct']) for g in (0, 32, 64))
    text += (f"The intact endpoint solves {control_counts(placement['measurements'])} pairs at +0/+32/+64. "
             f"Blocking readbacks gives {control_counts(data['controls']['blocked'])}; restoring the round clock gives "
             f"{control_counts(data['controls']['round_cue'])}; restoring ingress delivery gives "
             f"{control_counts(data['controls']['ingress_read'])}. Each count is out of 192.\n\n")
    before = next(r for r in data['diagnostics']['before']['measurements'] if r['gap'] == 32)
    after = next(r for r in data['diagnostics']['after']['measurements'] if r['gap'] == 32)
    text += (f"After this first phase, query-cue cosine remains {after['cue_query_cosine']:.6f} "
             f"(previously {before['cue_query_cosine']:.6f}). The controller allocates "
             f"{100*after['ca1_mean'][2]:.1f}% to refine but only {100*after['ca1_mean'][0]:.1f}% "
             'to accept. This observation motivated the separate CA1 repair.\n\n')
    text += '## Controller signal audit\n\n'
    od, nd = old['diagnostic'], new['diagnostic']
    text += table(['At +32 bytes', 'Legacy continuation', 'Bounded continuation'], [
        ['Effective acceptance', *[f"{100*d['effective_acceptance_mean']:.2f}%" for d in (od, nd)]],
        ['Returned residual, RMS', *[f"{d['residual_rms']:.6f}" for d in (od, nd)]],
        ['Query cue cosine', *[f"{d['cue_query_cosine']:.6f}" for d in (od, nd)]],
        ['Answers changed by blocking', *[f"{d['blocked_prediction_changes']}/64" for d in (od, nd)]]]) + '\n\n'
    text += 'The diagnostic bank is seed 881/start 4000, two templates and 64 answers, separate from final tests. Effective acceptance includes accepted rereads; raw terminal and three-way softmax probabilities are not directly comparable.\n\n'

    text += '## Reproducibility and delivery\n\n'
    text += table(['Locked validation gate', 'Legacy continuation', 'Bounded continuation'], [
        [label, *['Pass' if row['promotion_gates'][key] else 'Fail' for row in (old, new)]]
        for label, key in [('Paired recall', 'paired'), ('Single-fact retention', 'single'),
                           ('Corrections', 'correction'), ('Language NLL', 'language')]]) + '\n\n'
    text += '''Promotion still requires at least 50% both-correct pairs at delays 0 and 32, no more than a two-point single-fact drop, no more than a three-point correction drop, and no more than +0.05 language NLL versus the original strong parent. Validation is seed 683/start 3000 with eight templates. Fixed final endpoints are primary; validation-selected checkpoints are also preserved.

'''
    text += (f"Validation selects update {old['selected_update']} for legacy CA1 and update "
             f"{new['selected_update']} for bounded CA1. Update 0 is the unchanged starting "
             'weights under the new policy; the result tables above use update 16 in both arms.\n\n')
    text += f"The hippocampus models retain {placement['parameters']:,} unique parameters. The source schedules match within each comparison, including every mode, delay, document index and source count. The continuation starts at the next unused document and restarts the same mode/gap cycle in both arms. There is no replay, task parser or auxiliary answer target.\n\n"
    budget_rows = []
    for label, row in [('64-update path arm', placement), ('Each 16-update CA1 arm', new)]:
        t = row['totals']
        budget_rows.append([label, f"{t['memory_targets']:,}", f"{t['memory_bytes']:,}",
                            f"{t['language_bytes']:,}"])
    text += table(['Budget per arm', 'Answer targets', 'Task bytes', 'Language bytes'], budget_rows) + '\n\n'
    text += '''Sparse activation recomputation preserves the complete live credit window; state detaches only at optimizer updates. The new 64-update run uses 16-byte segments above 384 batch-byte positions for updates 1-16 and above 768 thereafter. The original comparison has its recorded adaptive recovery schedule. Both CA1 arms use the same 512-position threshold. Different recomputation schedules can change floating-point accumulation order.

'''
    text += f"{sum(data['tests'].values())} targeted tests pass: 21 existing hippocampus, five cue/reinstatement, five CA1 and 11 runtime/checkpoint checks. They cover warm-start parity, inactive rows, expert influence, accepted and rejected rereads, causal credit, checkpoint migration and recomputed logits/gradients. Legacy mode also reproduces public revision 59244b3 logits exactly on 16 trained-checkpoint histories and 32 question branches.\n\n"
    text += (f"Recorded training-update time is {placement['training_seconds']/60:.1f} minutes for the path arm, "
             f"{old['training_seconds']/60:.1f} for legacy CA1 and {new['training_seconds']/60:.1f} for bounded CA1. "
             'Validation, diagnostics and final evaluation are additional.\n\n')
    text += '''The source and checkpoint archives include the original parent, previous hippocampus, new path endpoint and both CA1 continuations; selected weights, full predictions, optimizer/RNG/stream state, exact corpus and SHA-256 checksums are preserved. Reproduction commands are in `docs/hippocampus_binding.md`; the current handoff is `STATUS.md`. Exact-resume source guards remain enforced: use the recorded revision to resume an old protocol, or start an explicit continuation under the new controller.
'''
    out.mkdir(parents=True, exist_ok=True)
    md = out / 'SHoEbox_Hippocampus_Binding_Report.md'
    md.write_text(text)
    return md


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, default=ROOT/'experiments/text_regions/results/hippocampus_binding/summary.json')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    md = write_report(json.loads(a.summary.read_text()), a.out)
    render_pdf(md, md.with_suffix('.pdf'), title='SHoEbox hippocampus retrieval control',
               footer_text='SHoEbox | Retrieval control repair | November 7, 2026 MT',
               page_break_before=('Earlier placement experiment', 'Reproducibility and delivery'))
    print(md)
    print(md.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
