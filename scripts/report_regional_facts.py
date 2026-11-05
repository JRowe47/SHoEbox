"""Build the completed regional-fact report from audited, saved results."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def percent(value):
    return f'{100 * value:.2f}%'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('artifacts/regional_facts'))
    p.add_argument('--out', type=Path, default=Path('docs/SHoEbox_Regional_Facts_Report.md'))
    args = p.parse_args()
    root = args.root
    selection = read(root / 'selection.json')
    audit = read(root / 'evaluation_audit.json')
    verification = read(root / 'verification.json')
    assert selection['all_passed'] and audit['all_passed'] and verification['all_passed']
    names = ('control', 'regional')
    reports = {name: read(root / name / 'evaluation.json') for name in names}
    diagnostics = {name: read(root / name / 'local_diagnostic.json') for name in names}
    histories = {name: read(root / name / 'history.json') for name in names}
    summaries = {name: read(root / name / 'summary.json') for name in names}
    assert all(s['completed_step'] == 100 for s in summaries.values())
    scores = {name: reports[name]['test_capabilities']['scores'] for name in names}
    labels = [('composition', 'Composition, K=2'), ('recall', 'Delayed recall, gap 16'),
              ('correction', 'Delayed correction, gap 16'), ('shared', 'Shared fact, all sources'),
              ('left', 'Left-private fact, all sources'), ('right', 'Right-private fact, all sources'),
              ('a_only/shared', 'A-only shared fact'), ('a_only/left', 'A-only left-private fact'),
              ('b_only/shared', 'B-only shared fact'), ('c_only/shared', 'C-only shared fact'),
              ('c_only/right', 'C-only right-private fact')]
    score_table = table(['Held-out task', 'Control', 'Regional supervision'],
                        [[label, *(percent(scores[n][key]) for n in names)] for key, label in labels])
    controls = [('k1', 'One round'), ('k2', 'Two rounds'), ('k4', 'Four rounds'),
                ('k2_no_exchange', 'Two rounds, exchange disabled'),
                ('k2_shuffled_bridge', 'Changed table, original answer'),
                ('k2_counterfactual', 'Changed table, changed answer'),
                ('k2_reset', 'State reset before query')]
    control_table = table(['Composition condition', 'Control', 'Regional supervision'],
                          [[label, *(percent(reports[n]['composition_controls'][key]['accuracy'])
                                     for n in names)] for key, label in controls])
    gate_labels = [('p_first', 'P: first local lookup'), ('q_second', 'Q: second local lookup'),
                   ('prefix_b', 'B: exact table after ingestion'), ('prefix_c', 'C: exact table after ingestion'),
                   ('query_b', 'B: exact table after query'), ('query_c', 'C: exact table after query')]
    gate_table = table(['Held-out regional check', 'Control', 'Regional supervision'],
                       [[label, *(percent(reports[n]['composition_progress']['metrics'][key]['accuracy'])
                                  for n in names)] for key, label in gate_labels])
    memory_rows = []
    for gap in ('16', '32'):
        for kind in ('recall', 'correction'):
            for condition in reports['control']['memory_controls'][gap][kind]['controls']:
                field = 'old_value_accuracy' if condition == 'withhold_update' else 'exact_accuracy'
                label = 'withheld: old value' if condition == 'withhold_update' else condition
                memory_rows.append([f'{kind}, gap {gap}, {label}', *(
                    percent(reports[n]['memory_controls'][gap][kind]['controls'][condition][field])
                    for n in names)])
    memory_table = table(['Memory condition', 'Control', 'Regional supervision'], memory_rows)
    local_rows = []
    for kind in ('recall', 'correction'):
        for node in ('P', 'Q'):
            row = diagnostics['regional']['records'][kind]['nodes'][node]['current']
            local_rows.append([f'{kind}: {node}', percent(row['write_accuracy']), percent(row['accuracy'])])
    local_table = table(['Regional memory head', 'At write', 'At query'], local_rows)
    fact_rows = []
    for kind in ('shared', 'left', 'right'):
        for node in (('P', 'Q') if kind == 'shared' else (('P',) if kind == 'left' else ('Q',))):
            row = diagnostics['regional']['records']['views/' + kind]['nodes'][node][kind]
            fact_rows.append([f'{kind}: {node}', percent(row['write_accuracy']), percent(row['accuracy'])])
    fact_table = table(['Regional fact head', 'At write', 'At query'], fact_rows)
    selected = selection['selected']
    selection_rows = [[n, selection['runs'][n]['selected_step'],
                       percent(selection['runs'][n]['selected_validation']['minimum_accuracy'])] for n in names]
    selection_table = table(['Run', 'Selected phase update', 'Weakest validation task'], selection_rows)
    totals = summaries['control']
    regional_validation = histories['regional'][-1]['validation']
    validation_query_b = regional_validation['composition']['metrics']['query_b']['accuracy']
    narrative = f'''# SHoEbox: regional facts and corrections

Completed November 5, 2026. Two matched 100-update continuations; all remaining training, evaluation, export and audit work is complete.

## Result

Training-only local supervision did not solve the integrated memory and fact tasks in this run. The control scores {percent(scores['control']['composition'])} on held-out composition; regional supervision scores {percent(scores['regional']['composition'])}. The validation rule selects `{selected['checkpoint']}` at phase update {selected['step']}. M7 remains open.

Regional supervision improves shared-fact answers from {percent(scores['control']['shared'])} to {percent(scores['regional']['shared'])}, with smaller private-fact gains, while trained-delay recall and correction fall. Both arms continue the same integrated update-200 parent with identical initialization, optimizer moments, examples, carried stream states and random states. The sole experimental difference is local-loss weight 0 versus 1. The regional memory/fact heads add 528 training-only parameters; the exported inference graph remains unchanged.

## Matched endpoint scores

{score_table}

Composition uses all 592 held-out permutation cases. Each memory condition and fact-query type uses 512 fresh episodes. Missing-source controls use independently generated full-factorial cohorts. The four-value reference is 25%; every task answer is chosen from all 256 byte logits.

---

## Composition and source retention

{control_table}

The two-round answer remains dependent on exchange and retained source state. Replacing a table changes the expected answer: scoring against that changed answer recovers high accuracy. Four rounds do not improve this trained two-round computation.

{gate_table}

Exact source-table retention is a 100% gate; the answer and local-lookup gates require at least 90%. The regional endpoint scored {percent(validation_query_b)} on B's post-query exact-table validation check, missing that gate. Its held-out table checks above all pass; checkpoint selection still follows the validation results.

## Validation selection

{selection_table}

Checkpoints are ranked using validation only: full qualification, composition preservation, weakest capability accuracy, macro accuracy, then lower macro NLL. The initial checkpoint is an allowed fallback. Held-out scores do not rank checkpoints. The selected control checkpoint is its update-100 endpoint; the regional arm retains its initial checkpoint as its validation choice.

---

## Delayed memory controls

{memory_table}

The gap-32 evaluation extends the delay beyond the gap-16 training setting. Reset controls remove carried state at the query. Withheld-correction rows score agreement with the old observed value; the correction was not supplied. Raw reports also include agreement with the withheld new value, full predictions, episode identities and execution counts.

## Reading the regional memory state

{local_table}

These frozen local-head diagnostics use 64 new held-out episodes per task, after checkpoint selection. They read P and Q at the observed write and again at the final query. The memory head can decode some newly written values, but its query-time accuracy is much weaker. Persistence and later readability remain the next concrete memory bottleneck.

---

## Fact acquisition diagnostics

{fact_table}

The shared and private fact heads remain weak even at the observed-write positions in these downstream nodes. This run therefore does not establish successful fact acquisition followed only by forgetting. The saved diagnostic includes every eligible node/factor, not just the P/Q rows shown here. The control's local heads were not trained and are retained only as a matched reference.

## Completion and verification

Both runs resumed from update 90 with source hashes verified, then completed updates 91-100 and final validation. The reproduced control update-100 measurements exactly match the original pre-interruption log, excluding timing. Full resumable checkpoints, optimizer state, streams and all validation endpoints are retained.

Each arm executed {totals['expert_executions']:,} experts across {totals['external_events']:,} external events, with {totals['composition_episodes']:,} composition episodes. Every update's examples, source counts and execution budget match between arms. All 355 regression tests across five suites passed before recovery; training-source hashes are unchanged.

The evaluation audit recomputes accuracy from {audit['prediction_groups_recomputed']} stored prediction groups, verifies generated targets and cohort identities, checks matched intervention budgets, and confirms validation-only selection. Inference exports strip all four auxiliary heads and reproduce the checked logits and recurrent states exactly. The additional completion audit verifies evaluation manifests, all completed stages and diagnostic arithmetic.

## Reproduce and resume

Extract the source and artifact archives into the same directory; both use the `SHoEbox/` root. Install `requirements.txt`, then run:

```bash
python scripts/audit_regional_facts.py
python scripts/audit_regional_facts.py --evaluations
python scripts/verify_regional_completion.py
```

For a fresh matched continuation, use the two commands in `docs/SHoEbox_Regional_Facts_Protocol.md`. To resume evaluation, rerun `scripts/evaluate_regional_facts.py` with the same checkpoint, output, count and batch arguments plus `--resume`. Completed stages are reused only when their manifest matches.

The final source archive includes code, protocol, this report and recorded JSON results. The artifact archive includes training and inference checkpoints, the integrated parent, recovery checkpoints, histories, predictions, diagnostics and logs. Its manifest records the source commit and publication status. No further training phase was started.
'''
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(narrative)
    print(args.out)


if __name__ == '__main__':
    main()
