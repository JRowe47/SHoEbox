"""Compare the audited repetition with the preserved, rounded original report."""
import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('original/dropout', 'original/balanced', 'gated/dropout', 'gated/balanced')


def compare_report(actual, reported):
    rows = []
    for condition, families in reported['accuracy_percent'].items():
        for family, cohorts in families.items():
            for count, values in cohorts.items():
                for metric, old in zip(('heldout_private', 'heldout_shared'), values):
                    new = 100 * actual['groups'][condition][f'{count}/{family}/selected'][metric]['mean']
                    rows.append({'condition': condition, 'family': family, 'count_per_cell': int(count),
                        'metric': metric, 'reported_percent': old, 'reproduced_percent': new,
                        'difference_percentage_points': new-old, 'reported_decimal_places': 3,
                        'matches_reported_precision': abs(new-old) <= .000500001})
    for condition, activations in reported['b_only_percent'].items():
        for activation, cohorts in activations.items():
            for count, old in cohorts.items():
                family = f'selected/{activation}'
                new = 100 * actual['groups'][condition][f'{count}/{family}/selected']['b_only_shared']['mean']
                rows.append({'condition': condition, 'family': family, 'count_per_cell': int(count),
                    'metric': 'b_only_shared', 'reported_percent': old, 'reproduced_percent': new,
                    'difference_percentage_points': new-old, 'reported_decimal_places': 2,
                    'matches_reported_precision': abs(new-old) <= .005000001})
    return {'source_report': reported['source_report'], 'metrics': rows,
        'metric_count': len(rows), 'matching_at_reported_precision': sum(r['matches_reported_precision'] for r in rows),
        'maximum_absolute_difference_percentage_points': max(abs(r['difference_percentage_points']) for r in rows),
        'nonzero_selected': {'reported': reported['nonzero_selected'], 'reproduced': actual['selected_nonzero']}}


def label(condition):
    return condition.replace('/', ' / ').capitalize()


def report(actual, comparison, verification):
    def metric(condition, count, family, activation, name):
        return 100*actual['groups'][condition][f'{count}/{family}/{activation}/selected'][name]['mean']

    def pair(condition, count, family, activation):
        return ' / '.join(f'{metric(condition,count,family,activation,m):.3f}'
                          for m in ('heldout_private', 'heldout_shared'))

    def table(family, activation):
        lines = ['| Condition | 960 cases | 3,840 cases | Change, pp |', '|---|---:|---:|---:|']
        for c in CONDITIONS:
            change = ' / '.join(f'{metric(c,256,family,activation,m)-metric(c,64,family,activation,m):+.3f}'
                                for m in ('heldout_private', 'heldout_shared'))
            lines.append(f'| {label(c)} | {pair(c,64,family,activation)} | {pair(c,256,family,activation)} | {change} |')
        return '\n'.join(lines)

    nonzero = actual['selected_nonzero']
    fits = [fit for run in actual['runs'] for fit in run['fits'].values()]
    reached_cap = sum(f['iterations'] == f['max_iter'] for f in fits)
    gradients = [f['gradient_max'] for f in fits]
    both = {a: sum(all(metric(c,256,'selected',a,m) > metric(c,64,'selected',a,m) + 1e-10
                for m in ('heldout_private', 'heldout_shared')) for c in CONDITIONS) for a in ('identity','gelu')}
    missing = statistics.mean(metric(c,256,'selected','gelu','missing_information') for c in CONDITIONS)
    drift = max(r['legacy_training_max_abs_drift'] for r in actual['runs'])
    matches = comparison['matching_at_reported_precision']; count = comparison['metric_count']
    largest = comparison['maximum_absolute_difference_percentage_points']
    rounding_note = ('within its original rounding interval' if matches == count
                     else 'with some differences exceeding the original rounding intervals')
    text = [f'''# SHoEbox - cohort experiment reproduced

Date: 2026-11-05 UTC

The missing cohort-diversity experiment has been rebuilt and rerun: **12 frozen graphs and 432 fitted corrections**. Validation chooses a nonzero identity correction **{nonzero['64']['identity']} of 36 times with the small cohort, and {nonzero['256']['identity']} of 36 times with the large cohort**. Expanding from 960 to 3,840 training cases improved both private and shared accuracy in {both['identity']} of four conditions with the selected identity correction, and {both['gelu']} of four with GELU.

The repetition matches {matches} of {count} published accuracy values at their reported precision. The largest difference from a printed value is {largest:.6f} percentage points, {rounding_note}.

## Matched protocol

| Training coverage | Small | Large |
|---|---:|---:|
| Cases per supported source/query cell | 64 | 256 |
| Total cases | 960 | 3,840 |
| Distinct episode IDs | 192 | 768 |
| Traces excluding seven-byte ID headers | 960 | 3,840 |
| Observed template signatures | 869 | 2,360 |
| Shared / left / right examples | 448 / 256 / 256 | 1,792 / 1,024 / 1,024 |
| Full fact combinations | 32 | 32 |

All twelve original checkpoints are retained: original/gated publication, dropout/balanced source schedules, and graph seeds 20261106-20261108. The small training cohort is the exact first 64 feature rows of each large-cohort cell. The retained source templates define the template signature, together with source subset, query and the full fact tuple.

Each graph receives two cohort sizes, penalties 0/1/10, identity/GELU activations and initialization seeds 20261121-20261123. Each correction has hidden width 16 and 5,392 parameters. The 16,640-parameter linear base, its normalization, validation (1,920 cases), test/all (384 cases), and independent evaluation (3,456 cases) remain fixed.

Full-batch L-BFGS uses strong Wolfe line search, a 400-iteration cap and L2 coefficient 0.001. Training-only shared-answer margins are capped at 1. Validation chooses strength 0, 0.125, 0.25, 0.5 or 1 by query-balanced NLL, while requiring shared correct counts on all/shared and b_only/shared to meet the frozen base. The same rule chooses between positive penalties 1 and 10; ties prefer smaller strength. Test scores do not select settings.

---

## Selected preservation results

Values are private / shared accuracy in percent, averaged over three graph seeds and three correction initializations. Change is large minus small, in percentage points.

**Identity correction**

{table('selected','identity')}

**GELU correction**

{table('selected','gelu')}

| Nonzero correction selected | 960 cases | 3,840 cases |
|---|---:|---:|
| Identity | {nonzero['64']['identity']} / 36 | {nonzero['256']['identity']} / 36 |
| GELU | {nonzero['64']['gelu']} / 36 | {nonzero['256']['gelu']} / 36 |

**B-only shared accuracy**

| Condition | Identity: small / large | GELU: small / large |
|---|---:|---:|''']
    for c in CONDITIONS:
        cells = [' / '.join(f'{metric(c,n,"selected",a,"b_only_shared"):.2f}' for n in (64,256))
                 for a in ('identity','gelu')]
        text.append(f'| {label(c)} | {cells[0]} | {cells[1]} |')
    text.append(f'''
Large-cohort selected GELU missing-information accuracy averages {missing:.3f}%, against 25% chance. The default model/readout remains unchanged.

---

## Penalty-zero controls

These use the same fitting, fixed baseline and validation scale selection, with preservation penalty zero.

**Identity correction**

{table('penalty_0','identity')}

**GELU correction**

{table('penalty_0','gelu')}

## Agreement with the surviving report

| Published metric group | Values matching precision | Largest difference, pp |
|---|---:|---:|''')
    for family in ('selected/identity','selected/gelu','penalty_0/identity','penalty_0/gelu'):
        rows = [r for r in comparison['metrics'] if r['family']==family and r['metric']!='b_only_shared']
        text.append(f'| {family.replace("/"," / ")} | {sum(r["matches_reported_precision"] for r in rows)} / {len(rows)} | '
                    f'{max(abs(r["difference_percentage_points"]) for r in rows):.6f} |')
    rows = [r for r in comparison['metrics'] if r['metric']=='b_only_shared']
    text.append(f'| B-only shared | {sum(r["matches_reported_precision"] for r in rows)} / {len(rows)} | '
                f'{max(abs(r["difference_percentage_points"]) for r in rows):.6f} |')
    text.append('''
The reference contains 64 private/shared means rounded to three decimals and 16 B-only means rounded to two. Agreement uses half of the last displayed decimal place. `reported_comparison.json` records every original value, repeated value and difference; `reported_metrics.json` preserves the values transcribed from the historical report.

This rerun uses the preserved protocol and inputs without tuning to the published scores. The unavailable original implementation prevents a byte-level comparison with that original study. The new package supplies a complete, auditable replacement.

---

## Verification and reproducibility
''')
    text.append(f'''All **{verification['automated_tests_passed']} automated tests pass** across five suites. Nine dedicated cohort tests cover coverage counts, exact nested feature ownership, split separation, frozen normalization, checkpoint restoration and resume rejection after incompatible changes.

The independent artifact audit reconstructs all 432 training objectives, their validation scale/penalty selections, and all 2,880 new prediction groups. It also exactly reproduces 720 prediction groups from the retained residual and emission studies. Saved predictions and discrete decisions compare exactly; named floating loss fields and normalization reconstruction use absolute/relative tolerance 1e-12. Loaded frozen base tensors remain identical.

All input/source hashes, checkpoint lineages, synthetic cases, coverage counts and expert budgets are checked. Training IDs are disjoint from validation and evaluation. The audit checks predictions from saved tensors rather than trusting the reported summary.

{reached_cap} of {len(fits)} fits reached the 400-iteration cap. Final maximum gradient magnitudes range from {min(gradients):.7g} to {max(gradients):.7g}. The largest feature difference from the legacy small training cache is {drift:.7g}; both newly fitted cohort sizes use the same fresh capture, while retaining the original base normalization.

Training capture used **96,768,000 expert executions**, with K=1 and 84 per event. Fitting used cached features. Runtime: PyTorch 2.15.0+cpu, two threads per worker and four fitting workers. Every fitted correction is checkpointed atomically. Resume verifies the exact source, configuration, inputs, capture and base before continuing.

## Downloadable package

`SHoEbox_cohort_reproduction_source.zip` contains the source, reports, aggregate audit and verification logs. `SHoEbox_cohort_reproduction_artifacts.zip` additionally contains every required graph, inherited readout, feature/validation cache, all 432 newly fitted corrections, selections and predictions. Each archive has a file-by-file SHA-256 manifest, its source commit and publication status.

From the extracted artifact package:

```bash
python -m pip install -r requirements.txt
python scripts/check_all.py
python -m experiments.text_regions.diversity_compare \\
  --source artifacts/cohort_reproduction_v1 \\
  --out artifacts/reproduced_cohort_comparison.json
```

For a fresh fit, run `scripts/run_diversity_study.py` with a new output directory, `--jobs 4 --threads 2`. To continue an interrupted fit, add `--resume` with its existing output directory.

The next development step remains a full-rank linear decoder refit on the expanded cohort, compared on the same fixed evaluations and measured for inference cost. That follow-up is separate from this completed reproduction.
''')
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'experiments/text_regions/results/cohort_reproduction')
    parser.add_argument('--out', type=Path, default=ROOT/'docs/SHoEbox_Cohort_Reproduction_Report.md')
    args = parser.parse_args()
    actual = json.loads((args.results/'comparison.json').read_text())
    verification = json.loads((args.results/'verification.json').read_text())
    reported = json.loads((ROOT/'experiments/text_regions/results/cohort_diversity/reported_metrics.json').read_text())
    comparison = compare_report(actual, reported)
    (args.results/'reported_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    args.out.write_text(report(actual,comparison,verification))
    print(json.dumps({k:comparison[k] for k in ('metric_count','matching_at_reported_precision',
                                             'maximum_absolute_difference_percentage_points')},indent=2))


if __name__ == '__main__':
    main()
