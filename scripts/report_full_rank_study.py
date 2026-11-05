"""Generate the full-rank phase report from audited results and measurements."""
import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def report(results):
    c = json.loads((results/'comparison.json').read_text())
    b = json.loads((results/'benchmark.json').read_text())
    v = json.loads((results/'verification.json').read_text())
    groups = c['groups']; lines = []

    def add(text=''):
        lines.append(text)

    def metric(group, head, key):
        return 100*groups[group][head][key]['mean']

    def pair(group, head):
        return f"{metric(group, head, 'heldout_private'):.2f} / {metric(group, head, 'heldout_shared'):.2f}"

    add('# SHoEbox - full-rank refits and compiled output heads')
    add('\nNovember 4, 2026 (America/Denver)\n')
    add('The next phase is complete: **72 full-rank byte-decoder fits across 12 frozen graphs**, '
        'compared with all 432 retained identity/GELU corrections. A full-matrix refit does not '
        'provide a consistent accuracy improvement over the compact identity correction. '
        'The useful engineering result is a compiled output head: linear refits and identity '
        'corrections fold into one matrix multiplication, with unchanged predicted bytes on every audited case.')
    add('\n## Main comparison\n')
    add('Values are held-out private / shared accuracy in percent on the 384-case all-source test set. '
        'Each condition averages three graph seeds; identity and GELU additionally average three correction '
        'initializations per graph. All updated heads use 3,840 training cases and the existing validation '
        'selection of preservation penalty and strength.')
    add('\n| Graph condition | Frozen base | Full-rank refit | Identity correction | GELU correction |')
    add('|---|---:|---:|---:|---:|')
    for group in groups:
        add('| '+group.replace('/', ' / ')+' | '+' | '.join(pair(group, head) for head in
            ('base', '256/linear/preserved/selected', '256/identity/selected/selected', '256/gelu/selected/selected'))+' |')
    add('\nThe original/dropout condition remains strongest with the retained identity correction: '
        f"{pair('original/dropout', '256/identity/selected/selected')}, versus "
        f"{pair('original/dropout', '256/linear/preserved/selected')} for the full-rank refit. "
        'The default graph and decoder remain unchanged. The compiled module and exported heads are available explicitly.')
    add('\n## Matched protocol\n')
    add('| Item | Setting |\n|---|---|')
    for key, value in [
        ('Frozen graphs', 'Original/gated publication x dropout/balanced source exposure x three graph seeds'),
        ('Training cohorts', '960 and 3,840 cases; 64 and 256 per supported source/query cell'),
        ('Cohort relationship', 'Exact nested prefixes of the already captured training features'),
        ('Input / output', '64 raw P/Q features to one common 256-byte decoder'),
        ('Normalization', 'The retained base training mean and scale; no refitting of normalization'),
        ('Validation', '1,920 fixed cases; query-balanced NLL with all/shared and B-only/shared guards'),
        ('Selection grid', 'Strength 0, 0.125, 0.25, 0.5, 1; preservation penalty 1 or 10'),
        ('Control', 'Penalty zero, with the same validation strength selection'),
        ('Evaluations', '384 test/all cases and 3,456 independent intervention cases'),
        ('Optimizer', 'Full-batch float64 L-BFGS, strong Wolfe; maximum 1,000 iterations'),
        ('Initialization', 'Frozen baseline matrix; deterministic with no RNG use'),
        ('Regularization', '0.001 coefficient; entire refitted weight and bias matrix'),
        ('Additional expert executions', 'Zero; all fitting and comparisons use retained feature caches')]:
        add(f'| {key} | {value} |')
    add('\nThe full-rank head has 16,640 fitted scalars. Each retained correction has 5,392 '
        'fitted scalars around the fixed 16,640-scalar base. The full-rank objective regularizes '
        'the entire new matrix; the earlier residual objectives regularize their factor matrices '
        'and biases. This compares practical decoder alternatives, rather than isolating rank '
        'under an identical regularizer. The residual comparisons retain their original 400-iteration fits.')
    add('\nThe preservation anchors are training cases whose shared answer the old head already gets right. '
        'A squared margin penalty discourages losing those answers, with target margins capped at one. '
        'Prediction consumes only the P/Q feature vector; query and source labels never select an output head.')
    add('\n---\n\n## Cohort size and preservation controls\n')
    add('Full-rank results below use validation-selected preservation and strength. All numbers are private / shared percent.')
    add('\n| Graph condition | 960 cases | 3,840 cases | Change, percentage points |\n|---|---:|---:|---:|')
    for group in groups:
        small = '64/linear/preserved/selected'; large = '256/linear/preserved/selected'
        change = ' / '.join(f'{metric(group, large, k)-metric(group, small, k):+.2f}' for k in ('heldout_private', 'heldout_shared'))
        add(f'| {group.replace("/", " / ")} | {pair(group, small)} | {pair(group, large)} | {change} |')
    add('\n| Nonzero full-rank update accepted | 960 cases | 3,840 cases |\n|---|---:|---:|')
    for label, family in [('Penalty zero', 'penalty_0'), ('Preservation selected', 'preserved')]:
        add(f"| {label} | {c['selected_nonzero']['64'][family]} / 12 | {c['selected_nonzero']['256'][family]} / 12 |")
    add('\n| Graph condition | Penalty-zero refit, full strength | Penalty-zero refit, selected strength | Preservation selected |\n|---|---:|---:|---:|')
    for group in groups:
        add('| '+group.replace('/', ' / ')+' | '+' | '.join(pair(group, head) for head in (
            '256/linear/penalty_0/full', '256/linear/penalty_0/selected', '256/linear/preserved/selected'))+' |')
    add('\nThe accuracy guards are enforced on validation. The test results above show where shared-answer '
        'loss still occurs on different cases; the full-rank refit does not remove that tradeoff.')
    add('\n## Independent B-only and missing-information controls\n')
    add('| Graph condition | Base B-only | Full-rank B-only | Identity B-only | GELU B-only |\n|---|---:|---:|---:|---:|')
    families = ('base', '256/linear/preserved/selected', '256/identity/selected/selected', '256/gelu/selected/selected')
    for group in groups:
        add('| '+group.replace('/', ' / ')+' | '+' | '.join(f"{metric(group, h, 'b_only_shared'):.2f}%" for h in families)+' |')
    missing = statistics.mean(metric(group, families[1], 'missing_information') for group in groups)
    add(f'\nSelected full-rank accuracy with the required information missing averages **{missing:.3f}%**, '
        'against 25% chance. All 256 output bytes remain eligible; the task answers are the four bytes A-D.')
    add('\n---\n\n## Compiled inference cost\n')
    add('Compilation absorbs the frozen normalization into the weights and bias. Identity corrections '
        'are multiplied out and added to the base matrix. Full-rank interpolation is likewise folded '
        'offline. GELU keeps its two correction matrices and nonlinearity, while normalization, strength '
        'and output bias are folded. The resulting modules contain frozen buffers and no trainable parameters.')
    add('\n| Compiled architecture | Stored scalars | Float64 bytes | Matrix MACs / case | Matrix multiplies |\n|---|---:|---:|---:|---:|')
    add('| Base, full-rank or identity | 16,640 | 133,120 | 16,384 | 1 |')
    add('| Active GELU correction plus base | 21,776 | 174,208 | 21,504 | 3 |')
    add('\nThe GELU row also evaluates 16 GELU elements. A zero-strength correction compiles to the base alone. '
        'Stored-scalar counts exclude Python object overhead; MACs count matrix multiply-accumulates, not '
        'biases, checks, argmax or allocation. Folding the identity correction reduces matrix MACs by '
        '23.8% and removes the separately stored correction and normalization arrays.')
    add('\n**Measured CPU prediction latency.** Values are medians across twelve graphs, in microseconds '
        'per batch. Each graph uses three timing repetitions with alternating reference/compiled order. '
        'The identity and GELU rows below force strength one, so neither benefits from fallback to the base.')
    add('\n| Head | Reference, batch 1 | Compiled, batch 1 | Reference, batch 64 | Compiled, batch 64 |\n|---|---:|---:|---:|---:|')
    for label, family in [('Base', 'base'), ('Full-rank, selected', 'linear_preserved_selected'),
            ('Identity, strength 1', 'identity_full'), ('GELU, strength 1', 'gelu_full')]:
        g = b['groups'][family]
        add(f"| {label} | {g['1']['reference']:.2f} | {g['1']['compiled']:.2f} | {g['64']['reference']:.2f} | {g['64']['compiled']:.2f} |")
    ig = b['groups']['identity_full']
    add(f"\nThe paired median identity speedup is **{ig['1']['median_paired_speedup']:.2f}x at batch 1** "
        f"and **{ig['64']['median_paired_speedup']:.2f}x at batch 64**. These timings cover the output decoder "
        'including input conversion, finite checks and argmax; they exclude the seven-node graph. '
        'They are not end-to-end model speedups.')
    add(f"\nTiming environment: {b['runtime']['cpu']}; PyTorch {b['runtime']['torch']}; one CPU thread; "
        f"{b['settings']['input_dtype']} inputs and float64 decoder arithmetic. No fitting or test suite "
        'was running alongside the recorded timing pass. Selected-head timings and all repetitions are retained in benchmark.json.')
    add('\n---\n\n## Verification and delivery\n')
    fits = [f for r in c['runs'] for f in r['fits'].values()]
    add(f"All **{v['automated_tests_passed']} automated tests pass** across five suites. The artifact audit "
        f"reconstructs all 72 new objectives and gradients, every validation decision, and {c['prediction_groups_exact']} "
        f"new prediction groups. It also re-audits the 432 inherited corrections and {c['inherited_prediction_groups_exact']} "
        'retained prediction groups, including their legacy baselines.')
    maximum = max(e['max_abs_logit_difference'] for r in c['runs'] for e in r['compiled_equivalence'].values())
    add(f"\nAll **204 compiled heads** are restored from their exported buffers and checked on both training "
        f"cohorts, validation and both evaluation cohorts: **{c['compiled_prediction_cases_exact']:,} exact predicted-byte "
        f"comparisons**. The largest logit difference from floating-point reassociation is {maximum:.3g}. "
        'These comparisons include repeated cases across heads and the nested training cohorts; they are not a count of unique episodes.')
    add(f"\nNone of the 72 new fits reached the 1,000-iteration cap. Iterations range from "
        f"{min(f['iterations'] for f in fits)} to {max(f['iterations'] for f in fits)}; final maximum gradient "
        f"magnitudes range from {min(f['gradient_max'] for f in fits):.3g} to {max(f['gradient_max'] for f in fits):.3g}. "
        'Stopping below the cap includes the optimizer\'s objective-change criterion; it is not a claim that every fit met the gradient tolerance.')
    add('\nSource hashes, cached inputs, graph checkpoints, frozen normalization, split separation and '
        'exact nested feature ownership are verified. Every completed fit is saved atomically; resuming '
        'with changed code, settings, inputs or baseline is rejected. Public outputs contain code, reports '
        'and aggregate results. Raw synthetic predictions and tensors are included in the downloadable artifact package.')
    add('\n## Reproduce and use\n')
    add('```bash\npython -m pip install -r requirements.txt\npython scripts/check_all.py\n'
        'python -m experiments.text_regions.full_rank_compare \\\n  --out artifacts/reproduced_full_rank_comparison.json\n```')
    add('\nFor new fits, use scripts/run_full_rank_study.py with a new --out directory. Add --resume '
        'only to continue that run. Use --export-compiled with full_rank_compare for a new export. '
        'scripts/benchmark_full_rank_readout.py repeats the decoder timings. The artifact ZIP includes '
        'all dependencies required for the audit; the source ZIP contains code, reports and aggregates.')
    add('\n## Next development step\n')
    add('Retain the compact identity correction as the leading low-cost decoder candidate, and use its '
        'compiled affine form in the next integration experiment. Train/select on the existing train/validation '
        'split, lock the choice, then evaluate once on a new template and fact-combination cohort. '
        'That will test whether the current gains carry beyond the fixed evaluation set used in these decoder phases. '
        'The full-rank phase supplies the comparison, export and timing tools needed for that step.')
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=Path('experiments/text_regions/results/full_rank'))
    p.add_argument('--out', type=Path, default=Path('docs/SHoEbox_Full_Rank_Readout_Report.md'))
    args = p.parse_args(); (ROOT/args.out).write_text(report(ROOT/args.results))


if __name__ == '__main__':
    main()
