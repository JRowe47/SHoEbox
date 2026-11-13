"""Render the contribution contract, complete regional comparison and evidence."""
import csv
import json
from pathlib import Path
import statistics
from report_relays import render_pdf

ROOT=Path(__file__).resolve().parents[1]
NAMES=('A1','B1','L2','S2','R2','F2','L3','S3','R3','F3','E4','F4','V4','M4','G5','EC-C','EC-X','Y')


def number(value):
    if value==0:return '0'
    return f'{value:+.5f}' if abs(value)>=.0001 else f'{value:+.2e}'


def main():
    checks=json.loads((ROOT/'results/contribution_checks.json').read_text())
    credit=json.loads((ROOT/'results/contribution_credit.json').read_text())
    speed=json.loads((ROOT/'results/contribution_throughput.json').read_text())
    verification=json.loads((ROOT/'results/contribution_verification.json').read_text())
    base={(r['seed'],r['memory']):r for r in checks['baselines']}
    rows={(r['seed'],r['kind'],r['name']):r for r in checks['interventions'] if r['memory']=='cold_adaptive'}
    mean=lambda kind,name:statistics.mean(rows[s,kind,name]['delta_nll'] for s in (1,2))
    table='\n'.join('| '+name+' | '+' | '.join(number(mean(k,name)) for k in ('neural_output','expert_transform','content_descriptor','retained_input'))+' |' for name in NAMES)
    memories='\n'.join(f"| {policy} | {base[1,policy]['nll']:.6f} | {base[2,policy]['nll']:.6f} | {statistics.mean(base[s,policy]['nll'] for s in (1,2)):.6f} |" for policy in ('cold_adaptive','cold_frozen','warm_frozen','warm_adaptive'))
    teaching=[]
    for name in ('L2','L3','E4','F4','V4','G5'):
        r=next(r for r in credit['regions'] if r['seed']==1 and r['region']==name)
        t=r['teaching'];c=r['lace']
        teaching.append(f"| {name} | {r['neural_executions']:,} | {t['neural_targets']+t['relay_targets']:,} | {int(c['credit_edges']):,} | {int(c['missing_parent']):,} |")
    spans=[]
    for name in ('L2','L3','E4','F4','G5','Y'):
        events=[base[s,'cold_adaptive']['trace'][name] for s in (1,2)]
        avg=sum(e['span_sum'] for e in events)/sum(e['publications'] for e in events)
        spans.append(f"| {name} | {avg:.2f} | {max(e['max_span'] for e in events)} |")
    med=speed['median_bytes_per_second']
    perf='\n'.join(f"| {label} | {med[before]:,.0f} | {med[after]:,.0f} | {100*(med[after]/med[before]-1):+.1f}% |" for label,before,after in (
        ('Production training','before_periodic','after_periodic'),
        ('Production evaluation','before_eval','after_eval'),
        ('Diagnostic evaluation vs production','after_eval','diagnostic_eval')))
    assert all(b['production_parity'] for b in checks['baselines'])
    assert all(r['event_stream_preserved'] for r in checks['interventions'] if r['kind']!='whole_source')
    assert all(r['fitted_state_unchanged'] and r['source_state_unchanged'] for r in checks['interventions'])
    out=ROOT/'deliverables';out.mkdir(exist_ok=True)
    with (out/'SHoEbox_Contribution_Checks.csv').open('w',newline='') as f:
        fields=['seed','memory','kind','name','baseline_nll','nll','delta_nll','targets','event_stream_preserved','intervention_applications']
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in checks['interventions'])
    text=f'''# SHoEbox: regional contribution checks

13 November 2026 | Fourth step of the composition repair | 1,031,936 parameters

## Result

Contribution checks now distinguish local learned computation from content transport, relay context and retained inputs. They are supported per-model evaluation controls, replacing the need for global selectors or source-text substitutions. Production remains the relays kernel, with unchanged learning behavior and version-31 / ABI-5 checkpoints.

The two existing million-byte-trained models reproduce their published reserved-test losses: **{base[1,'cold_adaptive']['nll']:.6f}** and **{base[2,'cold_adaptive']['nll']:.6f}**. We completed **{len(checks['interventions'])} interventions**, each over all **78,771 test targets**, plus eight baseline comparisons against production. Neural parameters, semantic weights, dictionaries and source checkpoints remain unchanged.

The learned expert transforms in **L3** and **F4** contribute measurably: removing them increases mean NLL by **{mean('expert_transform','L3'):.6f}** and **{mean('expert_transform','F4'):.6f}** nats/byte. E4, V4 and G5 remain weak on this test despite substantial realized teaching and credit. This completes the measurement repair; it identifies where prediction still makes little use of the upper regions.

## What each intervention measures

| Control | Contract |
| --- | --- |
| Neural output | Zero only neural publications, after the producer's private memory update. Preserve singleton relays and content. This includes the input skip and EC recall in the removed output. |
| Expert transform | Remove only the learned matrix/bias term. Keep input skip, recall, routing and relay handling. |
| Content descriptor | Zero outgoing descriptor/code, including relays. Preserve exact IDs, recognition, timing, context and provenance. Parent ID anchors still carry identity information. |
| Relay context | Zero context only on singleton relay publications at the receiving region. Preserve code, descriptor and intrinsic structure. |
| Retained input | Suppress previous-input carry; retain the current input coefficient. Bottom learned-memory reads are separate. |
| Semantic residual | Remove the local learned residual when a symbol is encoded; retain the structural descriptor and inherited relay codes. |
| Whole source | Remove source inputs and direct A1/Y decoder features. This intentionally conflates transport with computation and can change event activity. |

Positive delta NLL means the intervention damaged prediction. Effects are conditional and non-additive. The current task measures use of the learned paths on held-out text; it is not a retraining comparison or a claim that every upper region now has a useful abstraction.

<!-- pagebreak -->

## Complete regional comparison

Mean delta NLL across seeds 1 and 2, under cold memory with causal admission. Neural and expert columns retain constituent descriptors and relay context. Very small effects are shown explicitly rather than ranked as improvements.

| Region | Neural output | Expert transform | Descriptor | Retained input |
| --- | ---: | ---: | ---: | ---: |
{table}

The distinction changes interpretation. For L3, whole-source removal changes mean NLL by **{mean('whole_source','L3'):+.6f}**, compared with **{mean('neural_output','L3'):+.6f}** for its neural publication alone. Whole-source removal includes the fixed content transported through L3.

Descriptor effects at G5, EC-X and Y are zero because their outgoing descriptors have no immediate constituent consumers in this graph; their neural context is a different path. These zeros do not diagnose the usefulness of their local input codes. Parent descriptors deliberately use child structural descriptors, not child learned residuals.

The learned local symbol residual is most consequential at Y: removing it changes mean NLL by **{mean('semantic_residual','Y'):+.6f}**. At L3 it changes NLL by **{mean('semantic_residual','L3'):+.6f}**, and at F4 by **{mean('semantic_residual','F4'):+.6f}**. The CSV and JSON include every semantic-residual and whole-source result.

L2 is the only region publishing ordinary singleton relays in this test. Muting only that relay context changes mean NLL by **{mean('relay_context','L2'):+.6f}**. The other 34 seed/region relay cases are explicitly marked inactive, not reported as measured zero effects.

<!-- pagebreak -->

## Teaching, credit and event scale

Saved seed-1 training counters, read without additional training. Both seeds account for every forecast with zero expired producers or duplicate schedules. EC-C/EC-X use their memory-port teaching path and intentionally do not schedule this local relay teacher.

| Region | Neural executions | Local targets | Credit edges | Missing parents |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(teaching)}

Credit counts establish delivery, not benefit. E4/V4/G5 are no longer explained by the earlier canceled forecasts or near-total relay execution. Their weak expert-transform effects point to low task use of these learned responses. The next useful architectural investigation is how their learned messages reach prediction and what target makes them distinct from the stronger lower paths.

Completed publication spans in the reserved test, pooled over both seeds:

| Region | Mean span, bytes | Largest span, bytes |
| --- | ---: | ---: |
{chr(10).join(spans)}

These are explicit event spans, not bounds on recurrent memory or measures of conceptual abstraction. Fixed recipe names and nominal cadences alone cannot establish the realized event scale.

## Explicit memory evaluation

| Memory policy | Seed 1 NLL | Seed 2 NLL | Mean NLL |
| --- | ---: | ---: | ---: |
{memories}

Cold starts with an empty associative store; warm retains the checkpoint store, its coordinate centers and timestamp clock. Frozen blocks store writes; adaptive admits causal associations from the observed test prefix. Every mode starts fresh recurrent/event state and freezes neural parameters and dictionaries. Query normalization still evolves during inference. The historical frozen_copy behavior remains cold_adaptive. EC-C/Y neural interventions are also evaluated under every alternative memory policy in the JSON/CSV.

<!-- pagebreak -->

## Verification and performance

All **{verification['package_tests']} package tests** pass, including {verification['contribution_tests']} contribution tests. Across the repository and final package checks, **{verification['repository_unit_tests']} unit tests** pass, plus 12/64-expert packaging checks. ASan/UBSan pass for both the ordinary relay runtime and every diagnostic intervention/region. Tests cover independent selectors, invalid modes, unsavable diagnostic models, channel isolation, exact chunking, source preservation, explicit memory lifecycles, document draining and complete disabled-control parity.

The diagnostic hashes every publication's identity, span, source roles, occurrence, intrinsic structure, recognition and producer provenance. Every non-whole-source intervention preserves that full trace and execution counts in all measured runs. Frozen store fingerprints are unchanged; fitted neural, semantic and dictionary states are unchanged for every intervention.

Six alternating same-core trials per condition, after other workers finished. Each uses 1,024 warmup and 16,384 timed targets. Comparison source is the unchanged preceding commit f9eeb8a. Compilation, loading, snapshots and checkpoint I/O are excluded.

| Execution | Before, bytes/s | After, bytes/s | Change |
| --- | ---: | ---: | ---: |
{perf}

All timed production losses and disabled-diagnostic losses match exactly within their respective training/evaluation modes. Diagnostic tracing adds evaluation work; production compiles out its controls and counters. No neural parameters or serialized recurrent-state fields were added.

## Reproduce

Use the two relay-repair checkpoints, or regenerate them with the preceding study. The corpus split and checkpoint SHA-256 values are recorded in results/contribution_checks.json. No test models or intermediate run snapshots accompany this publication.

```bash
python -m unittest discover -s tests -v
python tests/run_contribution_sanitizer.py
python tests/run_relays_sanitizer.py
python src/run_contribution_checks.py --workers 6
python src/contribution_credit_audit.py
python src/contribution_performance.py \\
  --baseline-runtime /path/to/f9eeb8a/runtime
python src/report_contributions.py
```

Source base: https://github.com/JRowe47/SHoEbox/commit/{checks['base_commit']}
'''
    md=out/'SHoEbox_Contribution_Checks_Report.md';md.write_text(text)
    render_pdf(md,md.with_suffix('.pdf'),title='SHoEbox: regional contribution checks',
               header='SHoEbox / Regional contribution checks / 1M model')
    print(json.dumps(dict(interventions=len(checks['interventions']),L3_expert_delta=mean('expert_transform','L3'),F4_expert_delta=mean('expert_transform','F4'))))


if __name__=='__main__':main()
