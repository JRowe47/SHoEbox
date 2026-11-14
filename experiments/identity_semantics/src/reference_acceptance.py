"""Shared, fail-closed acceptance contracts for the enabled native reference.

Counters distinguish delivered credit, eligible update calls and measured
parameter changes (the isolated native tests check the last of these).
"""
import math

REGIONS = ('A1','B1','L2','S2','R2','F2','L3','S3','R3','F3',
           'E4','F4','V4','M4','G5','EC-C','EC-X','Y')
ZERO_COUNTERS = {
    'lace': ('missing_parent','overwritten_credit'),
    'messages': ('missing_feedback_producers',),
    'neural': ('missing_live_dependencies',),
    'local_credit': ('missing_producers',),
    'content_experts': ('missing_producers',),
    'composition': ('expired_input_edges','overwritten_input_credit'),
    'completion': ('expired_semantic_handles','semantic_budget_omissions','semantic_capacity_prunes'),
    'teaching': ('expired_producers','duplicate_schedules'),
    'acceptance': ('invalid_primary_spans','invalid_coverage_spans','primary_outside_coverage'),
}


def require(condition, message):
    # Deliberately not `assert`: acceptance must still work under python -O.
    if not condition:
        raise AssertionError(message)


def collect(model):
    primary_edges=[]
    for r in range(3,15):
        slots=model.fixed['slots'][model.fixed['starts'][r]:model.fixed['starts'][r+1]]
        immediate=[slot for slot in slots if not slot[2]]
        driver=int(immediate[0][0])
        if sum(int(slot[0])==driver for slot in immediate)>=2:
            primary_edges.append((REGIONS[driver],REGIONS[r]))
    from memory_inspection import read_memory_index
    return dict(memory_index=read_memory_index(model) if hasattr(model.lib,'inspect_live_memory') else None,
                capabilities=model.capabilities(), composition=model.composition_metrics(),
                completion=model.completion_metrics(), teaching=model.relay_teaching_status(),
                lace=model.lace_metrics(), messages=model.message_metrics(), neural=model.neural_metrics(),
                local_credit=model.local_credit_metrics(), content_experts=model.content_expert_metrics(),
                pattern_classes=model.pattern_metrics(), paired_memory=model.paired_memory_metrics(),
                acceptance=model.acceptance_metrics(),primary_edges=primary_edges,
                event=model.event_metrics() if hasattr(model,'event_metrics') else {})


def check_integrity(row):
    """No activity assumptions: suitable for short evaluation and continuations."""
    if not row['capabilities'].get('inline_diagnostics',True):
        require(row.get('memory_index') is not None,'Missing memory inspection')
        for region,v in row['memory_index'].items():
            require(v['invalid_retained_spans']==0,f'{region}: invalid retained span')
            require(v['missing_content_records']==0,f'{region}: missing content record')
        return True
    for group, fields in ZERO_COUNTERS.items():
        require(set(row[group]) == set(REGIONS), f'{group}: missing regional diagnostics')
        for region, values in row[group].items():
            for field in fields:
                require(values[field] == 0, f'{group}.{region}.{field}={values[field]}')
    for region,v in row.get('event',{}).items():
        require(v['missing']==0,f'event.{region}.missing={v["missing"]}')
    for region, v in row['acceptance'].items():
        require(v['recognized'] + v['provisional'] == v['formed'], f'{region}: recognition accounting')
        require(v['formed'] + v['relays'] <= v['publications'], f'{region}: publication accounting')
        for key in ('primary_span_histogram', 'coverage_span_histogram'):
            require(len(v[key]) == 65 and sum(v[key]) == v['publications'], f'{region}: {key} accounting')
            expected_zero = v['unclocked_fusions'] if key == 'primary_span_histogram' else 0
            require(v[key][0] == expected_zero, f'{region}: zero-span publication')
        require(v['unclocked_fusions'] == 0 or region in ('EC-C','EC-X','Y'), f'{region}: unclocked temporal unit')
        lengths, recognized = v['sequence_length_histogram'], v['recognized_sequence_length_histogram']
        require(len(lengths) == len(recognized) == 9, f'{region}: sequence histogram shape')
        require(lengths[0] == 0 and recognized[0] == 0, f'{region}: invalid sequence length')
        require(all(0 <= b <= a for a,b in zip(lengths,recognized)), f'{region}: sequence recognition accounting')
        expected = v['publications'] - v['relays'] if region in REGIONS[2:15] else 0
        require(sum(lengths) == expected, f'{region}: local sequence accounting')
    return True


def check_hierarchy(regions,edges=(('L2','L3'),('F3','F4'),('S3','E4'),('E4','G5'))):
    """A deterministic repeated-pattern workload, using primary clocks only."""
    means = {r:v['primary_span_sum']/max(1,v['publications']-v.get('unclocked_fusions',0)) for r,v in regions.items()}
    # Compare actual primary-driver edges with >=2 distinct occurrences. Side
    # connections are not temporal parents; G5's primary source is E4, not F4.
    require(bool(edges),'No repeated primary-source edges were checked')
    for lower,upper in edges:
        require(regions[upper]['publications'] > 0, f'{upper}: no publications')
        require(means[upper] > means[lower], f'{lower}->{upper}: primary span did not grow')
        require(regions[lower]['publications'] > regions[upper]['publications'],
                f'{lower}->{upper}: publication cadence did not slow')
    return means


def objective_deliveries(row):
    """Language/DFA transport and local prediction transport are separate."""
    out = {}
    for r in REGIONS:
        a,local = row['acceptance'][r],row['local_credit'][r]
        out[r] = {
            'language_dfa': dict(neural_deliveries=row['lace'][r]['credit_edges'],
                                 feedback_deliveries=row['messages'][r]['feedback_deliveries'],
                                 retained_input_deliveries=row['composition'][r]['input_credit_received'],
                                 expert_update_calls=a['language_dfa_update_calls'],
                                 content_updates=row['content_experts'][r]['task_updates']),
            'local_prediction': dict(objectives=local['objectives'],neural_deliveries=local['neural_deliveries'],
                                     feedback_deliveries=local['feedback_deliveries'],
                                     retained_input_deliveries=local['input_deliveries'],
                                     expert_update_calls=a['local_update_calls'],
                                     content_updates=row['content_experts'][r]['local_updates']),
        }
    return out


def check_training(row):
    """Strict coverage for the fixed repeated-text reference acceptance workload."""
    check_integrity(row)
    require(row['capabilities'].get('inline_diagnostics',True),'Historical coverage requires an explicit SHOEBOX_INLINE_DIAGNOSTICS=1 audit build')
    require(row['capabilities']['reference_profile'] == 'reference-v1', 'Full reference policy required')
    require(all(row['capabilities']['features'].values()), 'A reference feature is disabled')
    for r,v in row['acceptance'].items():
        require(v['publications'] > 0, f'{r}: unexercised publication path')
    means = check_hierarchy(row['acceptance'],row['primary_edges'])
    objectives = objective_deliveries(row)
    for objective in ('language_dfa','local_prediction'):
        for field in ('neural_deliveries','feedback_deliveries','retained_input_deliveries',
                      'expert_update_calls','content_updates'):
            require(sum(v[objective][field] for v in objectives.values()) > 0,
                    f'{objective}.{field}: path not exercised')
    for region in ('L3','S3','F3','E4','F4','G5'):
        require(row['local_credit'][region]['objectives'] > 0, f'{region}: no local objective')
    for field in ('reused','promoted','class_hits','forecast_hits'):
        require(sum(v[field] for v in row['pattern_classes'].values()) > 0, f'pattern_classes.{field}: no evidence')
    require(sum(sum(v['recognized_sequence_length_histogram'][2:]) for v in row['acceptance'].values()) > 0,
            'No recognized multi-frame temporal publication')
    for field in ('admissions','hits','utility_credit'):
        require(row['paired_memory'][field] > 0, f'paired_memory.{field}: path not exercised')
    return dict(primary_span_means=means, credit_by_objective=objectives)


def check_contribution(row, *, require_effect=False, preserve_events=False):
    for key in ('baseline_nll','intervention_nll','delta_nll','logit_rms'):
        require(math.isfinite(row[key]), f'Nonfinite contribution {key}')
    require(row['applications'] > 0, 'Contribution control was not exercised')
    require(row['fitted_unchanged'] and row['source_unchanged'], 'Contribution mutated fitted/source state')
    if require_effect:
        require(row['logit_rms'] > 0, 'Exercised contribution control had no downstream effect')
    if preserve_events:
        require(row['events_identical'], 'Channel intervention changed event identity or primary clocks')
    # Positive delta means useful prediction. Negative/zero results are evidence,
    # not a mechanism failure and never promoted to a claim of improvement.
    row['effect'] = 'beneficial' if row['delta_nll'] > 0 else 'harmful' if row['delta_nll'] < 0 else 'undetected'
    return row
