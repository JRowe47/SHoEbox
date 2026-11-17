"""Exact data-only ledgers; no model weights are evaluated or updated."""
import argparse,json
from pathlib import Path
from .stream import advance,new_cursor
from .tasks import cases
from .credit import Counter,augmented
from experiments.joint_cold_gate.adapter import prefix_labels

def summed(rows):return {k:sum(r[k] for r in rows) for k in rows[0]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();w=a.workspace
    checks=json.loads((w/'python_counts.json').read_text());final=[dict(policy=k,counts=augmented(v,k)) for k,v in checks['updates'].items()]
    # Failed A9 correctness invocations, if any, have separately retained counters.
    initial=[]
    for path in sorted(w.glob("python_counts_failed*.json")):
        record=json.loads(path.read_text())
        for policy,counts in record["updates"].items():
            initial.append(dict(receipt=path.name,policy=policy,counts=augmented(counts,policy)))
    research=[];fits=json.loads((w/'a9/results.json').read_text());stream_rows=json.loads((w/'streams.json').read_text())['rows']
    for sr in stream_rows:
        arm=sr['arm'];warmup=sr['phase']=='warmup';m=Counter();cursor,_=advance(m,new_cursor(warmup,not warmup,arm),250000 if warmup else 2000000)
        assert json.loads(json.dumps(cursor))==sr['cursor'];assert m.total==sr['counts']
        for row in fits['rows']:
            fit=row['arms'][arm]['warmup'] if warmup else row['arms'][arm]
            assert json.loads(json.dumps(cursor))==fit['cursor']
            for k in ('local_projection_steps','local_presence_steps','gate_targets','local_training_windows'):
                assert m.total[k]==fit['inventory'][k]-fit['initial_inventory'][k]
            counts=augmented(m.total,arm);counts.update(nuisance_bytes=cursor['nuisance_bytes'],replaced_padding_bytes=cursor['replaced_padding_bytes'])
            research.append(dict(seed=row['seed'],arm=arm,phase=sr['phase'],counts=counts))
    resume=[]
    for path in sorted((w/'a9').glob('*/*.qr')):
        cursor=json.loads(path.with_suffix('.cursor.json').read_text());m=Counter();m.n=cursor['model_bytes'];end,_=advance(m,cursor,1000)
        counts={k:2*v for k,v in augmented(m.total,path.name.split('.')[0]).items()}
        counts.update(nuisance_bytes=2*(end['nuisance_bytes']-cursor['nuisance_bytes']),replaced_padding_bytes=2*(end['replaced_padding_bytes']-cursor['replaced_padding_bytes']))
        resume.append(dict(file=str(path.relative_to(w/'a9')),counts=counts))
    result=dict(rows=final,initial_failed_rows=initial,totals=summed([r['counts'] for r in final+initial]),
        research=research,research_totals=summed([r['counts'] for r in research]),resume=resume,resume_totals=summed([r['counts'] for r in resume]),
        python_observed_bytes=checks['observed_bytes'],rejected_calls=checks['rejected_calls'],
        method='Successful Python updates instrumented directly; failed invocations retained directly; formal/cursor streams exactly replayed without weights. Native independent microchecks retain counts in stdout; historical microtrajectory scalar writes uninstrumented, full CPU charged.')
    result['objective_microchecks']=dict(python_native_calls=256,python_independent_calls=256,python_code_comparisons=131072,native_calls_per_build=1029,native_independent_reference_calls_per_build=768,native_code_comparisons_per_build=460032,native_finite_difference_coordinates_per_build=256,native_actual_affine_gradient_rows_per_build=4176,native_expected_affine_gradient_rows_per_build=4096,builds=2)
    result['presence_microchecks']=dict(python_logistic_events=650,python_finite_difference_coordinates=642,native_logistic_events_per_build=812,native_finite_difference_coordinates_per_build=802,native_local_calls_per_build=640,native_candidate_record_positive_rows_per_build=256,native_candidate_record_negative_rows_per_build=256,native_reference_record_positive_rows_per_build=256,native_reference_record_negative_rows_per_build=256,native_query_presence_rows_per_build=512,native_applicability_rows_per_build=512,native_hybrid_components_per_build=768,native_hybrid_code_comparisons_per_build=196608,native_hybrid_gradient_rows_per_build=12288,native_value_gradient_rows_per_build=4096,independent_expected_record_rows_per_build=1024,builds=2)
    result['codebook_disclosure_comparisons']=32640
    result['initial_failed_inference_bytes']=sum(sum(json.loads(p.read_text()).get('observed_bytes',{}).values()) for p in w.glob('python_counts_failed*.json'))
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('totals','research_totals','resume_totals')},indent=2))

if __name__=='__main__':main()
