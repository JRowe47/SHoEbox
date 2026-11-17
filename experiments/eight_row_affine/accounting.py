"""Collect complete correctness-site counters and non-overlapping CPU receipts."""
import argparse,collections,gzip,hashlib,json,struct
from pathlib import Path

SEMANTIC=('input_bytes','padding_bytes','nuisance_bytes','native_target_calls','native_update_targets','affine_rows','local_gradient_rows')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def collect(work,out):
    out.mkdir(parents=True,exist_ok=True);groups={};processes={};ledger=out/'events.jsonl.gz';checkpoint_kernel=collections.Counter()
    with gzip.open(ledger,'wt',compresslevel=6) as full:
        for process in sorted(work.glob('*/process.json')):
            folder=process.parent;name=folder.name;receipt=json.loads(process.read_text());processes[str(process.relative_to(work))]=receipt
            total=collections.Counter();functions=collections.Counter();builds=[];dump_count=0;site_count=0;explicit_zero=0
            for build in sorted((folder/'builds').glob('build_*')):
                info=json.loads((build/'compile.json').read_text());sites=json.loads((build/'sites.json').read_text())
                dumps=sorted((folder/'events').glob('*-'+build.name+'.bin'));builds.append(dict(build=build.name,**info,event_dumps=len(dumps)))
                for dump in dumps:
                    raw=dump.read_bytes();assert len(raw)==8*len(sites),(dump,len(raw),len(sites));values=struct.unpack('<'+'Q'*len(sites),raw)
                    dump_count+=1;site_count+=len(sites)
                    for site,value in zip(sites,values):
                        total[site['kind']]+=value;explicit_zero+=value==0
                        if name=='checkpoints' and any(k+'(' in (site['function'] or '') for k in ('a10_affine','a11_affine')):
                            checkpoint_kernel[site['kind']]+=value
                        if site['kind']=='function':functions[site['function']]+=value
                        full.write(json.dumps(dict(control=name,build=build.name,dump=dump.name,site=site,count=value),separators=(',',':'))+'\n')
            groups[name]=dict(exit_code=receipt['exit_code'],event_dumps=dump_count,registered_site_observations=site_count,
                explicit_zero_observations=explicit_zero,events=dict(total),function_calls=dict(functions),
                compiler_cpu_seconds=sum(x['compiler_cpu_seconds'] for x in builds),wrapper_cpu_seconds=sum(x['wrapper_cpu_seconds'] for x in builds),
                builds=builds,python_event_files={str(p.relative_to(work)):sha(p) for p in (folder/'python_events').glob('*.json')})
    for p in sorted((work/'bookkeeping').glob('*_process.json')):processes[str(p.relative_to(work))]=json.loads(p.read_text())
    totals=collections.Counter()
    for group in groups.values():totals.update(group['events'])
    checkpoint=work/'checkpoints/trajectories/result.json'
    extra={};cp=None
    if checkpoint.exists():
        cp=json.loads(checkpoint.read_text());assert cp['passed'] and cp['discarded_correctness_input_bytes']==216000
        observed=groups['checkpoints']['events'];assert observed['input_bytes']==108000
        # The 108k plain-library inputs are paired step-for-step with counted
        # execution. Only numeric-work events are doubled; serialization and
        # asymmetric Python bookkeeping are retained as actually observed.
        extra={key:observed[key] for key in SEMANTIC}
        assert observed['native_target_calls']*2==cp['totals']['targets']
    charged={k:totals[k]+extra.get(k,0) for k in SEMANTIC}
    # The first completed affine control predates the extra semantic labels,
    # but already has the exhaustive function/loop registry. Recover its exact
    # row totals from the unchanged registered generator and verify call counts.
    affine=groups['affine_retry']['function_calls']
    assert sum(v for k,v in affine.items() if k.startswith('float reference('))==71680
    for name in ('a10_affine','a11_affine'):
        assert sum(v for k,v in affine.items() if name+'(' in k)==5760
    affine_rows=35840*4*2-groups['affine_retry']['events'].get('affine_rows',0)
    assert affine_rows>=0
    charged['affine_rows']+=affine_rows
    python_checks={}
    for n in ('python_scalar','python_a10','python_a11'):
        p=work/n/'result.json'
        if p.exists():
            r=json.loads(p.read_text());expected=sum(x['input_bytes'] for x in r['updates'].values())+sum(r['observed_bytes'].values())
            assert groups[n]['events']['input_bytes']==expected,(n,expected)
            python_checks[n]=dict(expected_inputs=expected,actual_inputs=groups[n]['events']['input_bytes'],passed=True)
    if len(python_checks)==3:
        for key in SEMANTIC:assert len({groups[n]['events'][key] for n in python_checks})==1,key
    result=dict(research_fitting_bytes=0,accuracy_rescoring=False,audit_episodes=0,pilots=0,sweeps=0,
        extra_inference_policy_experiments=0,groups=groups,process_receipts=processes,
        whole_child_process_cpu_seconds=sum(x['cpu_seconds'] for x in processes.values()),
        cpu_rule='Each controlled child tree once. Compiler and wrapper costs are included subsets, never added twice. Later delivery receipts are separate additions.',
        counted_native_event_totals=dict(totals),uncounted_exact_peer_numeric_work=extra,charged_numeric_events=charged,
        affine_generator_projection_rows_added=affine_rows,
        checkpoint_kernel_events_counted=dict(checkpoint_kernel),
        checkpoint_kernel_events_with_exact_plain_peers={k:2*v for k,v in checkpoint_kernel.items()},
        python_independent_count_checks=python_checks,checkpoint_correctness=cp,
        affine=dict(original_seed=10107,trials=320,row_counts=[8,16],rows_per_path_per_build=35840,builds=2,
          paths=['independent_scalar','original_affine','A10','A11'],raw_log_comparison_groups_per_build=71680,
          pairwise_float_bit_comparisons_per_build=215040,model_inputs=0,model_targets=0),
        scratch=dict(automatic_block_object_events='Every nonstatic nonreference block declaration in project C++ is recorded with sizeof storage. Aggregate bytes are repeated activations, not resident memory.',
          register_vector_source_bound_bytes=544,retained_local_value_buffer_bytes=32,persistent_packing_bytes=0,
          generated_stack='production_retry/stack.so-model.su: complete generated frames, including spills; identical .text to timing library.',
          counted_checkpoint_kernel_activations_include_equal_plain_peer=True),
        event_ledger_sha256=sha(ledger),
        legacy_disclosure='A10 individual inherited microtest event counts remain incomplete in its immutable ledger. A11 instruments those sources before execution; no retrospective invented zero or replacement A10 run.',
        accounting_scope='Dynamic project functions, loops, comparison expressions and block-storage events, plus Python API/array/assertion call ledgers. Not an instruction-count estimate; standard-library internals are not relabeled as project events.')
    (out/'ACCOUNTING.json').write_text(json.dumps(result,indent=2)+'\n');return result
def main():
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('out',type=Path);a=p.parse_args();r=collect(a.work,a.out)
    print(json.dumps(dict(controls=len(r['groups']),cpu_seconds=r['whole_child_process_cpu_seconds'],numeric_events=r['charged_numeric_events']),indent=2))
if __name__=='__main__':main()
