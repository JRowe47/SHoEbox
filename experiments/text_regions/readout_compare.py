"""Audit portable frozen-readout results and aggregate twelve matched models."""
import argparse
from collections import Counter
import hashlib
import gzip
import json
import math
from pathlib import Path
import statistics

from .readout_diagnostic import EVAL_CONTROLS, FACTORS, SUBSETS, controlled_batch, visible
from .view_eval import accuracy_summary


HEADS = ('direct','mixed','raw')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def audit_predictions(report, predictions):
    """Rescore every answer using saved argmax bytes, independently of summaries."""
    expected = [(split,g) for split,r in report['evaluations'].items() for g in r['groups']]
    require(len(expected) == len(predictions), 'Prediction group inventory differs.')
    scores, identities, answers = {}, {}, {}
    for (split,group),raw in zip(expected,predictions):
        key = (split,group['control'],group['query'])
        require(key == (raw['split'],raw['control'],raw['query']), 'Prediction ordering/condition differs.')
        n = group['stop']-group['start']
        require(len(raw['answers']) == len(raw['episode_ids']) == n, 'Prediction cohort length differs.')
        require(len(set(raw['episode_ids'])) == n, 'Duplicate prediction identities.')
        require(set(raw['predictions']) == set(HEADS), 'Common decoder inventory differs.')
        require(Counter(raw['answers']) == {v:n//4 for v in b'ABCD'}, 'Evaluation answers are not balanced.')
        same_case = (split,group['query'])
        require(same_case not in identities or identities[same_case] == raw['episode_ids'], 'Controls changed cases.')
        require(same_case not in answers or answers[same_case] == raw['answers'], 'Controls changed answers.')
        identities[same_case],answers[same_case] = raw['episode_ids'],raw['answers']
        for head in HEADS:
            predicted = raw['predictions'][head]
            require(len(predicted) == n and all(type(v) is int and 0 <= v <= 255 for v in predicted), 'Invalid byte predictions.')
            computed = accuracy_summary(predicted,raw['answers'])
            saved = group['decoders'][head]
            require(all(saved.get(k) == v for k,v in computed.items()), 'Saved answer score differs from raw predictions.')
            require(saved['outside_answer_vocabulary'] == sum(v not in b'ABCD' for v in predicted), 'Out-of-vocabulary count differs.')
            require(math.isfinite(saved['answer_nats_per_byte']) and saved['answer_nats_per_byte'] >= 0, 'Invalid recorded cross entropy.')
            scores['/'.join((*key,head))] = computed['exact_accuracy']
    return scores


def expected_cohort(protocol):
    split,count = protocol['split'],protocol['count_per_cell']
    controls = tuple(SUBSETS) if split == 'train' else (('all',) if split == 'test' else EVAL_CONTROLS)
    cases,identities,groups = [],[],[]
    input_hash = hashlib.sha256()
    for control in controls:
        for query in FACTORS:
            if split == 'train' and not visible(control,query):
                continue
            batch = controlled_batch(split,0,count,control=control,query=query,
                **{k:protocol[k] for k in ('seed','gap','query_every')})
            require(Counter(batch.answers.tolist()) == {v:count//4 for v in b'ABCD'}, 'Reconstructed class coverage differs.')
            groups.append((control,query))
            for row,identity in enumerate(batch.identities):
                cases.append((control,identity));identities.append(identity)
                input_hash.update(json.dumps((control,query,identity)).encode())
                for source in 'ABC':
                    input_hash.update(bytes(batch.inputs[source][row].tolist()))
                    input_hash.update(bytes(batch.availability[source][row].tolist()))
                input_hash.update(bytes(batch.reset[row].tolist()))
    return {'case_sha256':digest(cases),'inputs_sha256':input_hash.hexdigest(),
            'fit_identities':sorted(set(identities)),'episodes':len(cases),'groups':groups}


def _check_factor_summary(score):
    counts = score['by_answer']
    require(set(counts) == set('ABCD'), 'Factor class inventory differs.')
    require(all(type(r[k]) is int for r in counts.values() for k in ('episodes','correct')), 'Invalid factor counts.')
    require(all(0 <= r['correct'] <= r['episodes'] for r in counts.values()), 'Invalid factor accuracy counts.')
    total,correct = sum(r['episodes'] for r in counts.values()),sum(r['correct'] for r in counts.values())
    require(score['episodes'] == score['valid_supported_episodes'] == total and score['exact_correct'] == correct,
            'Factor counts do not sum.')
    require(score['exact_accuracy'] == (correct/total if total else None), 'Factor accuracy arithmetic differs.')
    require(total <= score['total_episodes'], 'Factor validity exceeds cohort size.')


def compare(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent/manifest.get('base_directory','.')
    expected = {(p,s,seed) for p in ('original','gated') for s in ('dropout','balanced') for seed in (20261106,20261107,20261108)}
    require(len(manifest['runs']) == 12 and {(r['publication_policy'],r['source_schedule'],r['training_seed']) for r in manifest['runs']} == expected,
            'The study requires all twelve planned models.')
    references,records,file_manifest = {},[],[]
    for run in manifest['runs']:
        directory = base/run['directory']
        def load(name,lines=False):
            path = directory/name
            if not path.exists() and path.with_suffix(path.suffix+'.gz').is_file():
                path = path.with_suffix(path.suffix+'.gz')
            stored = path.read_bytes()
            content = gzip.decompress(stored) if path.suffix == '.gz' else stored
            file_manifest.append({'path':str(path.relative_to(base)),'size_bytes':len(stored),'sha256':hashlib.sha256(stored).hexdigest(),
                                  'decoded_size_bytes':len(content),'decoded_sha256':hashlib.sha256(content).hexdigest()})
            return [json.loads(line) for line in content.splitlines()] if lines else json.loads(content)
        report,predictions,inventory = load('results.json'),load('predictions.jsonl',True),load('fit_inventory.json')
        require(report.get('complete') is True and report['frozen_verified'] is True and report['checkpoint_step'] == 240, 'Incomplete frozen model diagnostic.')
        require(set(report['evaluations']) == {'test','independent'}, 'Missing evaluation cohort.')
        training = dict(report['training_protocol'])
        # The earliest original-policy checkpoint predates this explicit flag.
        training.setdefault('evidence_gated_ports',False)
        require(training['seed'] == run['training_seed'] and training['source_schedule'] == run['source_schedule'], 'Training provenance differs.')
        require(training['rounds'] == report['runtime']['rounds'] == 1, 'K must remain one.')
        require(training['evidence_gated_ports'] == (run['publication_policy'] == 'gated'), 'Publication policy differs.')
        require(report['fit_protocol']['policy']['evidence_gated_ports'] == training['evidence_gated_ports'],
                'Recorded policy differs from the actual frozen graph configuration.')
        require(training['view_objective'] and not any(training[k] for k in ('utility','baseline_detach_features','baseline_observed_only')),
                'Unexpected learning intervention.')
        require(inventory['fit_protocol'] == report['fit_protocol'] and inventory['model_tensor_sha256'] == report['model_tensor_sha256'],
                'Fitted heads belong to different features or model.')
        require(inventory['answer_decoder_fits'] == report['answer_decoder_fits'], 'Fitted byte head metadata differs.')
        require(report['fit_query_counts'] == {'shared':7*report['fit_protocol']['count_per_cell'],
                                              'left':4*report['fit_protocol']['count_per_cell'],
                                              'right':4*report['fit_protocol']['count_per_cell']}, 'Training query coverage differs.')
        require(report['answer_loss_weighting'] == 'equal query mass; uniform supported subsets within query', 'Query loss weighting differs.')
        cfg = report['fit_protocol']['policy']['graph_config']
        per_event = len(cfg['nodes'])*3*cfg['active_groups']*cfg['experts_per_group']
        require(cfg['rounds'] == 1 and cfg['width'] == 32 and len(cfg['readout']) == 2, 'Unexpected graph/readout dimensions.')
        for head,width in (('raw',64),('mixed',32)):
            fit = report['answer_decoder_fits'][head]
            require(fit['width'] == width and fit['output_bytes'] == 256 and fit['examples'] == report['fit_protocol']['episodes'], 'Decoder dimensions or fit count differs.')
            require(math.isfinite(fit['gradient_max']) and fit['gradient_max'] < 1e-5, 'Byte decoder optimization has not converged sufficiently.')
        for split,protocol in [('train',report['fit_protocol']),*[(s,r['protocol']) for s,r in report['evaluations'].items()]]:
            require(protocol['frozen_verified'] and protocol['model_tensor_sha256'] == report['model_tensor_sha256'], 'Model fingerprint changed across cohorts.')
            require(protocol['policy'] == report['fit_protocol']['policy'], 'Inference policy changed across cohorts.')
            signature = {k:protocol[k] for k in ('split','count_per_cell','seed','gap','query_every','batch_size','segment')}
            if split not in references:
                references[split] = {'settings':signature,'cohort':expected_cohort(protocol)}
            require(signature == references[split]['settings'], 'Diagnostic settings differ across models.')
            rebuilt = references[split]['cohort']
            require(all(protocol[k] == rebuilt[k] for k in ('case_sha256','inputs_sha256','episodes')), 'Cases or actual inputs fail independent reconstruction.')
            events = protocol['episodes']*(17+protocol['gap'])
            require(protocol['external_events'] == events and protocol['expert_executions'] == events*per_event, 'Graph work differs from fixed budget.')
            if split == 'train':
                require(inventory['fit_identities'] == rebuilt['fit_identities'], 'Fitted training identities differ.')
            else:
                evaluation = report['evaluations'][split]
                require([(g['control'],g['query']) for g in evaluation['groups']] == rebuilt['groups'], 'Evaluation group inventory differs.')
                for group in evaluation['groups']:
                    require(group['expert_executions'] == protocol['count_per_cell']*(17+protocol['gap'])*per_event, 'Group expert budget differs.')
                    require(group['information_present'] == visible(group['control'],group['query']), 'Information control annotation differs.')
                for conditions in evaluation['factor_decodability'].values():
                    for scores in conditions.values():
                        for score in scores.values():
                            _check_factor_summary(score)
        require(not set(inventory['fit_identities']) & {i for row in predictions for i in row['episode_ids']}, 'Fit/evaluation leakage.')
        rescored = audit_predictions(report,predictions)
        metrics = {}
        for head in HEADS:
            for split,control in (('test','all'),('independent','all'),('independent','b_only'),('independent','a_only'),('independent','c_only')):
                for factor in FACTORS:
                    metrics[f'{split}/{control}/{factor}/{head}'] = rescored[f'{split}/{control}/{factor}/{head}']
                metrics[f'{split}/{control}/overall/{head}'] = statistics.mean(rescored[f'{split}/{control}/{q}/{head}'] for q in FACTORS)
                metrics[f'{split}/{control}/private/{head}'] = statistics.mean(rescored[f'{split}/{control}/{q}/{head}'] for q in ('left','right'))
        probes = report['evaluations']['test']['factor_decodability']
        factor_metrics = {}
        for cut in ('prequery->prequery','final->final','prequery->final'):
            for feature,pairs in {'raw':(('raw','left'),('raw','right')),'mixed':(('mixed','left'),('mixed','right')),
                                  'source_nodes':(('A','left'),('C','right')),'output_nodes':(('P','left'),('Q','right'))}.items():
                values = [probes[f'{cut}/{node}/{factor}']['all']['query_matched']['exact_accuracy'] for node,factor in pairs]
                factor_metrics[cut+'/'+feature+'/private'] = statistics.mean(values)
        records.append({**run,'model_tensor_sha256':report['model_tensor_sha256'],'metrics':metrics,'factor_metrics':factor_metrics,
                        'training_protocol':training,'source_sha256':report['source_sha256'],
                        'fit_settings':{h:{k:report['answer_decoder_fits'][h][k] for k in ('l2','max_iter')} for h in ('raw','mixed')},
                        'fit_expert_executions':report['fit_protocol']['expert_executions'],
                        'evaluation_expert_executions':sum(r['protocol']['expert_executions'] for r in report['evaluations'].values())})
    reference = records[0]
    require(all(r['source_sha256'] == reference['source_sha256'] and r['fit_settings'] == reference['fit_settings'] for r in records), 'Runtime source or decoder fitting differs.')
    for policy in ('original','gated'):
        for seed in (20261106,20261107,20261108):
            pair = [r for r in records if r['publication_policy'] == policy and r['training_seed'] == seed]
            normalized = []
            for row in pair:
                protocol = dict(row['training_protocol'])
                protocol.pop('source_schedule');protocol.pop('dropout')
                if protocol['base_initializer']:
                    protocol['base_initializer'] = {'sha256':protocol['base_initializer']['sha256']}
                normalized.append(protocol)
            require(normalized[0] == normalized[1], 'Paired model training differs beyond source availability.')
    groups = {}
    for policy in ('original','gated'):
        for schedule in ('dropout','balanced'):
            cohort = [r for r in records if r['publication_policy'] == policy and r['source_schedule'] == schedule]
            groups[policy+'/'+schedule] = {kind:{key:{'mean':statistics.mean(r[kind][key] for r in cohort),
                'min':min(r[kind][key] for r in cohort),'max':max(r[kind][key] for r in cohort)} for key in cohort[0][kind]}
                for kind in ('metrics','factor_metrics')}
    return {'schema_version':1,'verified_models':12,'matched_schedule_pairs':6,'fine_tuning_seeds':[20261106,20261107,20261108],
            'diagnostic_cohorts':{s:{'settings':v['settings'],**{k:v['cohort'][k] for k in ('case_sha256','inputs_sha256','episodes')}} for s,v in references.items()},
            'integrity':{'answer_predictions_rescored':True,'cases_and_input_bytes_reconstructed':True,
                         'fit_evaluation_identities_disjoint':True,'paired_training_protocols_verified':True,
                         'frozen_fingerprints_and_fixed_budgets_verified':True,
                         'factor_summary_counts_checked':True,'factor_predictions_saved':False},
            'total_diagnostic_expert_executions':sum(r['fit_expert_executions']+r['evaluation_expert_executions'] for r in records),
            'runs':records,'groups':groups,'file_manifest':file_manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    result = compare(args.manifest)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'verified_models':result['verified_models'],'matched_pairs':result['matched_schedule_pairs'],'out':str(args.out)}))


if __name__ == '__main__':
    main()
