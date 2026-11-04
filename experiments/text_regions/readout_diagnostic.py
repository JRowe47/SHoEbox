"""Common 256-byte decoders and fixed factor probes on frozen graph outputs.

The answer decoders each receive only one final latent vector. Neither query
labels nor source-control labels select heads or become features. Capture runs
the graph once per batch and reuses the same observations for every diagnostic.
"""
import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .transfer_eval import capture_transfer, _policy
from .view_data import FACTOR_SOURCES, VIEWS, make_view_batch
from .view_eval import FACTORS, LinearReadout, accuracy_summary, evaluation_mode, fit_linear_readout
from .view_train import tensor_fingerprint


SUBSETS = {'all': 'ABC', 'a_only': 'A', 'b_only': 'B', 'c_only': 'C',
           'ab_only': 'AB', 'ac_only': 'AC', 'bc_only': 'BC'}
EVAL_CONTROLS = (*SUBSETS, 'no_private', 'reset_query')
MOMENTS = ('prequery', 'final')
REPRESENTATIONS = ('raw', 'mixed')


def visible(control, factor):
    if control not in EVAL_CONTROLS or factor not in FACTORS:
        raise ValueError('Unknown control or factor.')
    if control == 'reset_query':
        return False
    if control == 'no_private':
        return factor == 'shared'
    return any(source in SUBSETS[control] for source in FACTOR_SOURCES[factor])


def controlled_batch(split, start, count, *, control, query, seed, gap, query_every):
    """Apply a fixed whole-source intervention without changing observed bytes.

    Only inputs, availability, and reset are consumed by capture_transfer. The
    original labels remain scoring metadata; causal loss masks are not used.
    """
    if control not in EVAL_CONTROLS:
        raise ValueError('Unknown readout diagnostic control.')
    ordinary = control if control in ('no_private', 'reset_query') else 'all'
    batch = make_view_batch(split, start, count, seed=seed, gap=gap,
                            query_every=query_every, query_type=query, control=ordinary)
    if control in SUBSETS:
        available = {source: mask & (source in SUBSETS[control]) for source, mask in batch.availability.items()}
        retained = torch.tensor([[source in SUBSETS[control] for source in VIEWS]]).expand(count, -1).clone()
        batch = replace(batch, availability=available, retained_views=retained, control=control)
    return batch


def capture_readout_features(model, batch, segment=16):
    """Copy port features and reconstruct the exact existing final readout."""
    core = getattr(model, 'model', model)
    captured = capture_transfer(model, batch, segment)
    features, valid = {}, {}
    with evaluation_mode(model):
        for moment in MOMENTS:
            ports = captured['snapshots'][moment]
            for node in model.cfg.nodes:
                key = moment + '/' + node.name
                features[key], valid[key] = ports[node.name]['payload'], ports[node.name]['valid']
            raw = torch.cat([ports[name]['payload'] for name in model.cfg.readout], -1)
            raw_valid = torch.stack([ports[name]['valid'] for name in model.cfg.readout]).any(0)
            mixed = core.readout_norm(core.readout_mix(raw.to(core.readout.weight))).detach().cpu()
            features[moment+'/raw'], features[moment+'/mixed'] = raw, mixed
            valid[moment+'/raw'] = valid[moment+'/mixed'] = raw_valid
        reconstructed = core.readout(features['final/mixed'].to(core.readout.weight)).detach().cpu()
    if not torch.allclose(reconstructed, captured['logits'], rtol=1e-5, atol=1e-6):
        raise RuntimeError('Captured representations do not reconstruct the actual graph answer.')
    return {'features': features, 'valid': valid, 'logits': captured['logits'],
            'expert_executions': captured['expert_executions'], 'external_events': captured['external_events']}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def collect_features(model, *, split, count_per_cell=64, batch_size=64, segment=16,
                     seed=20261120, gap=8, query_every=4, on_cell=None):
    """Training covers every supported query/subset; evaluation includes controls."""
    multiple = 32 if split == 'train' else 64
    if split not in ('train', 'test', 'independent') or count_per_cell < multiple or count_per_cell % multiple:
        raise ValueError('Use full 32-tuple training cycles or 64-episode evaluation cycles.')
    if min(batch_size, segment, query_every) < 1 or gap < 0:
        raise ValueError('Invalid capture dimensions.')
    controls = tuple(SUBSETS) if split == 'train' else (('all',) if split == 'test' else EVAL_CONTROLS)
    before, policy, rng = tensor_fingerprint(model), _policy(model), torch.get_rng_state().clone()
    features, valid, factors = {}, {}, {q: [] for q in FACTORS}
    logits, answers, identities, query_labels, control_labels = [], [], [], [], []
    groups, input_hash = [], hashlib.sha256()
    cursor = 0
    for control in controls:
        for query in FACTORS:
            if split == 'train' and not visible(control, query):
                continue
            calls = events = 0
            for start in range(0, count_per_cell, batch_size):
                batch = controlled_batch(split, start, min(batch_size, count_per_cell-start),
                    control=control, query=query, seed=seed, gap=gap, query_every=query_every)
                captured = capture_readout_features(model, batch, segment)
                for key, value in captured['features'].items():
                    features.setdefault(key, []).append(value)
                    valid.setdefault(key, []).append(captured['valid'][key])
                logits.append(captured['logits']); answers.append(batch.answers)
                for factor in FACTORS:
                    factors[factor].append(batch.factors[factor])
                identities.extend(batch.identities)
                query_labels.extend(batch.query_types)
                control_labels.extend([control]*len(batch.identities))
                calls += captured['expert_executions']; events += captured['external_events']
                # Per-example hashing is invariant to the capture batch size.
                for row, identity in enumerate(batch.identities):
                    input_hash.update(json.dumps((control, query, identity)).encode())
                    for source in VIEWS:
                        input_hash.update(bytes(batch.inputs[source][row].tolist()))
                        input_hash.update(bytes(batch.availability[source][row].tolist()))
                    input_hash.update(bytes(batch.reset[row].tolist()))
            groups.append({'control':control, 'query':query, 'start':cursor, 'stop':cursor+count_per_cell,
                           'expert_executions':calls, 'external_events':events,
                           'information_present':visible(control, query)})
            cursor += count_per_cell
            if on_cell:
                on_cell({'split':split, 'control':control, 'query':query, 'completed_cells':len(groups)})
    if len({g['expert_executions'] for g in groups}) != 1 or len({g['external_events'] for g in groups}) != 1:
        raise RuntimeError('Source controls changed graph work at equal episode counts.')
    if tensor_fingerprint(model) != before or _policy(model) != policy or not torch.equal(torch.get_rng_state(), rng):
        raise RuntimeError('Feature collection mutated frozen tensors, policy or RNG.')
    cases = list(zip(control_labels, identities))
    if len(set(cases)) != len(cases):
        raise ValueError('Duplicate control/episode cases in diagnostic cohort.')
    return {'features':{k:torch.cat(v) for k,v in features.items()}, 'valid':{k:torch.cat(v) for k,v in valid.items()},
            'factors':{k:torch.cat(v) for k,v in factors.items()}, 'logits':torch.cat(logits), 'answers':torch.cat(answers),
            'identities':identities, 'queries':query_labels, 'controls':control_labels, 'groups':groups,
            'protocol':{'split':split, 'count_per_cell':count_per_cell, 'seed':seed, 'gap':gap, 'query_every':query_every,
                        'batch_size':batch_size, 'segment':segment, 'episodes':cursor, 'model_tensor_sha256':before,
                        'policy':policy, 'case_sha256':_digest(cases), 'inputs_sha256':input_hash.hexdigest(),
                        'expert_executions':sum(g['expert_executions'] for g in groups),
                        'external_events':sum(g['external_events'] for g in groups), 'frozen_verified':True}}


@dataclass(frozen=True)
class ByteDecoder:
    mean: torch.Tensor
    scale: torch.Tensor
    weight: torch.Tensor
    fit: dict

    def logits(self, features):
        x = features.detach().cpu().double()
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not torch.isfinite(x).all():
            raise ValueError('Decoder requires finite features of its fitted width.')
        design = torch.cat(((x-self.mean)/self.scale, torch.ones(len(x), 1, dtype=x.dtype)), 1)
        return design @ self.weight

    def predict(self, features):
        return self.logits(features).argmax(-1)


def fit_byte_decoder(features, answers, *, sample_weights=None, l2=.001, max_iter=200):
    """Deterministic convex 256-class fit; every output byte remains eligible."""
    x, y = features.detach().cpu().double(), answers.detach().cpu().long()
    if (x.ndim != 2 or len(x) < 1 or y.shape != (len(x),) or not torch.isfinite(x).all()
            or bool(((y < 0) | (y > 255)).any()) or not math.isfinite(l2) or l2 <= 0 or max_iter < 1):
        raise ValueError('Invalid byte-decoder fitting inputs or regularization.')
    mass = torch.ones(len(x), dtype=torch.double) if sample_weights is None else sample_weights.detach().cpu().double()
    if mass.shape != y.shape or not torch.isfinite(mass).all() or bool((mass <= 0).any()):
        raise ValueError('Fitting weights must be finite and positive.')
    mass = mass/mass.sum()
    mean = (x*mass[:,None]).sum(0)
    scale = ((x-mean).square()*mass[:,None]).sum(0).sqrt().clamp_min(1e-6)
    design = torch.cat(((x-mean)/scale, torch.ones(len(x), 1, dtype=x.dtype)), 1)
    weight = torch.zeros(design.shape[1], 256, dtype=torch.double, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight], max_iter=max_iter, history_size=30,
                                tolerance_grad=1e-7, tolerance_change=1e-11, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = (F.cross_entropy(design @ weight, y, reduction='none')*mass).sum() + .5*l2*weight.square().sum()
        loss.backward()
        return loss
    with torch.enable_grad():
        optimizer.step(closure)
        objective = float(closure().detach())
    prediction = (design @ weight.detach()).argmax(-1)
    info = {'examples':len(x), 'width':x.shape[1], 'output_bytes':256, 'l2':l2, 'max_iter':max_iter,
            'iterations':optimizer.state[weight]['n_iter'], 'objective':objective,
            'gradient_max':float(weight.grad.abs().max()), 'training_accuracy':float((prediction == y).double().mean()),
            'weighted_training_accuracy':float(((prediction == y).double()*mass).sum()),
            'initialization':'all zero; deterministic, no RNG use', 'optimizer':'full-batch L-BFGS with strong Wolfe search',
            'normalization':'training-weighted mean and standard deviation',
            'regularization':'0.5 * l2 * sum(weight squared), including intercept'}
    if not math.isfinite(objective) or not torch.isfinite(weight).all():
        raise RuntimeError('Byte decoder fit is nonfinite.')
    return ByteDecoder(mean, scale, weight.detach().clone(), info)


def fit_diagnostic(training, *, l2=.001, max_iter=200, ridge=.01):
    if training['protocol']['split'] != 'train':
        raise ValueError('Diagnostics may be fitted only on training combinations.')
    if not all(visible(c,q) for c,q in zip(training['controls'], training['queries'])):
        raise ValueError('Answer decoder training requires supported source/query combinations.')
    query_counts = {q:training['queries'].count(q) for q in FACTORS}
    if not all(query_counts.values()):
        raise ValueError('One common decoder must fit all three query types.')
    mass = torch.tensor([1/query_counts[q] for q in training['queries']], dtype=torch.double)
    answer_heads = {name:fit_byte_decoder(training['features']['final/'+name], training['answers'],
                                         sample_weights=mass, l2=l2, max_iter=max_iter) for name in REPRESENTATIONS}
    probes, probe_fits = {}, {}
    for key, feature in training['features'].items():
        for factor in FACTORS:
            supported = torch.tensor([visible(c,factor) for c in training['controls']])
            used = training['valid'][key] & supported
            labels = training['factors'][factor]
            counts = {chr(v):int((labels[used] == v).sum()) for v in b'ABCD'}
            name = key+'/'+factor
            probe_fits[name] = {'valid_supported_examples':int(used.sum()), 'by_answer':counts,
                                'status':'fitted' if all(counts.values()) else 'insufficient_valid_classes'}
            if all(counts.values()):
                probes[name] = fit_linear_readout(feature[used], labels[used], ridge)
    return {'schema_version':1, 'answer_heads':answer_heads, 'factor_heads':probes, 'factor_fits':probe_fits,
            'fit_protocol':training['protocol'], 'fit_identities':sorted(set(training['identities'])),
            'fit_query_counts':query_counts, 'answer_loss_weighting':'equal query mass; uniform supported subsets within query',
            'factor_fit':'fixed per factor, representation and moment; pooled across queries and source subsets', 'ridge':ridge}


def score_diagnostic(data, fitted):
    if data['protocol']['split'] not in ('test', 'independent'):
        raise ValueError('Evaluate only held-out or independent diagnostic cohorts.')
    if set(data['identities']) & set(fitted['fit_identities']):
        raise ValueError('Fit and evaluation identities overlap.')
    for key in ('model_tensor_sha256', 'policy', 'gap', 'query_every'):
        if data['protocol'][key] != fitted['fit_protocol'][key]:
            raise ValueError('Diagnostic feature/model protocol mismatch: '+key)
    all_logits = {'direct':data['logits'], **{k:h.logits(data['features']['final/'+k]) for k,h in fitted['answer_heads'].items()}}
    answers, rows, predictions = data['answers'], [], []
    for group in data['groups']:
        a,b = group['start'], group['stop']
        record = dict(group); record['decoders'] = {}
        raw = {'split':data['protocol']['split'], 'control':group['control'], 'query':group['query'],
               'episode_ids':data['identities'][a:b], 'answers':answers[a:b].tolist(), 'predictions':{}}
        for name, logits in all_logits.items():
            predicted = logits[a:b].argmax(-1)
            metrics = accuracy_summary(predicted.tolist(), answers[a:b].tolist())
            metrics['answer_nats_per_byte'] = float(F.cross_entropy(logits[a:b], answers[a:b]))
            metrics['outside_answer_vocabulary'] = int((~(predicted[:,None] == torch.tensor(list(b'ABCD'))).any(1)).sum())
            record['decoders'][name] = metrics
            raw['predictions'][name] = predicted.tolist()
        rows.append(record); predictions.append(raw)
    factors = {}
    for name, head in fitted['factor_heads'].items():
        fit_moment, node, factor = name.split('/')
        supported = torch.tensor([visible(c,factor) for c in data['controls']])
        query_match = torch.tensor([q == factor for q in data['queries']])
        labels = data['factors'][factor]
        for moment in MOMENTS:
            key = moment+'/'+node
            # A reset at the final query is a missing-information control only
            # at the final boundary; description-boundary facts are present.
            supported_at_cut = supported | ((moment == 'prequery') & torch.tensor([c == 'reset_query' for c in data['controls']]))
            predicted = head.predict(data['features'][key])
            valid = data['valid'][key] & supported_at_cut
            conditions = {}
            for control in dict.fromkeys(data['controls']):
                selected = torch.tensor([c == control for c in data['controls']])
                metrics = {}
                for selection, mask in (('pooled',selected), ('query_matched',selected & query_match)):
                    used = mask & valid
                    score = accuracy_summary(predicted[used].tolist(), labels[used].tolist())
                    score.update(total_episodes=int(mask.sum()), valid_supported_episodes=int(used.sum()))
                    metrics[selection] = score
                conditions[control] = metrics
            factors[fit_moment+'->'+moment+'/'+node+'/'+factor] = conditions
    return {'protocol':data['protocol'], 'groups':rows, 'factor_decodability':factors}, predictions


def save_diagnostic(path, fitted):
    payload = {**fitted, 'answer_heads':{k:asdict(v) for k,v in fitted['answer_heads'].items()},
               'factor_heads':{k:asdict(v) for k,v in fitted['factor_heads'].items()}}
    torch.save(payload, path)


def load_diagnostic(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    expected = {'schema_version','answer_heads','factor_heads','factor_fits','fit_protocol','fit_identities',
                'fit_query_counts','answer_loss_weighting','factor_fit','ridge'}
    if (not isinstance(payload, dict) or set(payload) != expected or type(payload.get('schema_version')) is not int
            or payload['schema_version'] != 1 or payload.get('fit_protocol',{}).get('split') != 'train'):
        raise ValueError('Unsupported training-only diagnostic artifact.')
    if set(payload.get('answer_heads',{})) != set(REPRESENTATIONS):
        raise ValueError('Exactly two common byte decoders required.')
    counts, identities = payload['fit_query_counts'], payload['fit_identities']
    if (not isinstance(counts,dict) or set(counts) != set(FACTORS)
            or any(type(v) is not int or v < 1 for v in counts.values())
            or not isinstance(identities,(list,tuple)) or not identities
            or any(not isinstance(v,str) or not v for v in identities) or len(set(identities)) != len(identities)):
        raise ValueError('Invalid fitted cohort identities or query counts.')
    if not isinstance(payload['factor_heads'],dict) or not isinstance(payload['factor_fits'],dict):
        raise ValueError('Missing fixed factor readout inventory.')
    for name, fit in payload['factor_fits'].items():
        parts = name.split('/')
        if len(parts) != 3 or parts[0] not in MOMENTS or parts[2] not in FACTORS:
            raise ValueError('Factor heads must be fixed by moment, representation and factor.')
        if not isinstance(fit,dict) or set(fit) != {'valid_supported_examples','by_answer','status'}:
            raise ValueError('Invalid factor fit record.')
        by_answer = fit['by_answer']
        if (type(fit['valid_supported_examples']) is not int or not isinstance(by_answer,dict)
                or set(by_answer) != set('ABCD') or any(type(v) is not int or v < 0 for v in by_answer.values())
                or sum(by_answer.values()) != fit['valid_supported_examples']):
            raise ValueError('Invalid factor class counts.')
        status = 'fitted' if all(by_answer.values()) else 'insufficient_valid_classes'
        if fit['status'] != status or (name in payload['factor_heads']) != (status == 'fitted'):
            raise ValueError('Factor fit inventory differs from its class coverage.')
    if not set(payload['factor_heads']) <= set(payload['factor_fits']):
        raise ValueError('Unrecorded factor head.')
    for group, columns in (('answer_heads',256), ('factor_heads',4)):
        for name, record in payload[group].items():
            record_fields = {'mean','scale','weight','fit' if group == 'answer_heads' else 'fitted_examples'}
            if not isinstance(record,dict) or set(record) != record_fields:
                raise ValueError('Invalid diagnostic head record.')
            mean, scale, weight = (record[k] for k in ('mean','scale','weight'))
            if (any(not torch.is_tensor(t) or not t.is_floating_point() or t.requires_grad or not torch.isfinite(t).all()
                    for t in (mean,scale,weight)) or mean.ndim != 1 or len(mean) < 1 or scale.shape != mean.shape
                    or weight.shape != (len(mean)+1,columns) or not bool((scale > 0).all())):
                raise ValueError('Invalid diagnostic head tensors.')
            if group == 'answer_heads':
                fit = record['fit']
                if (not isinstance(fit,dict) or fit.get('width') != len(mean) or fit.get('output_bytes') != 256
                        or fit.get('examples') != sum(counts.values())):
                    raise ValueError('Byte head dimensions or fit cohort differ from metadata.')
            elif record['fitted_examples'] != payload['factor_fits'][name]['valid_supported_examples']:
                raise ValueError('Factor head count differs from its fit record.')
            payload[group][name] = ByteDecoder(**record) if group == 'answer_heads' else LinearReadout(**record)
    return payload


def main():
    from .view_train import load_view_checkpoint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fit-per-cell',type=int,default=64); parser.add_argument('--count-per-query',type=int,default=128)
    parser.add_argument('--batch',type=int,default=64); parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--seed',type=int,default=20261120); parser.add_argument('--max-iter',type=int,default=200)
    parser.add_argument('--l2',type=float,default=.001); parser.add_argument('--ridge',type=float,default=.01)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('Positive thread count required.')
    torch.set_num_threads(args.threads)
    sources = [Path(__file__).with_name(name) for name in
               ('readout_diagnostic.py','model.py','learning.py','transfer_eval.py','view_data.py','view_eval.py','view_train.py')]
    source_hashes = {str(p.relative_to(Path(__file__).resolve().parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    model,_,_,saved = load_view_checkpoint(args.checkpoint,restore_rng=False)
    if model.cfg.rounds != 1:
        parser.error('This diagnostic protocol requires K=1.')
    before = tensor_fingerprint(model)
    args.out.mkdir(parents=True,exist_ok=True)
    options = {k:saved['protocol'][k] for k in ('gap','query_every','segment')}
    options.update(seed=args.seed,batch_size=args.batch)
    progress = lambda value: print(json.dumps(value),flush=True)
    started = time.perf_counter()
    training = collect_features(model,split='train',count_per_cell=args.fit_per_cell,on_cell=progress,**options)
    fitted = fit_diagnostic(training,l2=args.l2,max_iter=args.max_iter,ridge=args.ridge)
    save_diagnostic(args.out/'readouts.pt',fitted)
    restored = load_diagnostic(args.out/'readouts.pt')
    for name in REPRESENTATIONS:
        assert torch.equal(fitted['answer_heads'][name].weight,restored['answer_heads'][name].weight)
    report = {'schema_version':1,'complete':False,'checkpoint_step':saved['step'],'training_protocol':saved['protocol'],
              'model_tensor_sha256':before,'fit_protocol':training['protocol'],
              'fit_query_counts':fitted['fit_query_counts'],'answer_loss_weighting':fitted['answer_loss_weighting'],
              'answer_decoder_fits':{k:v.fit for k,v in fitted['answer_heads'].items()},
              'factor_fits':fitted['factor_fits'],'ridge':args.ridge,'evaluations':{},'source_sha256':source_hashes,
              'runtime':{'torch_version':str(torch.__version__),'threads':args.threads,'rounds':model.cfg.rounds}}
    with (args.out/'predictions.jsonl').open('w') as stream:
        for split in ('test','independent'):
            data = collect_features(model,split=split,count_per_cell=args.count_per_query,on_cell=progress,**options)
            result,predictions = score_diagnostic(data,fitted)
            report['evaluations'][split] = result
            for record in predictions:
                stream.write(json.dumps(record,separators=(',',':'))+'\n')
            stream.flush()
            if tensor_fingerprint(model) != before:
                raise RuntimeError('Frozen graph mutated during diagnostic.')
            report['frozen_verified'] = True
            report['seconds'] = time.perf_counter()-started
            report['complete'] = set(report['evaluations']) == {'test','independent'}
            if source_hashes != {str(p.relative_to(Path(__file__).resolve().parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
                raise RuntimeError('Diagnostic runtime source changed while the run was active.')
            temporary = args.out/'results.json.tmp'
            temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(args.out/'results.json')
    print(json.dumps({'complete':True,'seconds':report['seconds'],'out':str(args.out)}),flush=True)


if __name__ == '__main__':
    main()
