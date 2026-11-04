"""Matched linear and GELU byte decoders on frozen regional publications.

One shared head receives one observed vector. Query and source labels only
weight training examples or select reporting groups, never inference heads.
"""
import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .readout_diagnostic import collect_features, load_diagnostic, REPRESENTATIONS, visible
from .view_eval import FACTORS, accuracy_summary
from .view_train import load_view_checkpoint, tensor_fingerprint

ACTIVATIONS = ('identity', 'gelu')
ADAPTER_SEEDS = (20261121, 20261122, 20261123)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_protocol(left, right):
    """Tensor artifacts retain tuples; JSON serializes the same arrays as lists."""
    return json.dumps(left,sort_keys=True) == json.dumps(right,sort_keys=True)


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class EmissionAdapter:
    mean: torch.Tensor
    scale: torch.Tensor
    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor
    activation: str
    fit: dict

    def logits(self, features):
        x = features.detach().cpu().double()
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not torch.isfinite(x).all():
            raise ValueError('Expected finite observed features of the fitted width.')
        hidden = ((x-self.mean)/self.scale) @ self.w1 + self.b1
        if self.activation == 'gelu':
            hidden = F.gelu(hidden)
        elif self.activation != 'identity':
            raise ValueError('Unknown emission activation.')
        return hidden @ self.w2 + self.b2

    def predict(self, features):
        return self.logits(features).argmax(-1)


def fit_adapter(features, answers, *, sample_weights=None, activation='gelu',
                hidden=64, seed=20261121, l2=.001, max_iter=400):
    x, y = features.detach().cpu().double(), answers.detach().cpu().long()
    if (x.ndim != 2 or not len(x) or y.shape != (len(x),) or not torch.isfinite(x).all()
            or ((y < 0) | (y > 255)).any() or activation not in ACTIVATIONS
            or type(hidden) is not int or hidden < 1 or type(seed) is not int
            or not math.isfinite(l2) or l2 <= 0 or max_iter < 1):
        raise ValueError('Invalid emission fitting inputs.')
    mass = torch.ones(len(x), dtype=torch.double) if sample_weights is None else sample_weights.detach().cpu().double()
    if mass.shape != y.shape or not torch.isfinite(mass).all() or (mass <= 0).any():
        raise ValueError('Expected finite positive fitting weights.')
    mass = mass/mass.sum()
    mean = (x*mass[:,None]).sum(0)
    scale = ((x-mean).square()*mass[:,None]).sum(0).sqrt().clamp_min(1e-6)
    x = (x-mean)/scale
    generator = torch.Generator().manual_seed(seed)
    parameters = [torch.randn(x.shape[1], hidden, generator=generator, dtype=torch.double)/math.sqrt(x.shape[1]),
                  torch.zeros(hidden, dtype=torch.double),
                  .1*torch.randn(hidden, 256, generator=generator, dtype=torch.double)/math.sqrt(hidden),
                  torch.zeros(256, dtype=torch.double)]
    initial_hash = tensor_hash(parameters)
    for parameter in parameters:
        parameter.requires_grad_(True)
    w1,b1,w2,b2 = parameters
    def logits():
        z = x @ w1 + b1
        return (F.gelu(z) if activation == 'gelu' else z) @ w2 + b2
    optimizer = torch.optim.LBFGS(parameters, max_iter=max_iter, history_size=30,
        tolerance_grad=1e-7, tolerance_change=1e-11, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = (F.cross_entropy(logits(), y, reduction='none')*mass).sum()
        loss = loss + .5*l2*sum(p.square().sum() for p in parameters)
        loss.backward()
        return loss
    started = time.perf_counter()
    with torch.enable_grad():
        optimizer.step(closure)
        objective = float(closure().detach())
    with torch.no_grad():
        predicted = logits().argmax(-1)
        training_nll = float((F.cross_entropy(logits(),y,reduction='none')*mass).sum())
    if not math.isfinite(objective) or any(not torch.isfinite(p).all() for p in parameters):
        raise RuntimeError('Nonfinite emission fit.')
    metadata = {'width':x.shape[1], 'hidden':hidden, 'output_bytes':256, 'activation':activation,
        'seed':seed, 'examples':len(x), 'parameters':sum(p.numel() for p in parameters),
        'l2':l2, 'max_iter':max_iter, 'iterations':optimizer.state[w1]['n_iter'],
        'gradient_max':max(float(p.grad.abs().max()) for p in parameters),
        'objective':objective, 'weighted_training_nll':training_nll,
        'weighted_training_accuracy':float(((predicted == y).double()*mass).sum()),
        'initial_tensor_sha256':initial_hash, 'seconds':time.perf_counter()-started,
        'normalization':'training-weighted mean and standard deviation',
        'optimizer':'full-batch L-BFGS with strong Wolfe search',
        'regularization':'0.5 * l2 * squared sum of all layer weights and biases',
        'selection':'fixed settings and iteration cap; no evaluation-based selection'}
    return EmissionAdapter(mean,scale,*(p.detach().clone() for p in parameters),activation,metadata)


def fit_adapters(training, *, seeds=ADAPTER_SEEDS, hidden=64, l2=.001, max_iter=400, progress=None):
    if training['protocol']['split'] != 'train':
        raise ValueError('Fit only training combinations.')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Use distinct initialization seeds.')
    counts = {q:training['queries'].count(q) for q in FACTORS}
    if (not all(counts.values()) or sum(counts.values()) != len(training['answers'])
            or len(training['controls']) != len(training['answers'])
            or not all(visible(c,q) for c,q in zip(training['controls'],training['queries']))):
        raise ValueError('Fit every query using supported sources only.')
    mass = torch.tensor([1/counts[q] for q in training['queries']],dtype=torch.double)
    heads = {}
    for representation in REPRESENTATIONS:
        for seed in seeds:
            for activation in ACTIVATIONS:
                name = f'{representation}/{activation}/{seed}'
                heads[name] = fit_adapter(training['features']['final/'+representation],training['answers'],
                    sample_weights=mass,activation=activation,hidden=hidden,seed=seed,l2=l2,max_iter=max_iter)
                if progress:
                    progress({'fit':name, **heads[name].fit})
    return {'schema_version':1, 'heads':heads, 'fit_protocol':training['protocol'],
            'fit_identities':sorted(set(training['identities'])), 'query_counts':counts,
            'initialization_seeds':list(seeds),
            'loss_weighting':'equal query mass; uniform supported source subsets within query'}


def save_adapters(path, fitted):
    torch.save({**fitted,'heads':{k:asdict(v) for k,v in fitted['heads'].items()}},path)


def load_adapters(path):
    payload = torch.load(path,map_location='cpu',weights_only=True)
    fields = {'schema_version','heads','fit_protocol','fit_identities','query_counts',
              'initialization_seeds','loss_weighting'}
    if (not isinstance(payload,dict) or set(payload) != fields or payload['schema_version'] != 1
            or payload['fit_protocol'].get('split') != 'train'):
        raise ValueError('Invalid training-only emission artifact.')
    seeds = payload['initialization_seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('Invalid adapter initialization seeds.')
    expected = {f'{r}/{a}/{s}' for r in REPRESENTATIONS for a in ACTIVATIONS for s in seeds}
    if set(payload['heads']) != expected:
        raise ValueError('Incomplete common-head inventory.')
    counts, identities = payload['query_counts'],payload['fit_identities']
    if (set(counts) != set(FACTORS) or any(type(v) is not int or v <= 0 for v in counts.values())
            or not identities or len(set(identities)) != len(identities) or any(not isinstance(i,str) for i in identities)):
        raise ValueError('Invalid fitting cohort metadata.')
    for name,record in payload['heads'].items():
        if set(record) != set(EmissionAdapter.__dataclass_fields__):
            raise ValueError('Invalid head fields.')
        head = EmissionAdapter(**record)
        tensors = [getattr(head,k) for k in ('mean','scale','w1','b1','w2','b2')]
        if any(not torch.is_tensor(t) or not t.is_floating_point() or t.requires_grad or not torch.isfinite(t).all() for t in tensors):
            raise ValueError('Invalid head tensors.')
        d,h = head.fit.get('width'),head.fit.get('hidden')
        if (type(d) is not int or type(h) is not int or min(d,h) < 1
                or [tuple(t.shape) for t in tensors] != [(d,),(d,),(d,h),(h,),(h,256),(256,)]
                or not (head.scale > 0).all() or head.activation not in ACTIVATIONS
                or name != f'{name.split("/")[0]}/{head.activation}/{head.fit.get("seed")}'
                or head.fit.get('activation') != head.activation or head.fit.get('output_bytes') != 256
                or head.fit.get('parameters') != sum(t.numel() for t in tensors[2:])
                or head.fit.get('examples') != sum(counts.values())):
            raise ValueError('Head dimensions or metadata differ.')
        payload['heads'][name] = head
    return payload


def compact_capture(data):
    return {**{k:data[k] for k in ('answers','queries','controls','identities','groups','protocol')},
        'features':{f'final/{r}':data['features'][f'final/{r}'] for r in REPRESENTATIONS},
        'direct_predictions':data['logits'].argmax(-1),
        'direct_nll_per_case':F.cross_entropy(data['logits'],data['answers'],reduction='none')}


def score_adapters(data, fitted, reference):
    if data['protocol']['split'] not in ('test','independent'):
        raise ValueError('Score only evaluation cohorts.')
    if set(data['identities']) & set(fitted['fit_identities']):
        raise ValueError('Fit/evaluation identities overlap.')
    for key in ('model_tensor_sha256','policy','gap','query_every'):
        if data['protocol'][key] != fitted['fit_protocol'][key]:
            raise ValueError('Evaluation graph or protocol differs: '+key)
    logits = {f'linear/{r}':reference['answer_heads'][r].logits(data['features'][f'final/{r}']) for r in REPRESENTATIONS}
    logits.update({name:h.logits(data['features']['final/'+name.split('/')[0]]) for name,h in fitted['heads'].items()})
    predictions = {name:value.argmax(-1) for name,value in logits.items()}
    nll = {name:F.cross_entropy(value,data['answers'],reduction='none') for name,value in logits.items()}
    predictions['direct'],nll['direct'] = data['direct_predictions'],data['direct_nll_per_case']
    groups,raw = [],[]
    for group in data['groups']:
        a,b = group['start'],group['stop'];answers = data['answers'][a:b].tolist()
        scores = {}
        row = {'split':data['protocol']['split'],'control':group['control'],'query':group['query'],
               'episode_ids':data['identities'][a:b],'answers':answers,'predictions':{}}
        for name,predicted in predictions.items():
            values = predicted[a:b].tolist()
            scores[name] = {**accuracy_summary(values,answers),'answer_nats_per_byte':float(nll[name][a:b].mean())}
            row['predictions'][name] = values
        groups.append({**group,'decoders':scores});raw.append(row)
    return {'protocol':data['protocol'],'groups':groups},raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--prior-results',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--hidden',type=int,default=64)
    parser.add_argument('--max-iter',type=int,default=400)
    parser.add_argument('--l2',type=float,default=.001)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(ADAPTER_SEEDS))
    args = parser.parse_args()
    if args.out.exists() or args.threads < 1:
        parser.error('Use a new output directory and positive thread count.')
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    progress = lambda x: print(json.dumps(x),flush=True)
    files = [Path(__file__).with_name(n) for n in ('emission_adapter.py','readout_diagnostic.py',
        'model.py','learning.py','transfer_eval.py','view_data.py','view_eval.py','view_train.py')]
    source = {str(p.relative_to(Path(__file__).resolve().parents[2])):sha(p) for p in files}
    reference = load_diagnostic(args.reference)
    prior = json.loads(gzip.decompress(args.prior_results.read_bytes()))
    prior_rows = [json.loads(line) for line in args.prior_results.with_name('predictions.jsonl').read_text().splitlines()]
    model,_,_,saved = load_view_checkpoint(args.checkpoint,restore_rng=False)
    before = tensor_fingerprint(model)
    if before != reference['fit_protocol']['model_tensor_sha256'] or model.cfg.rounds != 1:
        raise ValueError('Expected the exact frozen reference graph with K=1.')
    datasets = {}
    for split in ('train','test','independent'):
        protocol = reference['fit_protocol'] if split == 'train' else prior['evaluations'][split]['protocol']
        capture = collect_features(model,split=split,count_per_cell=protocol['count_per_cell'],
            **{k:protocol[k] for k in ('seed','gap','query_every','batch_size','segment')},on_cell=progress)
        if not same_protocol(capture['protocol'], protocol):
            raise ValueError('Capture differs from the frozen reference protocol.')
        if split != 'train':
            baseline = {'direct':capture['logits'].argmax(-1),
                **{r:reference['answer_heads'][r].predict(capture['features']['final/'+r]) for r in REPRESENTATIONS}}
            for group,old in zip(capture['groups'],[r for r in prior_rows if r['split']==split],strict=True):
                a,b=group['start'],group['stop']
                if (capture['identities'][a:b] != old['episode_ids'] or capture['answers'][a:b].tolist() != old['answers']
                        or any(v[a:b].tolist()!=old['predictions'][k] for k,v in baseline.items())):
                    raise ValueError('Prior baseline predictions failed exact reproduction.')
        datasets[split] = compact_capture(capture)
    torch.save({'schema_version':1,'checkpoint_sha256':sha(args.checkpoint),'source_sha256':source,
                'datasets':datasets},args.out/'features.pt')
    progress({'capture_complete':True,'baseline_predictions_reproduced':True})
    del model,capture
    fitted = fit_adapters(datasets['train'],seeds=tuple(args.seeds),hidden=args.hidden,l2=args.l2,max_iter=args.max_iter,progress=progress)
    save_adapters(args.out/'adapters.pt',fitted)
    restored = load_adapters(args.out/'adapters.pt')
    for name,head in fitted['heads'].items():
        x=datasets['train']['features']['final/'+name.split('/')[0]]
        if not torch.equal(head.logits(x),restored['heads'][name].logits(x)):
            raise RuntimeError('Fitted head restoration changed logits.')
    report = {'schema_version':1,'complete':True,'model_tensor_sha256':before,'source_sha256':source,
        'checkpoint_sha256':sha(args.checkpoint),'reference_sha256':sha(args.reference),
        'prior_results_sha256':sha(args.prior_results),'features_sha256':sha(args.out/'features.pt'),
        'adapters_sha256':sha(args.out/'adapters.pt'),'fit_protocol':fitted['fit_protocol'],
        'fit_query_counts':fitted['query_counts'],'initialization_seeds':fitted['initialization_seeds'],
        'loss_weighting':fitted['loss_weighting'],'fits':{k:h.fit for k,h in fitted['heads'].items()},
        'training_protocol':saved['protocol'],'evaluations':{},'baseline_predictions_reproduced':True,
        'frozen_verified':True,'runtime':{'torch':str(torch.__version__),'threads':args.threads}}
    with (args.out/'predictions.jsonl').open('w') as stream:
        for split in ('test','independent'):
            result,rows = score_adapters(datasets[split],restored,reference)
            report['evaluations'][split]=result
            for row in rows:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    if source != {str(p.relative_to(Path(__file__).resolve().parents[2])):sha(p) for p in files}:
        raise RuntimeError('Source changed during the experiment.')
    report['seconds']=time.perf_counter()-started
    (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    progress({'complete':True,'seconds':report['seconds'],'out':str(args.out)})


if __name__ == '__main__':
    main()
