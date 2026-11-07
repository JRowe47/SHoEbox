"""Held-out recall plus paired observation changes and producer-state swaps."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.state import NodeState
from experiments.text_regions.data import split_bytes
from experiments.text_regions.graph_mixed_train import evaluate_language, evaluate_memory
from experiments.text_regions.task_stream import TaskStreamConfig, make_documents


def counterfactual_pairs(documents, values='0123', minimum_delay=96):
    """One first eligible exact query per document; one earlier byte changes."""
    pairs = []
    for document in documents:
        query = next((q for q in document.queries if q.kind == 'exact' and q.delay > minimum_delay), None)
        if query is None:
            continue
        if document.data[query.source_position] != query.answer or query.source_position >= query.position:
            raise ValueError('Query provenance is not its observed fact value.')
        alternative = ord(values[(values.index(chr(query.answer))+1) % len(values)])
        original = document.data[:query.position+1]
        changed = bytearray(original); changed[query.source_position] = alternative
        pairs.append(dict(document=document.index,query=asdict(query),original=original,
                          changed=bytes(changed),alternative=alternative))
    if not pairs:
        raise ValueError('No eligible delayed exact queries.')
    return pairs


def swap_producer_contents(state, owners, donors):
    nodes = dict(state.nodes)
    for name in owners:
        old = nodes[name]
        nodes[name] = NodeState(tuple(t[donors] for t in old.contextualizer),
                               tuple(t[donors] for t in old.blocks), old.clock)
    return replace(state,nodes=nodes)


@torch.no_grad()
def counterfactual_probe(model, pairs, context=32):
    sequences = [p[key] for p in pairs for key in ('original','changed')]
    length = max(map(len,sequences))-1
    batch = len(sequences)
    device = model.readout.weight.device
    x = torch.zeros(batch,length,dtype=torch.long,device=device)
    valid = torch.zeros_like(x,dtype=torch.bool)
    reset = torch.zeros_like(valid)
    for row, data in enumerate(sequences):
        # Left padding is inactive. Reset when the real document begins so
        # every row retains its original occurrence and round clocks.
        start = length-(len(data)-1)
        x[row,start:] = torch.tensor(list(data[:-1]),device=device)
        valid[row,start:] = True
        reset[row,start] = True
    state = None
    for start in range(0,length,context):
        mask = valid[:,start:start+context]
        result = model(x[:,start:start+context],state=state,reset=reset[:,start:start+context],
            availability={v:mask for v in model.cfg.views},node_masks={n:mask for n in model.nodes})
        state = result['state']
    query = torch.tensor([[data[-1]] for data in sequences],device=device)
    intact = model(query,state=state)['logits'][:,0]
    donors = torch.arange(batch,device=device).reshape(-1,2).flip(1).reshape(-1)
    swapped = model(query,state=swap_producer_contents(state,model.cfg.task_publication,donors))['logits'][:,0]
    targets = torch.tensor([v for p in pairs for v in (p['query']['answer'],p['alternative'])],device=device)
    rows = []
    for i,pair in enumerate(pairs):
        a,b=pair['query']['answer'],pair['alternative']
        def contrast(logits):
            return float((logits[2*i,a]-logits[2*i,b])-(logits[2*i+1,a]-logits[2*i+1,b]))
        rows.append(dict(document=pair['document'],delay=pair['query']['delay'],
            source_position=pair['query']['source_position'],query_position=pair['query']['position'],
            answer=a,alternative=b,intact_contrast=contrast(intact),swapped_contrast=contrast(swapped),
            original_prediction=int(intact[2*i].argmax()),changed_prediction=int(intact[2*i+1].argmax())))
    nll=F.cross_entropy(intact,targets)
    return dict(pairs=len(pairs),rows=rows,answer_nll=float(nll),accuracy=float(intact.argmax(-1).eq(targets).float().mean()),
        mean_directional_log_odds=sum(r['intact_contrast'] for r in rows)/len(rows),
        mean_swapped_directional_log_odds=sum(r['swapped_contrast'] for r in rows)/len(rows),
        producer_swap_nll_delta=float(F.cross_entropy(swapped,targets)-nll),
        producer_swap_mean_absolute_logit_change=float((swapped-intact).abs().mean()),
        paired_correct_fraction=sum(r['original_prediction']==r['answer'] and
                                   r['changed_prediction']==r['alternative'] for r in rows)/len(rows))


def exact_cohort(traces,threshold=96):
    rows=[r for r in traces if r['kind']=='exact' and r['delay']>threshold]
    return dict(count=len(rows),nll=sum(r['nll'] for r in rows)/len(rows) if rows else None,
        accuracy=sum(r['prediction']==r['answer'] for r in rows)/len(rows) if rows else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'artifacts/state_credit')
    p.add_argument('--data',type=Path,default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--checkpoints',nargs='+',choices=('parent','state_best','state_last','publication_best','publication_last'))
    a=p.parse_args()
    torch.set_num_threads(1)
    study=torch.load(a.root/'state/last.pt',map_location='cpu',weights_only=True)
    protocol=study['protocol']; cfg=GraphConfig.from_dict(study['config'])
    data_config=TaskStreamConfig.from_dict(study['data_config'])
    raw=a.data.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=study['raw_sha256']:
        raise ValueError('Evaluation corpus mismatch.')
    _,_,test=split_bytes(raw)
    documents={split:make_documents(data_config,split=split,seed=protocol['data_seed'],
        count=protocol['eval_count']*2,distractor_bytes=test) for split in ('test','long_delay')}
    pairs=counterfactual_pairs(documents['long_delay'],data_config.values)
    paths=dict(parent=a.root/'parent.pt',state_best=a.root/'state/best.pt',state_last=a.root/'state/last.pt',
               publication_best=a.root/'publication/best.pt',publication_last=a.root/'publication/last.pt')
    output_path=a.root/'evaluation.json'
    output=json.loads(output_path.read_text()) if output_path.exists() else {}
    cache_path=a.root/'evaluation_cache.json'
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    def persist():
        tmp=cache_path.with_suffix('.tmp'); tmp.write_text(json.dumps(cache,indent=2)+'\n'); tmp.replace(cache_path)
    for name,path in paths.items():
        if a.checkpoints and name not in a.checkpoints:
            continue
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['config']!=study['config']:
            raise ValueError('Evaluation graph mismatch.')
        model=RegionLanguageModel(cfg); model.load_state_dict(saved['weights']); model.eval()
        digest=hashlib.sha256(json.dumps(dict(schema=1,protocol=protocol,config=cfg.to_dict(),
            corpus=study['raw_sha256'],data=data_config.to_dict(),
            evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),sort_keys=True).encode())
        for key,value in sorted(saved['weights'].items()):
            digest.update(repr((key,str(value.dtype),tuple(value.shape))).encode())
            digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        signature=digest.hexdigest(); row=cache.setdefault(signature,{})
        for measurement in ('language','test','long_delay','counterfactual'):
            if measurement not in row:
                if measurement=='language':
                    row[measurement]=evaluate_language(model,test,protocol['batch'],protocol['context'],protocol['eval_chunks']*2)
                elif measurement=='counterfactual':
                    row[measurement]=counterfactual_probe(model,pairs,protocol['context'])
                else:
                    metric,traces=evaluate_memory(model,data_config,protocol['data_seed'],len(documents[measurement]),
                        protocol['context'],measurement,documents=documents[measurement])
                    metric['exact_beyond_96']=exact_cohort(traces)
                    row[measurement]=metric
                    (a.root/f'{signature}_{measurement}_traces.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in traces))
                persist()
                print(json.dumps(dict(checkpoint=name,measurement=measurement)),flush=True)
        output[name]=dict(step=saved['step'],signature=signature,**row)
        output_path.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    main()
