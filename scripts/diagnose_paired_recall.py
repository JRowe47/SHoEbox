"""Audit retained content and query credit without changing model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.task_stream import TaskStreamConfig
from experiments.text_regions.data import split_bytes
from experiments.text_regions.paired_recall import paired_batch
from experiments.text_regions.ports import tensor_list
from scripts.run_memory_selectivity import write_json


def features(state):
    groups=dict(ingress=state.ingress,nodes=state.nodes,ports=state.ports,
                feedback=(state.recall_feedback,state.publication_demand))
    result={}
    for name,group in groups.items():
        tensors=[x.flatten(1).double() for x in tensor_list(group) if x.is_floating_point()]
        if tensors: result[name]=torch.cat(tensors,1)
    result['all']=torch.cat(list(result.values()),1)
    return result


def signal_size(value,size):
    x=value.reshape(-1,size,size,value.shape[-1])
    return dict(old_centered_rms=float((x-x.mean(1,keepdim=True)).square().mean().sqrt()),
        new_centered_rms=float((x-x.mean(2,keepdim=True)).square().mean().sqrt()),
        state_rms=float(x.square().mean().sqrt()),features=x.shape[-1])


@torch.no_grad()
def snapshots(model,data,text,*,split,start,count,gap,seed):
    result={}; answers=[]; keys=[]
    # Bound peak memory while amortizing per-event execution across templates.
    for offset in range(0,count,4):
        batch=paired_batch(data,split=split,seed=seed,start=start+offset,
            count=min(4,count-offset),gap=gap,distractor_bytes=text)
        x=batch['history']; boundary=batch['sources'][0][0]+2
        second=batch['sources'][0][1]+2
        state=None
        stages=[('first_write',0,boundary),('second_write',boundary,second)]
        if gap: stages.append(('before_query',second,x.shape[1]))
        for stage,left,right in stages:
            state=model(x[:,left:right],state=state)['state']
            for group,value in features(state).items():
                result.setdefault(stage,{}).setdefault(group,[]).append(value)
        if not gap:
            for group,value in features(state).items():
                result.setdefault('before_query',{}).setdefault(group,[]).append(value)
        answers.append(batch['answers'].reshape(-1,2)-ord(data.values[0]))
        keys.append(batch['query'][:,0].reshape(-1,2))
    return {stage:{name:torch.cat(parts) for name,parts in groups.items()}
            for stage,groups in result.items()},torch.cat(answers),torch.cat(keys)


def ridge_readout(train,test,labels,targets,size,alpha=1.):
    # A post-hoc diagnostic only. The feature floor prevents arbitrary
    # amplification of effectively constant state coordinates.
    center=train.mean(0); scale=train.std(0,unbiased=False).clamp_min(1e-3)
    x=(train-center)/scale; z=(test-center)/scale
    y=F.one_hot(labels,num_classes=size).double().flatten(1)
    mean=y.mean(0); y=y-mean
    coefficient=torch.linalg.solve(x@x.T+alpha*torch.eye(len(x),dtype=x.dtype),y)
    scores=(z@x.T@coefficient+mean).reshape(-1,2,size)
    correct=scores.argmax(-1).eq(targets)
    return dict(older_accuracy=float(correct[:,0].double().mean()),
                newer_accuracy=float(correct[:,1].double().mean()),
                both_accuracy=float(correct.all(-1).double().mean()),
                train_rows=len(train),test_rows=len(test),alpha=alpha,feature_scale_floor=.001)


def query_readout(train,test,labels,targets,keys,test_keys,size,alpha=1.):
    """Diagnostic key-conditioned linear readout; no temporal-position hint.

    Independent coefficient sets for each literal query key are equivalent to
    a linear probe on state tensor-product query-one-hot. The probe is never
    installed in the language model. Center/scale use fit data only.
    """
    center=train.mean(0); scale=train.std(0,unbiased=False).clamp_min(.001)
    x=(train-center)/scale; z=(test-center)/scale
    prediction=torch.full_like(targets,-1)
    for key in test_keys.unique():
        fit_rows,fit_branch=(keys==key).nonzero(as_tuple=True)
        rows,branch=(test_keys==key).nonzero(as_tuple=True)
        if not len(fit_rows): raise ValueError('Probe fitting bank misses a query key.')
        inputs=x[fit_rows]
        y=F.one_hot(labels[fit_rows,fit_branch],num_classes=size).double()
        mean=y.mean(0)
        coefficient=torch.linalg.solve(inputs@inputs.T+alpha*torch.eye(len(inputs),dtype=x.dtype),y-mean)
        prediction[rows,branch]=(z[rows]@inputs.T@coefficient+mean).argmax(-1)
    correct=prediction.eq(targets); conflicts=targets[:,0]!=targets[:,1]
    return dict(older_accuracy=float(correct[:,0].double().mean()),
        newer_accuracy=float(correct[:,1].double().mean()),
        conflicting_both_accuracy=float(correct[conflicts].all(-1).double().mean()),
        conflicting_pairs=int(conflicts.sum()),alpha=alpha,feature_scale_floor=.001,
        query_key_source='literal query byte; diagnostic only')


def credit(model,batch):
    leaves=[]
    def record(module,args,output): leaves.append(output)
    hook=model.hat.byte_embedding.register_forward_hook(record)
    try:
        result=model(batch['full']); logits=result['logits'][:,-1]
        if len(leaves)!=batch['full'].shape[1]:
            raise ValueError('Expected one ingress embedding per event.')
        length=len(leaves)
        positions=dict(old_entity=0,old_attribute=2,old_value=4,
                       new_entity=6,new_attribute=8,new_value=10,
                       query_entity=length-4,query_attribute=length-2)
        answer_gradients={}
        for branch,name in enumerate(('older','newer')):
            loss=F.cross_entropy(logits[branch::2],batch['answers'][branch::2])
            gradients=torch.autograd.grad(loss,leaves,retain_graph=branch==0,allow_unused=True)
            answer_gradients[name]={key:0. if gradients[index] is None else
                float(gradients[index][branch::2].flatten(1).norm(dim=1).mean())
                for key,index in positions.items()}
        scores=logits.detach().reshape(-1,2,256)
        values=torch.tensor(list(map(ord,'0123')))
        answer_grid=scores.index_select(-1,values).reshape(4,4,2,4)
        old_signal=(answer_grid-answer_grid.mean(0,keepdim=True)).square().mean().sqrt()
        new_signal=(answer_grid-answer_grid.mean(1,keepdim=True)).square().mean().sqrt()
        return dict(answer_embedding_gradient_mean_l2=answer_gradients,
            value_logit_old_centered_rms=float(old_signal),
            value_logit_new_centered_rms=float(new_signal),
            query_switch_logit_rms=float((scores[:,0]-scores[:,1]).square().mean().sqrt()),
            query_switches=int((scores[:,0].argmax(-1)!=scores[:,1].argmax(-1)).sum()),
            histories=len(scores))
    finally:
        hook.remove(); model.zero_grad(set_to_none=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'data/tiny_shakespeare.txt')
    p.add_argument('--templates',type=int,default=16)
    a=p.parse_args(); torch.set_num_threads(1)
    if a.out.exists(): p.error('Diagnostic output already exists.')
    if a.templates<1: p.error('At least one template required.')
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    raw=a.data.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=saved['raw_sha256']: raise ValueError('Corpus mismatch.')
    model=RegionLanguageModel(GraphConfig.from_dict(saved['config']))
    model.load_state_dict(saved['weights']); model.eval()
    data=TaskStreamConfig.from_dict(saved['data_config'])
    if data.values!='0123' or data.attributes!=('x',):
        raise ValueError('This diagnostic is declared for the four-value, single-attribute phase.')
    train,validation,_=split_bytes(raw)
    result=dict(identity=dict(checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256((ROOT/'experiments/text_regions/paired_recall.py').read_bytes()).hexdigest(),
        raw_sha256=saved['raw_sha256'],seed=683,fit_start=40000,probe_test_start=50000,
        templates=a.templates,model_weights_changed=False),gaps={})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for gap in (0,32):
        fit,labels,keys=snapshots(model,data,train,split='train',start=40000,
                            count=a.templates,gap=gap,seed=683)
        test,targets,test_keys=snapshots(model,data,validation,split='validation',start=50000,
                              count=a.templates,gap=gap,seed=683)
        signals={stage:{name:signal_size(value,len(data.values)) for name,value in groups.items()}
                 for stage,groups in test.items()}
        probes={name:ridge_readout(value,test['before_query'][name],labels,targets,len(data.values))
                for name,value in fit['before_query'].items()}
        binding=query_readout(fit['before_query']['all'],test['before_query']['all'],
                              labels,targets,keys,test_keys,len(data.values))
        batch=paired_batch(data,split='validation',seed=683,start=60000,count=1,
                           gap=gap,distractor_bytes=validation)
        gradients=credit(model,batch)
        for name,value in model.state_dict().items():
            if not torch.equal(value,saved['weights'][name]):
                raise RuntimeError('Diagnostic changed model weights.')
        result['gaps'][str(gap)]=dict(state_signals=signals,probes=probes,
                                    query_conditioned_probe=binding,credit=gradients)
        write_json(a.out,result)
        print(json.dumps(dict(gap=gap,probes=probes,credit=gradients)),flush=True)


if __name__=='__main__': main()
