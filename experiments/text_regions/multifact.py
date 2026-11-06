"""Independent field memories feeding the existing two-round composer.

Field addresses are an explicit query interface, not inferred language. The
producer experts learn what to retain from raw interleaved source bytes. Shared
replicas use generic latest-publication arbitration; payloads stay continuous.
"""
from dataclasses import replace
import torch
from torch.nn import functional as F
from .config import GraphConfig, NodeSpec, EdgeSpec
from .model import RegionLanguageModel
from .multifact_data import OWNERS, SOURCE, FIELD, batch_digest
from .retained_composition import empty_inputs, masks, teacher_inputs, tables
from .ports import map_tensors

COUNTERS=('expert_executions','external_events','source_occurrences')
OTHER_FIELD=dict(F='L',L='F',G='R',R='G')


def edge_name(owner): return owner+'__A__forward'


def build(parent_path, seed):
    parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    base=GraphConfig.from_dict(parent['config'])
    if base.independent_sources != ('F',):raise ValueError('Expected the integrated single-producer parent.')
    cfg=replace(base,nodes=base.nodes+tuple(NodeSpec(n,'memory',n) for n in OWNERS if n!='F'),
                edges=base.edges+tuple(EdgeSpec(n,'A') for n in OWNERS if n!='F'),
                selective_publication=OWNERS,independent_sources=OWNERS,seed=seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=RegionLanguageModel(cfg)
    weights=model.state_dict()
    for name in weights:
        original=name
        for owner in OWNERS:
            for prefix in ('nodes.','independent_hats.','independent_banks.','connections.ingress.'):
                if name.startswith(prefix+owner+'.'):original=prefix+'F.'+name[len(prefix+owner+'.'):]
            if name.startswith('connections.edges.'+edge_name(owner)+'.'):
                original=name.replace(edge_name(owner),edge_name('F'),1)
        weights[name]=parent['weights'][original].clone()
    # Vocabulary-equivalent initialization: private fields use ':' where the
    # parent used '='. Only learned input-embedding rows are permuted; runtime
    # bytes are untouched and both kinds of field reach both source producers.
    for owner in ('L','R'):
        key='independent_hats.'+owner+'.byte_embedding.weight'
        weights[key][[58,61]]=weights[key][[61,58]].clone()
    model.load_state_dict(weights);configure(model)
    return model


def trainable(name):
    return any(name.startswith(prefix+n+'.') for n in OWNERS
               for prefix in ('nodes.','independent_hats.','independent_banks.','connections.ingress.')) or any(
               name=='connections.edges.'+edge_name(n)+'.value.weight' for n in OWNERS)


def configure(model):
    for name,p in model.named_parameters():p.requires_grad_(trainable(name))


def optimizer_for(model,producer_lr=.0003,edge_lr=.001):
    configure(model)
    return torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if p.requires_grad and '.edges.' in n], 'lr':edge_lr},
        {'params':[p for n,p in model.named_parameters() if p.requires_grad and '.edges.' not in n], 'lr':producer_lr}],
        betas=(.9,.95),weight_decay=.01)


def segment(model,batch,start,end,state=None,capture=False,policy='learned'):
    if policy not in ('learned','always'):raise ValueError('Unknown publication policy.')
    n,t=batch.inputs[:,start:end,0].shape;inputs=empty_inputs(model,n,t)
    available={v:torch.zeros(n,t,dtype=torch.bool) for v in inputs}
    active=masks(model,n,t,())
    for owner in OWNERS:
        inputs[owner]=batch.inputs[:,start:end,SOURCE[owner]]
        available[owner]=batch.available[:,start:end,SOURCE[owner]]
        active[owner]=available[owner]
    return model(inputs,state=state,reset=batch.reset[:,start:end],availability=available,node_masks=active,
        rounds=1,feedback=False,return_node_activity=OWNERS if capture else (),return_publication_activity=capture,
        publication_overrides={n:'always' for n in OWNERS} if policy=='always' else {})


def ingest(model,batch,state=None,policy='learned',start=0,end=None):
    end=batch.inputs.shape[1] if end is None else end;costs=dict.fromkeys(COUNTERS,0)
    for offset in range(start,end,16):
        result=segment(model,batch,offset,min(offset+16,end),state,policy=policy)
        state=result['state'].detach()
        for k in costs:costs[k]+=result[k]
    return state,costs


def workspace(state):
    fresh=state.reset(torch.ones_like(state.epochs,dtype=torch.bool))
    return replace(fresh,ingress={**fresh.ingress,**{n:state.ingress[n] for n in OWNERS}},
        nodes={**fresh.nodes,**{n:state.nodes[n] for n in OWNERS}},
        ports=fresh.ports.commit({n:state.ports.latest[n] for n in (*OWNERS,*('@'+n for n in OWNERS))}),
        events_seen=state.events_seen,rounds_seen=state.rounds_seen)


def select_owner(state,address,policy='latest'):
    """Read an explicit logical field address; never inspect the fact value."""
    if address.ndim!=1 or address.dtype!=torch.long or not bool(((address>=0)&(address<3)).all()):
        raise ValueError('Field addresses must be int64 [B] values 0, 1, or 2.')
    if policy not in ('latest','left','right'):raise ValueError('Unknown shared-source policy.')
    left,right=state.ports.latest['F'],state.ports.latest['G']
    # Occurrence is monotonic over live queries and selective workspace resets.
    use_right=right.valid & (~left.valid | (right.occurrence>left.occurrence))
    if policy=='left':use_right=torch.zeros_like(use_right)
    if policy=='right':use_right=torch.ones_like(use_right)
    common=torch.where(use_right,2,0)
    return torch.where(address==0,common,torch.where(address==1,1,3))


def query(model,first,second,state,address,rounds=2,feedback=True,control='intact',shared_policy='latest',trace=False):
    if first.shape!=second.shape or first.ndim!=2 or first.shape[1]!=4 or address.shape!=(first.shape[0],):
        raise ValueError('Aligned four-byte tables and one field address per row required.')
    if control not in ('intact','no_edge','reset_memory','shuffle_memory'):raise ValueError('Unknown query control.')
    state=workspace(state);n=first.shape[0]
    if control=='reset_memory':state=state.reset(torch.ones(n,dtype=torch.bool))
    if control=='shuffle_memory':
        state=replace(state,ports=state.ports.commit({o:map_tensors(state.ports.latest[o],lambda x:x.roll(1,0)) for o in OWNERS}))
    selected=select_owner(state,address,shared_policy)
    inputs=empty_inputs(model,n,4);inputs.update(B=first,C=second)
    available={v:torch.full((n,4),v in ('B','C'),dtype=torch.bool) for v in inputs}
    prefix=model(inputs,state=state,availability=available,node_masks=masks(model,n,4,('B','C')),rounds=1,feedback=False)
    inputs=empty_inputs(model,n,1)
    for v in ('B','C'):inputs[v].fill_(ord('?'))
    available={v:torch.full((n,1),v in ('B','C'),dtype=torch.bool) for v in inputs}
    edge_masks={edge_name(o):(selected.eq(i)&(control!='no_edge'))[:,None] for i,o in enumerate(OWNERS)}
    result=model(inputs,state=prefix['state'],availability=available,edge_masks=edge_masks,
        node_masks=masks(model,n,1,tuple(o for o in model.nodes if o not in OWNERS)),rounds=rounds,feedback=feedback,trace=trace)
    for k in COUNTERS:result[k]+=prefix[k]
    result['selected_owner']=selected
    return result


def project(model,owner,payload):
    edge=model.connections.edges[edge_name(owner)]
    return edge.gate(payload).sigmoid()*edge.value(payload)


def backward(model,batch,table_batches,weight=1.,reference=None):
    """Accumulate the segmented objective without changing weights or clearing gradients."""
    if weight <= 0:raise ValueError('A positive objective weight is required.')
    reference=teacher_inputs(model,torch.arange(65,69)) if reference is None else reference.detach()
    scale=reference.square().mean().clamp_min(1e-8)
    denominators={o:(int(batch.retained[o].ne(-100).sum()),int(batch.writes[o].sum()),
        int(batch.writes[OTHER_FIELD[o]].sum()),
        int((~batch.writes[o]&~batch.writes[OTHER_FIELD[o]]&batch.available[:,:,SOURCE[o]]).sum())) for o in OWNERS}
    if any(min(v)<1 for v in denominators.values()):raise ValueError('Every owner needs observed write/hold events.')
    state=None;metrics=dict(retention=0.,candidate=0.,gate=0.,regularizer=0.,nats=0.);costs=dict.fromkeys(COUNTERS,0)
    for start in range(0,batch.inputs.shape[1],16):
        end=min(start+16,batch.inputs.shape[1]);r=segment(model,batch,start,end,state,True)
        local=r['aux']*0
        for owner in OWNERS:
            known=batch.retained[owner][:,start:end].ne(-100);writes=batch.writes[owner][:,start:end]
            other=batch.writes[OTHER_FIELD[owner]][:,start:end]
            observed=batch.available[:,start:end,SOURCE[owner]];holds=~writes&~other&observed
            nk,nw,no,nh=denominators[owner];activity=r['publication_activity'][owner]
            payload=r['node_activity'][owner]['payload'];candidate=activity['candidate'][:,:,0];logit=activity['logit'][:,:,0]
            retention=(project(model,owner,payload[known])-reference[batch.retained[owner][:,start:end][known]-65]).square().sum()/(nk*model.cfg.width*scale)
            acquisition=(project(model,owner,candidate[writes])-reference[batch.inputs[:,start:end,SOURCE[owner]][writes]-65]).square().sum()/(nw*model.cfg.width*scale)
            # Opposite-field values are the consequential negative examples.
            # Give them their own denominator instead of diluting them among
            # dozens of easy filler bytes.
            gate=.5*F.softplus(-logit[writes]).sum()/nw+.25*F.softplus(logit[other]).sum()/no+.25*F.softplus(logit[holds]).sum()/nh
            for key,value in (('retention',retention),('candidate',acquisition),('gate',gate)):
                local=local+value/4;metrics[key]+=float(value.detach())/4
        regularizer=(model.cfg.auxiliary_loss_weight*r['aux']+model.cfg.hat_balance_weight*r['hat_aux'])*(end-start)/batch.inputs.shape[1]
        local=local+regularizer;metrics['regularizer']+=float(regularizer.detach())
        for k in costs:costs[k]+=r[k]
        if end==batch.inputs.shape[1]:
            for field,tb in enumerate(table_batches):
                result=query(model,tb.first,tb.second,r['state'],torch.full_like(batch.answers[:,0],field))
                ce=F.cross_entropy(result['logits'][:,-1],tb.answers)/3;local=local+ce;metrics['nats']+=float(ce.detach())
                for k in costs:costs[k]+=result[k]
        (weight*local).backward();state=r['state'].detach()
    return dict(loss=sum(metrics.values()),batch_sha256=batch_digest(batch),
                table_keys=[tb.keys for tb in table_batches],**metrics,**costs)


def update(model,optimizer,batch,table_batches):
    optimizer.zero_grad(set_to_none=True)
    metrics=backward(model,batch,table_batches)
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(gradient_norm=float(norm),**metrics)


def table_batches(batch,split,start,seed,request=0):
    split='test' if split=='independent' else split
    return [tables(split,start,len(batch.answers),seed,batch.answers[:,field],request*3+field) for field in range(3)]
