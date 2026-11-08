"""CPU graph executor: hard skips, per-row recipes and dependency-safe waves."""
from collections import defaultdict
from dataclasses import replace
import torch
from .config import NAMES
from .transport import Frame
from .hippocampus import SparseMemory
from .runtime import parallel_regions
from .sparse_state import (vector,validity,empty_frame,take_frame,scatter_frame,
                           publish_rows,take_node,scatter_node,scatter_tensor)


def memory_returns(model, state, latents, selected, revision):
    batch, event = state.batch,state.event
    sample = next(model.parameters())
    content = latents.get('EC-C',sample.new_zeros(batch,model.cfg.for_region('EC-C').width))
    context = latents.get('EC-X',sample.new_zeros(batch,model.cfg.for_region('EC-X').width))
    outputs = {'EC-C': [], 'EC-X': []}
    for row in selected.nonzero().flatten().tolist():
        rows = torch.tensor([row],device=content.device)
        result = model.hippocampus.read(content[row:row+1],context[row:row+1],state.memory.streams[row],event,revision-1)
        for owner,frames in result.items():
            for index,frame in enumerate(frames):
                if len(outputs[owner]) <= index:
                    outputs[owner].append(empty_frame(frame.source,batch,frame.width,rows.device))
                outputs[owner][index] = scatter_frame(outputs[owner][index],frame,rows)
    return outputs


def sparse_event(model, values, state, source, trace, continuation, memory_on, observer):
    from .model import CorticalState
    b,event = state.batch,state.event
    device = values.device
    nodes,ports,latents = dict(state.nodes),dict(state.ports),dict(state.latents)
    observed = Frame(f'external:{source}',event,0,values)
    previous = state.output
    if continuation is not None:
        previous = None if bool((continuation == -1).all()) else Frame('Y-output',event-1,0,continuation)
    records,losses,edge_losses = [],[],[]
    statistics = {'region_rows':{n:0 for n in NAMES},'hat_byte_steps':0,'region_calls':0,
                  'hat_cohort_calls':0,'edge_rows':0,'blocked_rows':0}

    def accept(name, rows, frames, result, revision, gate=None, recipes=None):
        nonlocal ports
        if gate is not None:
            result['frames'] = {w:replace(f,strength=gate if f.strength is None else f.strength*gate)
                                for w,f in result['frames'].items()}
        nodes[name] = scatter_node(nodes[name],result['state'],rows)
        old = latents.get(name,result['latent'].new_zeros(b,result['latent'].shape[-1]))
        latents[name] = scatter_tensor(old,result['latent'],rows)
        ports = publish_rows(ports,name,result['frames'],rows,b)
        losses.append(result['aux']*len(rows))
        statistics['region_rows'][name] += len(rows)
        recipes = [(rows,frames)] if recipes is None else [(subset,inputs) for subset,inputs,_ in recipes]
        statistics['hat_cohort_calls'] += len(recipes)
        statistics['hat_byte_steps'] += sum(len(subset)*sum(f.width for f in inputs) for subset,inputs in recipes)
        if observer is not None and (name in ('A1','B1') or revision == model.cfg.rounds-1):
            observer(name,event,revision,result)
        if trace:
            cursor=0
            for subset,inputs in recipes:
                records.append({'region':name,'event':event,'round':revision,'rows':subset.tolist(),
                                'inputs':[f.trace() for f in inputs],
                                'experts':result['experts'][cursor:cursor+len(subset)].cpu().tolist()})
                cursor+=len(subset)

    rows = torch.arange(b,device=device)
    def ingress(name):
        result = model.regions[name](name,[observed],nodes[name],event,0,observe=observer is not None)
        return name,result
    for name,result in parallel_regions(ingress,['A1','B1']):
        statistics['region_calls'] += 1
        accept(name,rows,[observed],result,0)
    query,active,node_gates,router_loss,probabilities = model.graph_router.regions(latents,b)
    statistics['selected'] = active.detach()
    statistics['probabilities'] = probabilities.detach()
    all_logits = None

    for revision in range(model.cfg.rounds):
        snapshot = ports
        returns = memory_returns(model,state,latents,active[:,NAMES.index('EC-C')],revision) if memory_on else {'EC-C':[],'EC-X':[]}
        for wave in model.execution_waves:
            jobs = []
            for spec in wave:
                if spec.order == 1:
                    continue
                selected_rows = active[:,NAMES.index(spec.name)].nonzero().flatten()
                if not selected_rows.numel():
                    continue
                candidates = [(snapshot if edge.delayed else ports).get(f'{edge.source}:{edge.width}', ())[-edge.count:]
                              for edge in spec.inputs]
                available = torch.stack([torch.stack([validity(f) for f in history]).any(0) if history else
                                         torch.zeros(b,dtype=torch.bool,device=device) for history in candidates],-1)
                chosen,weights,edge_loss = model.graph_router.edges(spec,query.index_select(0,selected_rows),
                                                                   available.index_select(0,selected_rows))
                edge_losses.append(edge_loss)
                choices = chosen.detach().tolist()
                # Recipes are grouped by actual available frame slots as well
                # as selected edges. No padding step enters any recurrent HAT.
                groups = defaultdict(list)
                validity_lists = [[validity(f).tolist() for f in history] for history in candidates]
                extra = returns.get(spec.name,[])
                extra_valid = [validity(f).tolist() for f in extra]
                prev_valid = validity(previous).tolist() if previous is not None and spec.name=='Y' else None
                for local,row in enumerate(selected_rows.tolist()):
                    recipe = []
                    for edge,picked in enumerate(choices[local]):
                        if picked:
                            for slot,valid in enumerate(validity_lists[edge]):
                                if valid[row]: recipe.append(('edge',edge,slot))
                    for slot,valid in enumerate(extra_valid):
                        if valid[row]: recipe.append(('memory',slot,0))
                    if prev_valid is not None and prev_valid[row]: recipe.append(('previous',0,0))
                    if recipe:
                        groups[tuple(recipe)].append((local,row))
                    else:
                        statistics['blocked_rows'] += 1
                cohorts = []
                for recipe,members in groups.items():
                    local = torch.tensor([p[0] for p in members],device=device)
                    subset = torch.tensor([p[1] for p in members],device=device)
                    frames = []
                    for kind,index,slot in recipe:
                        if kind == 'edge':
                            frame = take_frame(candidates[index][slot],subset)
                            gate = weights.index_select(0,local)[:,index]
                            frame = replace(frame,strength=gate if frame.strength is None else frame.strength*gate)
                        elif kind == 'memory':
                            frame = take_frame(extra[index],subset)
                        else:
                            frame = take_frame(previous,subset)
                        frames.append(frame)
                    statistics['edge_rows'] += len(subset)*sum(choices[members[0][0]])
                    cohorts.append((subset,frames,take_node(nodes[spec.name],subset)))
                if cohorts:
                    jobs.append((spec.name,cohorts))

            def execute(job):
                name,cohorts = job
                if len(cohorts)>1:
                    output = model.regions[name](name,[],None,event,revision,
                                                  observe=observer is not None and revision==model.cfg.rounds-1,
                                                  cohorts=cohorts)
                    return name,torch.cat([c[0] for c in cohorts]),[],output,cohorts
                subset,frames,previous_node=cohorts[0]
                output=model.regions[name](name,frames,previous_node,event,revision,
                                            observe=observer is not None and revision==model.cfg.rounds-1)
                return name,subset,frames,output,None

            for name,subset,frames,result,recipes in parallel_regions(execute,jobs):
                statistics['region_calls'] += 1
                gate = node_gates.index_select(0,subset)[:,NAMES.index(name)]
                accept(name,subset,frames,result,revision,gate,recipes)
                if name == 'Y':
                    if all_logits is None:
                        all_logits = result['logits'][1].new_zeros(b,256)
                    all_logits = scatter_tensor(all_logits,result['logits'][1][:,0],subset)

    if all_logits is None or not bool(validity(ports['Y:1'][-1]).all()):
        raise RuntimeError('The mandatory live output path did not execute.')
    output = replace(ports['Y:1'][-1],source='Y-output',origin_source='Y')
    memory = state.memory
    if memory_on:
        streams = list(memory.streams)
        c = ports.get(f'EC-C:{model.cfg.max_port_bytes}')
        x = ports.get(f'EC-X:{model.cfg.max_port_bytes}')
        if c and x:
            fresh = validity(c[-1]) & validity(x[-1]) & (vector(c[-1].event,b,device)==event) & (vector(x[-1].event,b,device)==event)
            for row in fresh.nonzero().flatten().tolist():
                subset = torch.tensor([row],device=device)
                streams[row] = model.hippocampus.write(take_frame(c[-1],subset,scalar=True),take_frame(x[-1],subset,scalar=True),
                                                       latents['EC-C'][row:row+1],latents['EC-X'][row:row+1],streams[row])
        memory = SparseMemory(tuple(streams))
    count = sum(statistics['region_rows'].values())
    aux = torch.stack(losses).sum()/count + router_loss
    if edge_losses:
        aux = aux+torch.stack(edge_losses).mean()
    return {'logits':all_logits,'aux':aux,'trace':records,'routing':statistics,
            'state':CorticalState(b,nodes,ports,memory,latents,output,event+1,state.signature)}
