"""Event-clock admission with two detail levels and bounded identity maps."""
from dataclasses import replace
import torch
from .hippocampus_interface import EngramHandle, LocalAnchor, BindingOffer, AnchorLink, BindingHandle, BindingRecord
from .hippocampus_index import WorkBudget


def event_offers(model, state, keys, sources, latents, active, grounded):
    service = model.hippocampus
    result = [[] for _ in state.hippocampus.rows]
    for ri,region in enumerate(service.regions):
        owners = service.owners[region]
        values = torch.stack([latents[n] for n in owners],1)
        trace = state.hippocampus.cues[:,ri,None].expand_as(values)
        cue = keys[:,ri,None].expand(-1,len(owners),-1)
        key,role,write,erase = service.interface.offer(trace,values,cue)
        eligible = torch.stack([active[n] for n in owners],1)
        pick = write.squeeze(-1).masked_fill(~eligible,-1).argmax(-1)
        for bi,row in enumerate(state.hippocampus.rows):
            if not bool(grounded[bi,ri]) or not bool(eligible[bi].any()):
                continue
            # Source/abstract levels share this transport event identity. It has
            # no question labels or semantic entity IDs and is never a feature.
            event = int(state.events_seen[bi])
            event_id = (row.namespace,region,event)
            abstract = int(pick[bi])
            source = int(eligible[bi].nonzero()[0])
            for level,oi in enumerate((source,abstract)):
                owner = owners[oi]
                detail = 'source' if level == 0 else 'abstract'
                value = sources[bi,ri] if level == 0 else values[bi,oi]
                k = keys[bi,ri] if level == 0 else key[bi,oi]
                roles = role[bi,oi]+service.detail_roles.weight[level]
                capsule = model.nodes[owner].encode_anchor(value)
                parents = () if level == 0 else model.memory_interface.parents(state.memory_context[owner],bi)
                provenance = 'external' if level == 0 else ('mixed' if parents else 'derived')
                h = EngramHandle(row.namespace,region,owner,-1,0,0,service.config.representation_version)
                version = event if level == 0 else int(state.nodes[owner].clock[bi])
                sources_used = (tuple((p.source,int(p.epoch[bi]),int(p.version[bi]),int(p.occurrence[bi]))
                    for view in (('text',) if state.input_mode == 'text' else model.memory_views[region])
                    for p in (state.ports.latest['@'+view],) if bool(p.valid[bi])) if level == 0 else model.memory_interface.sources(state.memory_context[owner],bi))
                anchor = LocalAnchor(h,capsule,k,roles,provenance,row.round+1,version,
                                     event,event_id,detail,parents,sources_used)
                code,_ = service.separate(k)
                actions = service.interface.actions(torch.cat((k,keys[bi,ri],roles))).softmax(-1)
                result[bi].append(BindingOffer((row.namespace,row.round,owner,detail),row.namespace,
                    owner,anchor,k,code,keys[bi,ri],write[bi,oi,0],erase[bi,oi,0],actions,row.round+1,provenance))
    return tuple(tuple(x) for x in result)


def admit_event(service, row, offer):
    """Revise same-event pages in place; new events allocate bounded pages."""
    from .hippocampus import hard_gate
    c = service.config
    if offer.namespace != row.namespace or offer.ready_round != row.round+1 or offer.provenance == 'reinstated':
        return row
    a = offer.anchor
    producer = (offer.producer,a.detail)
    stamp = offer.offer_id[1]
    if stamp <= row.committed.get(producer,-1):
        return row
    committed = dict(row.committed); committed[producer] = stamp
    row = replace(row,committed=committed)
    identity = (*a.event_id,a.detail)
    old_handle = row.event_anchors.get(identity)
    old = row.anchor(old_handle,row.round+1) if old_handle is not None else None
    # Identical transport source versions (including repeated source-level
    # computation) refresh no page and consume no capacity.
    if old is not None and old.handle.node == a.handle.node and (
            old.source_version == a.source_version or torch.equal(old.capsule.detach(),a.capsule.detach())):
        return row
    region = a.handle.region
    bootstrap = c.bootstrap and not row.anchors.get(region)
    action = int(offer.actions.detach().argmax())
    if (c.admission and not bootstrap and float(offer.write.detach()) < .5) or (action == 3 and not bootstrap):
        # Preserve a closed write exactly, while providing normal task credit
        # through an existing same-event content path inside the live window.
        if old is not None and c.admission:
            local = dict(row.anchors[region])
            local[old.handle.slot] = replace(old,capsule=old.capsule+(offer.write-offer.write.detach())*(a.capsule-old.capsule))
            anchors = dict(row.anchors); anchors[region] = local
            row = replace(row,anchors=anchors)
        return row
    counters, identities = dict(row.anchor_counters), dict(row.event_anchors)
    local = dict(row.anchors.get(region,{}))
    if old is None:
        serial = counters.get(region,0); counters[region] = serial+1
        slot = serial%c.anchors_per_region
        evicted = local.get(slot)
        if evicted is not None and evicted.event_id:
            identities.pop((*evicted.event_id,evicted.detail),None)
        handle = replace(a.handle,slot=slot,generation=serial//c.anchors_per_region+1,version=1)
    else:
        handle = replace(a.handle,slot=old.handle.slot,generation=old.handle.generation,version=old.handle.version+1)
    gate = offer.write.new_ones(()) if bootstrap or not c.admission else hard_gate(offer.write)
    anchor = replace(a,handle=handle,capsule=a.capsule*gate)
    local.pop(handle.slot,None); local[handle.slot] = anchor
    anchors = dict(row.anchors); anchors[region] = local; identities[identity] = handle
    row = replace(row,anchors=anchors,anchor_counters=counters,event_anchors=identities)
    if c.mode == 'local':
        return row
    event_bindings = dict(row.event_bindings)
    prior = event_bindings.get(a.event_id)
    target = row.binding(prior,row.round+1) if prior is not None else None
    budget = WorkBudget(c)
    candidates = service.directory.lookup(row,offer.key,offer.code.detach().nonzero().flatten().tolist(),row.round+1,budget,stamp)
    if target is None and c.revision and action in (0,1):
        compatible = [r for r in candidates if any(l.handle.node == offer.producer for l in r.links)]
        if compatible:
            target = max(compatible,key=lambda r:float((r.key*offer.key).detach().sum()))
    counts = dict(row.binding_counters)
    if target is None:
        shard = service.directory.route(offer.key,WorkBudget(c),explore=False)[0]
        serial = counts.get(shard,0); counts[shard] = serial+1
        slot = serial%c.bindings_per_shard
        target_page = row.bindings.get(shard,{}).get(slot)
        handle_b = BindingHandle(row.namespace,shard,slot,1 if target_page is None else target_page.handle.generation+1,1)
        links, created = (),offer.ready_round
    else:
        target_page = target
        handle_b = replace(target.handle,version=target.handle.version+1)
        # Level identity, not producer identity alone, keeps direct and abstract
        # representations distinct even when one node owns both.
        links = tuple(l for l in target.links if l.handle != old_handle
            and row.anchor(l.handle,row.round+1) is not None)
        created = target.created
    if target_page is not None and target_page.event_id:
        event_bindings.pop(target_page.event_id,None)
    strength = 1+offer.actions[action]-offer.actions[action].detach()
    links = (*links,AnchorLink(handle,offer.key,anchor.role,strength))[-c.links_per_binding:]
    record = BindingRecord(handle_b,offer.key,offer.code,offer.context,links,
        tuple(r.handle for r in candidates if r is not target)[:c.neighbors],offer.key.new_zeros(()),
        offer.write,created,offer.ready_round,c.representation_version,a.event_id)
    bucket = dict(row.bindings.get(handle_b.shard,{})); bucket[handle_b.slot] = record
    bindings = dict(row.bindings); bindings[handle_b.shard] = bucket
    postings,recent = service.directory.install(row,record,target_page)
    event_bindings[a.event_id] = handle_b
    return replace(row,bindings=bindings,binding_counters=counts,postings=postings,recent=recent,event_bindings=event_bindings)
