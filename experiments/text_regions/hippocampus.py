"""Shallow DG/CA3/CA1 policy; all durable state is passed explicitly.

Sparse allocation/routing use hard decisions. DG, selected-link weights and
admitted writes use declared straight-through estimators. Credit is exact through
live tensor paths under those estimators, truncated at optimizer boundaries.
No historical content gradients are synthesized from transport identities.
"""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from .hippocampus_interface import (HippocampusState, MemoryRow, RegionalInterface,
    EngramHandle, BindingHandle, LocalAnchor, AnchorLink, BindingRecord,
    BindingOffer, CueRequest, ReinstatementRequest, CreditReceipt, RecallResult)
from .hippocampus_index import SparseDirectory, WorkBudget


def hard_gate(probability):
    return probability.ge(.5).to(probability.dtype) + (probability-probability.detach())


class Hippocampus(nn.Module):
    def __init__(self, graph_config):
        super().__init__()
        self.config = c = graph_config.hippocampus
        self.width = w = graph_config.width
        self.regions = tuple(graph_config.regions)
        self.owners = graph_config.regions
        self.interface = RegionalInterface(w,c)
        d = c.address_width
        self.dg = nn.Linear(d,c.dg_cells)
        self.context_score = nn.Linear(d,1,bias=False)
        self.age_utility = nn.Parameter(torch.tensor([-.02,.1]))
        self.null = nn.Linear(d,1)
        self.completion = nn.Linear(d,1)
        self.comparison = nn.Sequential(nn.Linear(d*5+2,d),nn.Tanh(),nn.Linear(d,3))
        self.return_descriptor = nn.Linear(w,d,bias=False)
        self.role_gate = nn.Linear(d,w,bias=False)
        self.compose = nn.Linear(c.readbacks*w,w,bias=False)
        self.publication_feedback_gain = nn.Parameter(torch.zeros(()))
        self.directory = SparseDirectory(c)
        # One zero factor preserves the warm-start logits; the nonzero readers,
        # selectors and role controls acquire gradients once this projection moves.
        nn.init.zeros_(self.compose.weight)
        nn.init.zeros_(self.null.weight)
        nn.init.constant_(self.null.bias,-1.)
        with torch.no_grad():
            self.comparison[-1].bias.copy_(torch.tensor([1.,-1.,0.]))

    def initial_state(self,batch,device,dtype):
        return HippocampusState(tuple(MemoryRow() for _ in range(batch)),
            torch.zeros(batch,len(self.regions),self.width,device=device,dtype=dtype))

    @staticmethod
    def snapshot(state):
        return state  # Immutable values and copy-on-write ownership containers.

    @staticmethod
    def reset_namespace(state,rows):
        return state.reset(rows)

    def read_local(self,row,request,nodes,context=None):
        anchor=row.anchor(request.handle,request.watermark)
        if anchor is None or request.handle.semantic_version != self.config.representation_version:
            return RecallResult(request.original_cue.new_zeros(self.width),(),'stale')
        value=nodes[request.handle.node].read_anchor(anchor.capsule[None],request.local_cue[None],
            None if context is None else context[None])[0]
        return RecallResult(value,((request.binding,request.handle),),'ok')

    def maintain(self,row):
        """Bounded touched-page repair/re-key after semantic weight drift.

        Re-key DG postings from preserved address descriptors. Incompatible
        semantic namespace versions are invalidated, never guessed or translated
        by transport. The declared budget is independent of storage capacity.
        """
        c=self.config
        for offset in range(c.maintenance_budget):
            position=row.round*c.maintenance_budget+offset
            shard=position%c.shards
            slot=(position//c.shards)%c.bindings_per_shard
            old=row.bindings.get(shard,{}).get(slot)
            if old is None:
                continue
            links=tuple(link for link in old.links if row.anchor(link.handle,row.round+1) is not None
                        and link.handle.semantic_version==c.representation_version)
            code,_=self.separate(old.key)
            updated=replace(old,links=links,code=code,representation_version=c.representation_version)
            # Reindexing leaves association identity and causal content version
            # intact; only descriptors/validity change in this new snapshot.
            bucket=dict(row.bindings[shard]); bucket[slot]=updated
            bindings=dict(row.bindings); bindings[shard]=bucket
            postings,recent=self.directory.install(row,updated,old)
            row=replace(row,bindings=bindings,postings=postings,recent=recent)
        return row

    def separate(self,key):
        scores = self.dg(key).tanh()
        probabilities = scores.softmax(-1)*self.config.dg_active
        cells = scores.topk(self.config.dg_active,dim=-1).indices
        hard = torch.zeros_like(scores).scatter(-1,cells,1.)
        return hard+(probabilities-probabilities.detach()),cells

    def advance_cues(self,state,ingress,available):
        b,r,w = ingress.shape
        trace,key = self.interface.cue(ingress.reshape(-1,w),state.cues.reshape(-1,w))
        trace = trace.reshape(b,r,w)
        trace = torch.where(available[...,None],trace,state.cues)
        key = key.reshape(b,r,-1)
        return replace(state,cues=trace),key

    def compare(self,query,expected,returned,valid,age,terminal=False):
        features = torch.cat((query,expected,returned,query-returned,expected-returned,
                              valid[...,None],age[...,None]),-1)
        logits = self.comparison(features)
        if terminal:
            # When the local reread budget is exhausted, only accept/null are
            # legal. Probability assigned to an unavailable refine action must
            # not silently suppress an otherwise accepted return.
            decisions = logits[...,:2].softmax(-1)
            return torch.cat((decisions,torch.zeros_like(decisions[...,:1])),-1)
        return logits.softmax(-1)

    def _candidates(self,row,request,code,cells,budget):
        if self.config.mode == 'shared':
            records = self.directory.lookup(row,request.cue,cells,request.watermark,budget,request.watermark)
            for _ in range(self.config.refinement_steps if self.config.control != 'no_refinement' else 0):
                records = self.directory.expand(row,records,request.watermark,budget)
            return records
        # The local arm visits only the requester's bounded anchor bank. It has
        # no directory, global bindings, coactivity links or CA3 neighbor graph.
        result = []
        anchors = row.anchors.get(request.requester_region,{})
        limit = min(self.config.posting_reads,self.config.candidates_per_shard)
        for anchor in reversed(tuple(anchors.values())):
            if len(result) >= limit:
                break
            if anchor.ready_round > request.watermark:
                continue
            handle = BindingHandle(row.namespace,0,anchor.handle.slot,anchor.handle.generation,anchor.handle.version)
            budget.posting(0); budget.candidate(handle)
            result.append(BindingRecord(handle,anchor.key,code.new_zeros(code.shape),anchor.role,
                (AnchorLink(anchor.handle,anchor.key,anchor.role,anchor.key.new_ones(())),),(),
                anchor.key.new_zeros(()),anchor.key.new_ones(()),anchor.ready_round,anchor.ready_round,
                anchor.handle.semantic_version))
        return result

    def query(self,state,keys,nodes,requesters,active,contexts=None,available_owners=None,regions=None):
        """Batched neural work, bounded metadata gathers, immutable causal reads.

        Returns residuals [B,R,W], receipts per world, and actual work counters.
        Query has no state mutation, so paired branches cannot contaminate peers.
        """
        c = self.config
        regions = self.regions if regions is None else tuple(regions)
        b,r,d = keys.shape
        if r != len(regions):
            raise ValueError('One key per requested regional endpoint is required.')
        q = keys.reshape(-1,d)
        codes,cells = self.separate(q)
        records,requests,budgets,worlds = [],[],[],[]
        for row_index,row in enumerate(state.rows):
            for region_index,region in enumerate(regions):
                j = row_index*r+region_index
                request = CueRequest((row.namespace,row.round,region),row.namespace,region,
                                     requesters[region],q[j],q[j],row.round)
                budget = WorkBudget(c)
                found = self._candidates(row,request,codes[j],cells[j].tolist(),budget) if bool(active[row_index,region_index]) else []
                # Transport validity masks descriptors as well as payloads.
                # These are bounded metadata checks, not owner content reads.
                checked=[]
                for record in found:
                    links=[]
                    for link in record.links:
                        allowed=(available_owners is None or bool(available_owners.get(
                            link.handle.node,active.new_zeros(b))[row_index]))
                        if (row.anchor(link.handle,request.watermark) is not None and allowed
                                and link.handle.semantic_version==c.representation_version
                                and record.representation_version==c.representation_version):
                            links.append(link)
                        else:
                            budget.stale+=1
                    if links:
                        checked.append(replace(record,links=tuple(links)))
                found=checked
                records.append(found); requests.append(request); budgets.append(budget); worlds.append(row)
        count = max(1,max(map(len,records)))
        if not any(records):
            return keys.new_zeros(b,r,self.width),tuple(() for _ in state.rows),[x.metrics() for x in budgets]
        def stack_feature(fn,empty):
            return torch.stack([torch.stack([fn(x) for x in items]+[empty]*(count-len(items))) for items in records])
        z = q.new_zeros(d)
        key_bank = stack_feature(lambda x:x.key,z)
        context_bank = stack_feature(lambda x:x.context,z)
        code_bank = stack_feature(lambda x:x.code,codes.new_zeros(c.dg_cells))
        utility = stack_feature(lambda x:x.utility,q.new_zeros(()))
        valid = torch.tensor([[True]*len(items)+[False]*(count-len(items)) for items in records],device=q.device)
        ages = q.new_tensor([[max(0,request.watermark-x.created) for x in items]+[0]*(count-len(items))
                             for request,items in zip(requests,records)]).log1p()
        base = (key_bank*q[:,None]).sum(-1)*4 + (code_bank*codes[:,None]).sum(-1)/c.dg_active
        base = base + self.context_score(context_bank*q[:,None]).squeeze(-1)
        base = base + self.age_utility[0]*ages + self.age_utility[1]*utility
        base = base.masked_fill(~valid,-1e4)
        null = self.null(q)
        scores = base
        adjacency = q.new_zeros(len(records),count,count)
        if c.mode == 'shared' and c.control != 'no_refinement':
            for j,items in enumerate(records):
                indices = {x.handle:k for k,x in enumerate(items)}
                for k,x in enumerate(items):
                    for neighbor in x.neighbors:
                        if neighbor in indices:
                            adjacency[j,k,indices[neighbor]] = 1.
            for _ in range(c.refinement_steps):
                probabilities = torch.cat((scores,null),-1).softmax(-1)[:,:count]
                expected = (probabilities[...,None]*key_bank).sum(1)
                refine = self.compare(q,expected,expected,valid.any(-1).float(),ages.mean(-1))[:,2]
                support = torch.bmm(adjacency,probabilities[...,None]).squeeze(-1)
                scores = base + self.completion(q).sigmoid()*refine[:,None]*support
                for budget in budgets:
                    budget.refinements += 1
        probs = torch.cat((scores,null),-1).softmax(-1)
        picks = torch.cat((scores,null),-1).topk(min(c.selected_bindings,count+1),-1).indices.tolist()
        read_capsules,read_cues,read_roles,read_weights,read_owners,contributors = [],[],[],[],[],[]
        for j,(items,indices,row,request,budget) in enumerate(zip(records,picks,worlds,requests,budgets)):
            entries=[]
            if c.read and c.control != 'blocked':
                for index in indices:
                    if index >= len(items):
                        # Null outranks remaining candidates: stop, rather than
                        # forcing a read solely because top-k has spare slots.
                        break
                    record = items[index]
                    for link in record.links:
                        if not budget.read(link.handle):
                            continue
                        anchor = row.anchor(link.handle,request.watermark)
                        allowed = available_owners is None or bool(available_owners.get(link.handle.node,active.new_zeros(b))[j//r])
                        if anchor is None or not allowed or link.handle.semantic_version != c.representation_version:
                            budget.stale += 1
                            continue
                        weight = 1.+(probs[j,index]-probs[j,index].detach())
                        entries.append((anchor,link,weight,record.handle))
            # Shuffle only causally valid selected links within this world.
            if c.control == 'shuffled' and len(entries)>1:
                anchors = [e[0] for e in entries]
                entries = [(anchors[(k+1)%len(entries)],e[1],e[2],e[3]) for k,e in enumerate(entries)]
            entries=entries[:c.readbacks]
            contributors.append(tuple((e[3],e[0].handle) for e in entries))
            for anchor,link,weight,binding in entries:
                read_capsules.append(anchor.capsule)
                read_cues.append(q[j]+link.cue)
                read_roles.append(link.role)
                read_weights.append(weight*link.strength)
                read_owners.append(anchor.handle.node)
            for _ in range(c.readbacks-len(entries)):
                read_capsules.append(q.new_zeros(self.width)); read_cues.append(z); read_roles.append(z)
                read_weights.append(q.new_zeros(())); read_owners.append('')
        capsules = torch.stack(read_capsules)
        cues = torch.stack(read_cues)
        values = capsules.new_zeros(capsules.shape)
        # Dispatch only selected owners; no scan of all regional payloads.
        for owner in dict.fromkeys(o for o in read_owners if o):
            indices = torch.tensor([i for i,o in enumerate(read_owners) if o == owner],device=q.device)
            local_context = None
            if contexts is not None:
                rows = (indices//(r*c.readbacks))
                local_context = contexts[owner].index_select(0,rows)
            values = values.index_copy(0,indices,nodes[owner].read_anchor(
                capsules.index_select(0,indices),cues.index_select(0,indices),local_context))
        roles = torch.stack(read_roles)
        values = values*(1.+self.role_gate(roles).tanh())*torch.stack(read_weights)[:,None]
        values = values.reshape(-1,c.readbacks,self.width)
        if c.control == 'pooled':
            values = values.mean(1,keepdim=True).expand_as(values)
        expected = (probs[:,:count,None]*key_bank).sum(1)
        returned = F.normalize(self.return_descriptor(values.sum(1)),dim=-1)
        has_return = q.new_tensor([bool(x) for x in contributors])
        can_refine = bool(c.refinement_steps and c.control != 'no_refinement')
        bounded = c.ca1_policy == 'bounded'
        comparison = self.compare(q,expected,returned,has_return,ages.mean(-1),
                                  terminal=bounded and not can_refine)
        accepted = comparison[:,0]
        # CA1's refine action conditions a second read of the SAME selected local
        # capsules, without another transport read or candidate-budget reset.
        # This is a shallow query-conditioned computation, not an extra graph round.
        if can_refine:
            refined_cues = cues + (q-returned)[:,None].expand(-1,c.readbacks,-1).reshape_as(cues)
            refined = capsules.new_zeros(capsules.shape)
            for owner in dict.fromkeys(o for o in read_owners if o):
                indices = torch.tensor([i for i,o in enumerate(read_owners) if o == owner],device=q.device)
                context = None if contexts is None else contexts[owner].index_select(0,indices//(r*c.readbacks))
                refined = refined.index_copy(0,indices,nodes[owner].read_anchor(
                    capsules.index_select(0,indices),refined_cues.index_select(0,indices),context))
            refined = refined*(1.+self.role_gate(roles).tanh())*torch.stack(read_weights)[:,None]
            refined = refined.reshape_as(values)
            if c.control == 'pooled':
                refined = refined.mean(1,keepdim=True).expand_as(refined)
            if bounded:
                refined_return = F.normalize(self.return_descriptor(refined.sum(1)),dim=-1)
                terminal = self.compare(q,expected,refined_return,has_return,ages.mean(-1),terminal=True)
                refined_accept = comparison[:,2]*terminal[:,0]
                # Accept the original read OR refine then accept the reread.
                # Do not multiply both branches by the original accept action.
                values = comparison[:,0,None,None]*values + refined_accept[:,None,None]*refined
                accepted = accepted + refined_accept
            else:
                values = values + comparison[:,2,None,None]*(refined-values)
        gain = torch.ones_like(accepted) if bounded and can_refine else accepted
        residual = self.compose(values.reshape(-1,c.readbacks*self.width))*gain[:,None]*has_return[:,None]
        receipts=[[] for _ in state.rows]
        for j,parts in enumerate(contributors):
            if parts:
                request=requests[j]
                receipts[j//r].append(CreditReceipt((*request.request_id,'credit'),request.request_id,
                    parts,accepted[j],state.window))
        return residual.reshape(b,r,self.width),tuple(tuple(x) for x in receipts),[x.metrics() for x in budgets]

    def feedback(self,row,receipts,window):
        if not self.config.feedback or self.config.control == 'no_feedback':
            return row
        bindings,credited = dict(row.bindings),dict(row.credited)
        for receipt in receipts:
            namespace,stamp,requester = receipt.request_id
            if namespace != row.namespace or receipt.window != window or stamp <= credited.get(requester,-1):
                continue
            credited[requester] = stamp
            for handle,anchor in receipt.contributors:
                record = row.binding(handle,row.round)
                if record is None or row.anchor(anchor,row.round) is None:
                    continue
                shard = dict(bindings.get(handle.shard,{}))
                # A usage signal, never a surrogate copy of the task gradient.
                current=shard[handle.slot]
                shard[handle.slot] = replace(current,utility=.95*current.utility+.05*receipt.support)
                bindings[handle.shard]=shard
        return replace(row,bindings=bindings,credited=credited)

    def commit(self,state,offers,receipts=()):
        rows=[]
        for i,row in enumerate(state.rows):
            row=self.feedback(row,receipts[i] if receipts else (),state.window)
            for offer in offers[i]:
                row=self.offer(row,offer)
            if self.config.mode=='shared':
                row=self.maintain(row)
            rows.append(replace(row,round=row.round+1))
        return replace(state,rows=tuple(rows))

    def offer(self,row,offer):
        c=self.config
        if (offer.namespace != row.namespace or offer.provenance == 'reinstated'
                or offer.ready_round != row.round+1):
            return row
        producer,stamp = offer.producer,offer.offer_id[1]
        if stamp <= row.committed.get(producer,-1):
            return row
        committed=dict(row.committed); committed[producer]=stamp
        row=replace(row,committed=committed)
        region=offer.anchor.handle.region
        empty=not row.anchors.get(region)
        bootstrap=c.bootstrap and empty
        if c.admission and not bootstrap and float(offer.write.detach()) < .5:
            # Hard hold is exactly a no-write in forward execution. Its tangent
            # reaches a bounded, existing owner capsule, so a closed admission
            # gate can still learn from future task use inside the live window.
            local=dict(row.anchors.get(region,{}))
            previous=next((a for a in reversed(tuple(local.values())) if a.handle.node==producer),None)
            if previous is not None:
                tangent=offer.write-offer.write.detach()
                local[previous.handle.slot]=replace(previous,capsule=previous.capsule+tangent*(offer.anchor.capsule-previous.capsule))
                anchors=dict(row.anchors); anchors[region]=local
                return replace(row,anchors=anchors)
            return row
        action=int(offer.actions.detach().argmax())
        if action == 3 and not bootstrap:
            return row
        # Capsules are ready before the index can refer to them. New versions
        # always allocate a new identity; old snapshots retain immutable pages.
        counters=dict(row.anchor_counters)
        serial=counters.get(region,0); counters[region]=serial+1
        slot=serial%c.anchors_per_region
        handle=replace(offer.anchor.handle,slot=slot,generation=serial//c.anchors_per_region+1,
                       version=1,semantic_version=c.representation_version)
        gate=offer.write.new_ones(()) if bootstrap or not c.admission else hard_gate(offer.write)
        anchors=dict(row.anchors); local=dict(anchors.get(region,{}))
        anchor=replace(offer.anchor,handle=handle,capsule=offer.anchor.capsule*gate)
        local.pop(slot,None); local[slot]=anchor; anchors[region]=local
        row=replace(row,anchors=anchors,anchor_counters=counters)
        if c.mode == 'local':
            return row
        budget=WorkBudget(c)
        cells=offer.code.detach().nonzero().flatten().tolist()
        candidates=self.directory.lookup(row,offer.key,cells,row.round+1,budget,stamp)
        target=None
        if c.revision and action in (0,1) and candidates:
            # Identity is predicted from learned descriptors, never from slot,
            # entity labels or arrival order. Do not merge incomparable producers.
            compatible=[x for x in candidates if any(l.handle.node==producer for l in x.links)]
            if compatible:
                target=max(compatible,key=lambda x:float((x.key*offer.key).detach().sum()))
        shard=target.handle.shard if target else self.directory.route(offer.key,WorkBudget(c),explore=False)[0]
        binding_counters=dict(row.binding_counters)
        serial=binding_counters.get(shard,0)
        old=target
        if target is None:
            slot=serial%c.bindings_per_shard
            # Bounded replacement comparison. Unrelated pages are never scanned.
            existing=row.bindings.get(shard,{})
            if c.retention and slot in existing and candidates:
                pool=[x for x in candidates if x.handle.shard==shard]
                if pool:
                    old=min(pool,key=lambda x:float(x.utility.detach())-.001*(row.round-x.created))
                    slot=old.handle.slot
            old=existing.get(slot)
            binding_counters[shard]=serial+1
            generation=1 if old is None else old.handle.generation+1
            bh=BindingHandle(row.namespace,shard,slot,generation,1)
            key,code=offer.key,offer.code
            links=()
            created=offer.ready_round
        else:
            bh=replace(target.handle,version=target.handle.version+1)
            erase=offer.erase if action==1 else offer.erase*0
            key=F.normalize((1-erase)*target.key+gate*offer.key,dim=-1)
            code,_=self.separate(key)
            links=tuple(l for l in target.links if l.handle.node!=producer)
            created=target.created
            # Revision uses a new capsule and handle; old in-flight snapshots stay
            # valid. No transient read gate multiplies persistent stored content.
            prior=next((row.anchor(l.handle,row.round) for l in target.links
                        if l.handle.node==producer and row.anchor(l.handle,row.round) is not None),None)
            if prior is not None:
                anchor=replace(anchor,capsule=(1-erase)*prior.capsule+gate*offer.anchor.capsule)
                local=dict(row.anchors[region]); local[handle.slot]=anchor
                anchors=dict(row.anchors); anchors[region]=local; row=replace(row,anchors=anchors)
        # Learned coactivity association only between compatible, current-round
        # offers. This bounded gate does not equate simultaneity with truth.
        if target is None and action==0:
            for record in candidates:
                if record.revised==offer.ready_round and float((record.key*offer.key).detach().sum())>.5:
                    links=tuple(l for l in record.links if l.handle.region!=region)[:c.links_per_binding-1]
                    break
        strength=1.+(offer.actions[action]-offer.actions[action].detach())
        link=AnchorLink(handle,offer.key,anchor.role,strength)
        links=(*links,link)[-c.links_per_binding:]
        neighbors=tuple(x.handle for x in candidates if x.handle!=bh and x.handle!=getattr(old,'handle',None))[:c.neighbors]
        record=BindingRecord(bh,key,code,offer.context,links,neighbors,
            offer.key.new_zeros(()),offer.write,created,offer.ready_round,c.representation_version)
        shard_records=dict(row.bindings.get(shard,{})); shard_records[bh.slot]=record
        bindings=dict(row.bindings); bindings[shard]=shard_records
        postings,recent=self.directory.install(row,record,old)
        return replace(row,bindings=bindings,binding_counters=binding_counters,postings=postings,recent=recent)

    def make_offers(self,state,keys,latents,nodes,active,versions,grounded):
        """One learned producer per region/row; selected owner identity is transport.

        Source activity is externally grounded. Computation with no current
        ingress is marked reinstated and cannot automatically write itself back.
        """
        b,r,d=keys.shape
        result=[[] for _ in state.rows]
        for ri,region in enumerate(self.regions):
            owners=self.owners[region]
            values=torch.stack([latents[n] for n in owners],1)
            trace=state.cues[:,ri,None].expand_as(values)
            cue=keys[:,ri,None].expand(-1,len(owners),-1)
            key,role,write,erase=self.interface.offer(trace,values,cue)
            eligible=torch.stack([active[n] for n in owners],1)
            pick=write.squeeze(-1).masked_fill(~eligible,-1).argmax(-1)
            idx=torch.arange(b,device=values.device)
            selected=values[idx,pick]
            capsule=selected.new_zeros(selected.shape)
            for oi,name in enumerate(owners):
                rows=pick.eq(oi).nonzero().flatten()
                if rows.numel():
                    capsule=capsule.index_copy(0,rows,nodes[name].encode_anchor(selected.index_select(0,rows)))
            k=key[idx,pick]; roles=role[idx,pick]; codes,_=self.separate(k)
            actions=self.interface.actions(torch.cat((k,keys[:,ri],roles),-1)).softmax(-1)
            for bi,row in enumerate(state.rows):
                oi=int(pick[bi]); owner=owners[oi]
                if not bool(eligible[bi,oi]) or not bool(grounded[bi,ri]):
                    continue
                handle=EngramHandle(row.namespace,region,owner,-1,0,0,self.config.representation_version)
                anchor=LocalAnchor(handle,capsule[bi],k[bi],roles[bi],'derived',row.round+1,int(versions[owner][bi]))
                result[bi].append(BindingOffer((row.namespace,row.round,owner),row.namespace,owner,
                    anchor,k[bi],codes[bi],keys[bi,ri],write[bi,oi,0],erase[bi,oi,0],actions[bi],row.round+1,'derived'))
        return tuple(tuple(x) for x in result)
