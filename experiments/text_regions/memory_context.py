"""Bounded first-class evidence, shared by graph edges and episodic recall.

The query has a reserved lane. Slots keep raw content and causal identities;
selection/confidence regulate consumption, never decay the stored content.
All state is explicit so forks, resets and recomputation have the same meaning.
"""
from dataclasses import dataclass, replace
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MemoryContextConfig:
    enabled: bool = False
    slots: int = 8
    block_depths: tuple[int, ...] = (0, 2)
    requests_per_round: int = 4
    refresh_interval: int = 8
    request_change: float = .08
    decouple_working_state: bool = True
    protect_forward: bool = True
    event_links: bool = True

    def validate(self):
        if not 4 <= self.slots <= 8:
            raise ValueError('Working context requires four to eight slots.')
        if (len(set(self.block_depths)) != len(self.block_depths)
                or any(i not in (0, 1, 2) for i in self.block_depths)):
            raise ValueError('Memory access depths must be distinct sparse blocks.')
        if self.requests_per_round < 1 or self.refresh_interval < 1 or not 0 <= self.request_change <= 2:
            raise ValueError('Invalid memory request budget or refresh policy.')


@dataclass(frozen=True)
class MemoryContext:
    query: torch.Tensor             # [B,W], never competes for a slot
    values: torch.Tensor            # [B,S,W], raw capsule or publication
    keys: torch.Tensor
    roles: torch.Tensor
    valid: torch.Tensor             # [B,S]
    identities: torch.Tensor        # kind,node,namespace,slot,generation,version,semantic,event
    strength: torch.Tensor          # read control, separate from content
    cue: torch.Tensor               # current request, separate from cached request
    request: torch.Tensor           # last directory request, not current question
    fetched: torch.Tensor           # last request round, -1 before first request

    def select_rows(self, rows):
        from .ports import map_tensors
        return map_tensors(self, lambda x: x.index_select(0, rows))

    def reset(self, rows):
        from .ports import map_tensors, choose_rows
        result = map_tensors(self, lambda x: choose_rows(rows, torch.zeros_like(x), x))
        return replace(result, fetched=torch.where(rows, -1, self.fetched))


class SlotReader(nn.Module):
    """One small query-conditioned reader, reused at every access point."""
    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(2*width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.null = nn.Linear(2*width, 1)
        self.age = nn.Parameter(torch.tensor(-.02))
        nn.init.eye_(self.value.weight)
        nn.init.zeros_(self.null.weight)
        nn.init.zeros_(self.null.bias)

    def forward(self, working, context, event, capture=None):
        query = torch.cat((context.query, working), -1)
        q = self.query(query)
        keys = self.key(context.keys + context.roles)
        score = (q[:, None]*keys).sum(-1)/math.sqrt(q.shape[-1])
        age = (event[:, None]-context.identities[..., 7]).clamp_min(0).to(score.dtype).log1p()
        score = (score+self.age*age).masked_fill(~context.valid, -torch.inf)
        weights = torch.cat((score, self.null(query)), -1).softmax(-1)[:, :-1]
        weights = weights*context.strength*context.valid
        if capture is not None:
            capture['weights'] = weights
        return (self.value(context.values)*weights[..., None]).sum(1)


class MemoryContextInterface(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.options = cfg, cfg.memory_context
        self.names = tuple(n.name for n in cfg.nodes)
        self.indices = {n: i for i, n in enumerate(self.names)}
        self.blocked_sources = ()  # Evaluation intervention, never an inference label.
        self.regions = {n.name: n.region for n in cfg.nodes}
        w, d = cfg.width, cfg.hippocampus.address_width
        self.node_roles = nn.Embedding(len(self.names), w)
        nn.init.normal_(self.node_roles.weight, std=.02)
        self.request = nn.Linear(3*w, d, bias=False)
        self.question = nn.Linear(3*w, w, bias=False)
        self.key = nn.Linear(d, w, bias=False)
        self.role = nn.Linear(d, w, bias=False)
        self.hold = nn.Linear(2*w, 1)
        self.refresh = nn.Linear(2*w, 1)
        nn.init.zeros_(self.refresh.weight)
        nn.init.constant_(self.refresh.bias, -2.)
        self.reader = SlotReader(w)
        self.block_gain = nn.Parameter(torch.full((3,), .1))
        out = w if cfg.readout_mode == 'mixed' else len(cfg.readout)*w
        self.composer = nn.Linear(w, out, bias=False)
        self.composer_query = nn.Linear((len(cfg.readout)+1)*w,w,bias=False)
        nn.init.normal_(self.composer.weight, std=.01)

    def empty(self, batch, device, dtype):
        s, w, d = self.options.slots, self.cfg.width, self.cfg.hippocampus.address_width
        z = lambda *shape: torch.zeros(*shape, device=device, dtype=dtype)
        return MemoryContext(z(batch,w), z(batch,s,w), z(batch,s,w), z(batch,s,w),
            torch.zeros(batch,s,device=device,dtype=torch.bool),
            torch.zeros(batch,s,8,device=device,dtype=torch.long), z(batch,s), z(batch,d), z(batch,d),
            torch.full((batch,),-1,device=device,dtype=torch.long))

    def valid_slots(self, context, state, available):
        """Validate raw handles against the current causal cut, not slot numbers."""
        from .hippocampus_interface import EngramHandle
        mask = context.valid.clone()
        for bi in range(len(mask)):
            for si in mask[bi].nonzero().flatten().tolist():
                kind, ni, epoch, slot, generation, version, semantic, event = context.identities[bi,si].tolist()
                name = self.names[ni]
                allowed = bool(available[name][bi])
                if kind == 1:
                    pub = state.ports.latest[name]
                    ok = bool(pub.valid[bi]) and int(pub.epoch[bi]) == epoch and int(pub.version[bi]) == version
                else:
                    row = state.hippocampus.rows[bi]
                    handle = EngramHandle(epoch,self.regions[name],name,slot,generation,version,semantic)
                    ok = row.anchor(handle,row.round) is not None and semantic == self.cfg.hippocampus.representation_version
                mask[bi,si] = allowed and ok
        return replace(context, valid=mask)

    def begin_round(self, model, state, source, available):
        """Fair bounded requests using fresh ingress plus committed working state."""
        b = len(state.events_seen)
        contexts, keys, questions, needs, probabilities = {}, [], [], [], []
        for name in self.names:
            ri = model.hippocampus.regions.index(self.regions[name])
            previous = state.nodes[name].working
            if previous is None:
                previous = torch.zeros_like(source[:,ri])
            pieces = torch.cat((source[:,ri],state.hippocampus.cues[:,ri],
                                previous+self.node_roles.weight[self.indices[name]]),-1)
            key, question = F.normalize(self.request(pieces),dim=-1), self.question(pieces)
            old = state.memory_context[name] if name in state.memory_context else self.empty(b,source.device,source.dtype)
            old = self.valid_slots(old,state,available)
            probability = self.refresh(torch.cat((question,previous),-1)).sigmoid().squeeze(-1)
            changed = (1-F.cosine_similarity(key,old.request,dim=-1)) > self.options.request_change
            expired = state.rounds_seen-old.fetched >= self.options.refresh_interval
            invalidated = (state.memory_context.get(name,old).valid & ~old.valid & old.identities[...,0].eq(2)).any(-1)
            need = (old.fetched.lt(0) | changed | expired | invalidated | probability.ge(.5)) & available[name]
            contexts[name] = replace(old,query=torch.where(available[name][:,None],question,old.query),
                                     cue=torch.where(available[name][:,None],key,old.cue))
            keys.append(key); questions.append(question); needs.append(need); probabilities.append(probability)
        need = torch.stack(needs,1)
        admitted = torch.zeros_like(need)
        limit = min(self.options.requests_per_round,self.cfg.hippocampus.service_budget)
        for bi in range(b):
            start = int(state.rounds_seen[bi])*limit % len(self.names)
            order = [(start+i)%len(self.names) for i in range(len(self.names))]
            chosen = [i for i in order if bool(need[bi,i])][:limit]
            admitted[bi,chosen] = True
        capture = {}
        _, receipts, work = model.hippocampus.query(state.hippocampus,torch.stack(keys,1),model.nodes,
            {},admitted,{n:state.nodes[n].working if state.nodes[n].working is not None else source.new_zeros(b,self.cfg.width)
                         for n in self.names},available,
            regions=tuple(self.regions[n] for n in self.names),requester_nodes=self.names,slot_capture=capture,
            event_times=state.events_seen)
        for ni,name in enumerate(self.names):
            old = contexts[name]
            candidates = [[] for _ in range(b)]
            for bi in range(b):
                if not bool(admitted[bi,ni]):
                    continue
                for entry in capture.get('entries', [[]]*(b*len(self.names)))[bi*len(self.names)+ni]:
                    anchor,link,weight,binding = entry
                    h = anchor.handle
                    ident = [2,self.indices[h.node],h.namespace,h.slot,h.generation,h.version,h.semantic_version,anchor.source_event]
                    # The forward value is exactly one: ordinary gradients train
                    # refresh/selection without repeatedly attenuating evidence.
                    p = probabilities[ni][bi]
                    strength = weight*link.strength*(1+p-p.detach())
                    candidates[bi].append((anchor.capsule,self.key(anchor.key),self.role(anchor.role),ident,strength,False))
            contexts[name] = self.merge(old,candidates,state.events_seen)
            contexts[name] = replace(contexts[name],request=torch.where(admitted[:,ni,None],keys[ni],old.request),
                fetched=torch.where(admitted[:,ni],state.rounds_seen,old.fetched))
        totals = {k:sum(w.get(k,0) for w in work) for k in work[0]} if work else {}
        totals.update(context_requests=int(admitted.sum()),context_reuses=int((~admitted & ~need).sum()),
                      context_deferred=int((need & ~admitted).sum()),
                      capsule_fetches=capture.get('capsule_fetches',0),shared_fetch_hits=capture.get('shared_fetch_hits',0))
        return replace(state,memory_context=contexts),receipts,[totals]

    def merge(self, old, candidates, events, keep_old=True):
        """Learned bounded hold/replace; identity deduplication precedes selection."""
        rows = []
        for bi,new in enumerate(candidates):
            entries = {}
            if keep_old:
                for si in old.valid[bi].nonzero().flatten().tolist():
                    ident = old.identities[bi,si].tolist()
                    entries[tuple(ident[:7])] = (old.values[bi,si],old.keys[bi,si],old.roles[bi,si],ident,old.strength[bi,si],False)
            for item in new:
                entries[tuple(item[3][:7])] = item
            items = list(entries.values())
            if items:
                values = torch.stack([x[0] for x in items])
                key = torch.stack([x[1] for x in items])
                q = old.query[bi].expand_as(values)
                score = (q*key).sum(-1)/math.sqrt(q.shape[-1])+self.hold(torch.cat((q,values),-1)).squeeze(-1)
                # Protect direct forward evidence from the older projected graph
                # bottleneck. Other slots compete through learned relevance.
                protected = [i for i,x in enumerate(items) if x[5]][:self.options.slots]
                ranked = score.detach().argsort(descending=True).tolist()
                picks = (protected+[i for i in ranked if i not in protected])[:self.options.slots]
                prob = score.softmax(-1)
                selected = [(items[i][0],items[i][1],items[i][2],items[i][3],
                             items[i][4]*(1+prob[i]-prob[i].detach())) for i in picks]
            else:
                selected = []
            zero = old.query[bi].new_zeros(self.cfg.width)
            count = len(selected)
            selected += [(zero,zero,zero,[0]*8,zero.new_zeros(()))]*(self.options.slots-count)
            rows.append((torch.stack([x[0] for x in selected]),torch.stack([x[1] for x in selected]),
                         torch.stack([x[2] for x in selected]),old.identities.new_tensor([x[3] for x in selected]),
                         torch.stack([x[4] for x in selected]),old.valid.new_tensor([True]*count+[False]*(self.options.slots-count))))
        return replace(old,values=torch.stack([r[0] for r in rows]),keys=torch.stack([r[1] for r in rows]),
            roles=torch.stack([r[2] for r in rows]),identities=torch.stack([r[3] for r in rows]),
            strength=torch.stack([r[4] for r in rows]),valid=torch.stack([r[5] for r in rows]))

    def gather(self, name, state, current, delayed, active, event_index, edge_masks, substitutions):
        old = state.memory_context[name]
        # Local publications are refreshed from the graph cut on every use;
        # delayed edges must never accidentally read a current-stage value.
        old = replace(old,valid=old.valid & old.identities[...,0].eq(2))
        candidates = [[] for _ in range(len(active))]
        for edge in self.cfg.edges:
            if edge.destination != name or (edge.kind != 'forward' and delayed is None):
                continue
            pub = (current if edge.kind == 'forward' else delayed)[edge.source]
            used = pub.valid & active
            if edge.key in edge_masks:
                used = used & edge_masks[edge.key][:,event_index]
            payload = substitutions[edge.key].payload[:,event_index] if edge.key in substitutions else pub.payload
            for bi in used.nonzero().flatten().tolist():
                ident = [1,self.indices[edge.source],int(pub.epoch[bi]),0,0,int(pub.version[bi]),0,int(pub.occurrence[bi])]
                protected = self.options.protect_forward and edge.kind == 'forward'
                candidates[bi].append((payload[bi],payload[bi],self.node_roles.weight[self.indices[edge.source]],
                                       ident,payload.new_ones(()),protected))
        return self.merge(old,candidates,state.events_seen)

    def interpret(self, context, state, model, active):
        """Current owner computation over cached raw capsules; no directory read."""
        valid = context.valid & active[:,None]
        if model.cfg.hippocampus.control == 'blocked' or not model.cfg.hippocampus.read:
            valid = valid & context.identities[...,0].ne(2)
        for source in self.blocked_sources:
            if source not in self.indices:
                raise ValueError('Unknown memory source intervention.')
            valid = valid & context.identities[...,1].ne(self.indices[source])
        context = replace(context,valid=valid)
        hip = context.valid & context.identities[...,0].eq(2)
        if not bool(hip.any()):
            return context
        key_bank = context.values.new_zeros(*context.valid.shape,self.cfg.hippocampus.address_width)
        dispatch = []
        for ni,name in enumerate(self.names):
            positions = (hip & context.identities[...,1].eq(ni)).nonzero()
            if not len(positions):
                continue
            bi,si = positions.unbind(-1)
            key = torch.stack([state.hippocampus.rows[int(b)].anchors[self.regions[name]][int(context.identities[b,s,3])].key
                               for b,s in positions])
            working = state.nodes[name].working
            if working is None:
                working = context.query.new_zeros(len(active),self.cfg.width)
            dispatch.append((name,bi,si,key,working[bi]))
            key_bank = key_bank.index_put((bi,si),key)
        def read(cues):
            values = context.values
            for name,bi,si,key,working in dispatch:
                interpreted = model.nodes[name].read_anchor(context.values[bi,si],cues[bi],working,stored_key=key)
                values = values.index_put((bi,si),interpreted)
            return values
        values = read(context.cue)
        service = model.hippocampus
        expected = F.normalize((key_bank*hip[...,None]).sum(1),dim=-1)
        returned = F.normalize(service.return_descriptor((values*hip[...,None]).sum(1)),dim=-1)
        age = (state.events_seen[:,None]-context.identities[...,7]).clamp_min(0).to(values.dtype).log1p()
        age = (age*hip).sum(1)/hip.sum(1).clamp_min(1)
        has_return = hip.any(-1).to(values.dtype)
        can_refine = bool(self.cfg.hippocampus.refinement_steps and self.cfg.hippocampus.control != 'no_refinement')
        decision = service.compare(context.cue,expected,returned,has_return,age,terminal=not can_refine)
        accepted = decision[:,0]
        if can_refine:
            refined = read(context.cue+(context.cue-returned))
            descriptor = F.normalize(service.return_descriptor((refined*hip[...,None]).sum(1)),dim=-1)
            terminal = service.compare(context.cue,expected,descriptor,has_return,age,terminal=True)
            refine_accept = decision[:,2]*terminal[:,0]
            accepted = accepted+refine_accept
            # Conditional content and its probability are separate. Only the
            # temporary consumption weight sees accept/null; raw slots remain
            # untouched. A reread that CA1 accepted is the reread delivered.
            mixture = (decision[:,0,None,None]*values+refine_accept[:,None,None]*refined)/accepted[:,None,None].clamp_min(1e-8)
            values = torch.where(hip[...,None],mixture,values)
        strength = context.strength*torch.where(hip,accepted[:,None],torch.ones_like(context.strength))
        return replace(context,values=values,strength=strength)

    def compose(self, latent, state, model):
        names = self.cfg.readout
        query = torch.stack([state.memory_context[n].query for n in names]).mean(0)
        base = self.empty(len(query),query.device,query.dtype)
        base = replace(base,query=query,cue=torch.stack([state.memory_context[n].cue for n in names]).mean(0))
        candidates = [[] for _ in range(len(query))]
        for name in names:
            ctx = state.memory_context[name]
            for bi in range(len(query)):
                for si in ctx.valid[bi].nonzero().flatten().tolist():
                    candidates[bi].append((ctx.values[bi,si],ctx.keys[bi,si],ctx.roles[bi,si],
                        ctx.identities[bi,si].tolist(),ctx.strength[bi,si],False))
        context = self.merge(base,candidates,state.events_seen,keep_old=False)
        available = {n:torch.ones_like(state.events_seen,dtype=torch.bool) for n in self.names}
        context = self.valid_slots(context,state,available)
        context = self.interpret(context,state,model,available[self.names[0]])
        working = self.composer_query(torch.cat((query,*(state.ports.latest[n].payload for n in names)),-1)).tanh()
        return latent+self.composer(self.reader(working,context,state.events_seen))

    def parents(self, context, row_index):
        from .hippocampus_interface import EngramHandle
        result = []
        for si in (context.valid[row_index] & context.identities[row_index,:,0].eq(2)).nonzero().flatten().tolist():
            _,ni,epoch,slot,generation,version,semantic,event = context.identities[row_index,si].tolist()
            name = self.names[ni]
            result.append(EngramHandle(epoch,self.regions[name],name,slot,generation,version,semantic))
        return tuple(dict.fromkeys(result))

    def sources(self, context, row_index):
        result = []
        for si in (context.valid[row_index] & context.identities[row_index,:,0].eq(1)).nonzero().flatten().tolist():
            _,ni,epoch,slot,generation,version,semantic,event = context.identities[row_index,si].tolist()
            result.append((self.names[ni],epoch,version,event))
        return tuple(dict.fromkeys(result))
