"""HAT -> parent SHoE -> two recurrent SHoEs, with half-shared expert banks."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from events import ByteBatch
from hat import HAT, HATState
from kron import MinKron
from semantic_core import Config, ExpertBank, SemanticRouter, LaceController, normalize, switch_balance
from hierarchy_reference import HierarchyConfig, ParentSHoE, SplitExpertBank


@dataclass
class HATConfig(HierarchyConfig):
    vocab_size: int = 256
    width: int = 100
    experts: int = 1024
    expert_hidden: int = 4
    groups: int = 32
    active_groups: int = 2
    experts_per_group: int = 8
    address_dim: int = 16
    laces: int = 64
    active_laces: int = 2
    max_links: int = 8
    max_slots: int = 256
    hat_levels: int = 2
    hat_symbols: int = 64
    hat_active_symbols: int = 2
    kron_heads: int = 4
    kron_key: int = 8
    kron_value: int = 8
    chunk_size: int = 32
    hat_balance_weight: float = .001

    def child_config(self,index=0):
        return Config(vocab_size=256,context=self.context,width=self.width,experts=self.experts,
                      expert_hidden=self.expert_hidden,groups=self.groups,active_groups=self.active_groups,
                      experts_per_group=self.experts_per_group,address_dim=self.address_dim,
                      laces=self.laces,active_laces=self.active_laces,seed=self.seed+1009*index)

    def validate(self):
        self.child_config().validate()
        if self.vocab_size!=256: raise ValueError('HAT uses all 256 byte values.')
        if self.max_links<2 or self.max_slots<256: raise ValueError('Insufficient address capacity.')
        if not 1<=self.hat_active_symbols<=self.hat_symbols or self.hat_levels<1:
            raise ValueError('Invalid symbol hierarchy.')


@dataclass
class StreamState:
    hat: HATState
    children: tuple
    events_seen: torch.Tensor

    def detach(self):
        return StreamState(self.hat.detach(),tuple(tuple(v.detach() for v in child)
                           for child in self.children),self.events_seen.detach())

    def reset(self,rows):
        rows=rows.to(device=self.events_seen.device,dtype=torch.bool)
        if rows.shape!=self.events_seen.shape: raise ValueError('Reset mask must have shape [B].')
        return StreamState(self.hat.reset(rows),tuple(tuple(v*(~rows)[:,None,None,None] for v in child)
                           for child in self.children),self.events_seen*(~rows))

    def tensors(self):
        return self.hat.tensors()+[v for child in self.children for v in child]+[self.events_seen]

    def storage_bytes(self):
        return sum(v.numel()*v.element_size() for v in self.tensors())

    def to_dict(self):
        return {'hat':{'batch_size':self.hat.batch_size,'levels':self.hat.levels,'clocks':self.hat.clocks},
                'children':self.children,'events_seen':self.events_seen}

    @classmethod
    def from_dict(cls,data):
        return cls(HATState(**data['hat']),tuple(tuple(x) for x in data['children']),data['events_seen'])


class StreamingRouter(SemanticRouter):
    """SHoE routing driven by recurrent context; no chunk-local prefix average."""
    def __init__(self,c,layer,count):
        super().__init__(c,layer,count)
        del self.key  # Historical-key indexing disappears with sparse attention.

    def forward(self,x,prior):
        c=self.c;prototypes=normalize(self.prototypes)
        query=normalize(self.query(x)+c.lace_gain*self.message(prior@prototypes))
        scores=4*F.linear(query,prototypes)
        message=prior.clamp_min(1e-8).log()
        scores=scores+c.lace_gain*(message-message.mean(-1,keepdim=True))
        prob=scores.softmax(-1);groups=scores.topk(c.active_groups,-1).indices
        candidates=self.membership[groups];valid=candidates>=0
        keys=normalize(prototypes[groups][...,None,:]+.35*torch.tanh(self.expert_residual[candidates.clamp_min(0)]))
        member_scores=(query[...,None,None,:]*keys).sum(-1)*8
        member_scores=member_scores.masked_fill(~valid,-torch.inf)
        member_prob=member_scores.softmax(-1)
        picked=member_scores.topk(c.experts_per_group,-1).indices
        ids=candidates.gather(-1,picked)
        eg=member_prob.gather(-1,picked)*valid.sum(-1,keepdim=True)/c.experts_per_group
        gg=prob.gather(-1,groups)*c.groups/c.active_groups
        return {'groups':groups,'group_scores':scores,'group_prob':prob,
                'experts':ids.flatten(-2),'gates':(eg*gg[...,None]).flatten(-2),
                'aux':switch_balance(prob,groups)}


class RecurrentBlock(nn.Module):
    def __init__(self,c,cfg,layer,count,shared):
        super().__init__()
        self.norm1,self.norm2=nn.LayerNorm(c.width),nn.LayerNorm(c.width)
        self.memory=MinKron(c.width,cfg.kron_heads,cfg.kron_key,cfg.kron_value,cfg.chunk_size)
        self.router=StreamingRouter(c,layer,count)
        self.experts=SplitExpertBank(c,shared,count-shared.up.shape[0])

    def forward(self,x,prior,state,backend='chunk',memory_enabled=True):
        memory,final=self.memory(self.norm1(x),state,backend,not memory_enabled)
        x=x+memory
        z=self.norm2(x);route=self.router(z,prior)
        x=x+self.experts(z,route['experts'],route['gates'])
        return x,final,route


class RecurrentChild(nn.Module):
    def __init__(self,c,cfg,shared):
        super().__init__();self.cfg=c
        self.blocks=nn.ModuleList([RecurrentBlock(c,cfg,l,n,shared[l]) for l,n in enumerate(c.expert_counts)])
        self.lace=LaceController(c)
        self.final_norm=nn.LayerNorm(c.width)
        self.decoder=nn.Linear(c.width,256,bias=False)
        nn.init.normal_(self.decoder.weight,std=.02)

    def probe(self,context):
        evidence=torch.stack([b.router.pilot(context) for b in self.blocks],-2)
        prior,lace=self.lace(evidence,context)
        return {'evidence':evidence,'prior':prior,'match':(evidence/4).amax(-1).mean(-1),
                'lace_ids':lace['lace_ids']}

    def execute(self,payload,prior,states,backend='chunk',memory_enabled=True,trace=False):
        x=payload;aux=x.new_zeros(());actual=[];final=[];details=[]
        for layer,block in enumerate(self.blocks):
            x,memory,route=block(x,prior[:,:,layer],states[layer],backend,memory_enabled)
            final.append(memory);aux=aux+route['aux'];actual.append(route['group_scores'])
            if trace: details.append({k:v.detach() for k,v in route.items()})
        logits=self.decoder(self.final_norm(x))
        with torch.no_grad():
            observed,lace=self.lace(torch.stack(actual,-2).detach(),F.layer_norm(payload.detach(),(self.cfg.width,)))
        return {'logits':logits,'state':tuple(final),'aux':aux/len(self.blocks),
                'feedback_prior':observed,'feedback_lace_ids':lace['lace_ids'],'trace':details}


class HATSHoE(nn.Module):
    def __init__(self,cfg=None):
        super().__init__();self.cfg=cfg or HATConfig();self.cfg.validate()
        c=self.cfg.child_config()
        self.hat=HAT(self.cfg)
        shared=[n//2 for n in c.expert_counts]
        for i,n in enumerate(c.expert_counts):
            if sum(shared)==self.cfg.experts//2: break
            if n%2: shared[i]+=1
        self.shared_counts=tuple(shared)
        self.shared_banks=nn.ModuleList([ExpertBank(c,n) for n in shared])
        self.children_blocks=nn.ModuleList([RecurrentChild(self.cfg.child_config(i),self.cfg,self.shared_banks) for i in range(2)])
        self.parent=ParentSHoE(c)

    def initial_state(self,batch,device=None):
        device=device or self.hat.byte_embedding.weight.device
        dtype=self.hat.byte_embedding.weight.dtype
        children=tuple(tuple(b.memory.initial_state(batch,device,dtype) for b in child.blocks)
                       for child in self.children_blocks)
        return StreamState(self.hat.initial_state(batch),children,torch.zeros(batch,dtype=torch.long,device=device))

    def route_only(self,payload,positions=None,reinforce=True,force_discovery=False,
                   disable_upward=False,forced_child=None,scramble_upward=False):
        """No child attention or expert forward is called anywhere in this method."""
        cfg,c = self.cfg,self.cfg.child_config()
        b,t,d = payload.shape
        if positions is None: positions=torch.arange(t,device=payload.device)[None].expand(b,-1)
        context = F.layer_norm(payload,(d,))
        evidence,prior,initial_lace = self.parent.initial(context)
        initial_scores,_,value,mass = self.parent.scores(prior,context,use_memory=reinforce)
        initial_prob = initial_scores.softmax(-1)
        confidence,preferred = initial_prob.max(-1)
        selected_value = value.gather(-1,preferred[...,None]).squeeze(-1)
        selected_mass = mass.gather(-1,preferred[...,None]).squeeze(-1)
        fast = ((confidence >= cfg.fast_threshold) & (selected_value > 0)
                & (selected_mass >= cfg.minimum_route_mass)
                & (self.parent.feedback_updates >= cfg.fast_warmup_updates))
        explore=torch.zeros_like(fast)
        if self.training:
            explore=torch.rand(b,t,device=payload.device)<cfg.exploration
            fast = fast & ~explore
        if force_discovery or not reinforce: fast=torch.zeros_like(fast)
        probed = (~fast)[...,None].expand(-1,-1,2).clone()
        probed.scatter_(-1,preferred[...,None],True)
        local_priors = payload.new_zeros(b,t,2,c.layers,c.groups)
        local_evidence = torch.zeros_like(local_priors)
        matches = payload.new_full((b,t,2),-1.)
        child_contexts=[]
        for child in self.children_blocks:
            z=payload
            child_contexts.append(F.layer_norm(z,(d,)))

        def probe(which,mask):
            indices=mask.reshape(-1).nonzero().flatten()
            if indices.numel()==0: return
            z=child_contexts[which].reshape(-1,d)[indices][None]
            answer=self.children_blocks[which].probe(z)
            local_priors[:,:,which].reshape(-1,c.layers,c.groups)[indices]=answer['prior'][0]
            local_evidence[:,:,which].reshape(-1,c.layers,c.groups)[indices]=answer['evidence'][0]
            matches[:,:,which].reshape(-1)[indices]=answer['match'][0]
        for which in range(2): probe(which,probed[:,:,which])
        rejected = fast & (matches.gather(-1,preferred[...,None]).squeeze(-1)<cfg.match_threshold)
        for which in range(2):
            extra=rejected & (~probed[:,:,which])
            probe(which,extra)
            probed[:,:,which]=probed[:,:,which]|extra
        fast=fast & ~rejected
        observed=local_priors.flip(-1) if scramble_upward else local_priors
        relation=self.parent.relation_logits.softmax(-1)
        support=torch.einsum('clag,btclg->btcla',relation,observed).clamp_min(1e-8)
        message=support.log()
        message=message-message.mean(-1,keepdim=True)
        sender_weight=initial_prob*probed.to(payload.dtype)
        sender_weight=sender_weight/sender_weight.sum(-1,keepdim=True).clamp_min(1e-8)
        upward=(message*sender_weight[...,None,None]).sum(2)
        final_prior,final_lace=self.parent.lace(
            evidence+(0 if disable_upward else cfg.upward_gain)*upward,context)
        expected=torch.einsum('btla,clag->btclg',final_prior,relation)
        compatibility=(expected*observed).sum(-1).clamp_min(1e-8).log().mean(-1)+math.log(c.groups)
        scores,gains,_,_=self.parent.scores(final_prior,context,use_memory=reinforce)
        if not disable_upward:
            scores=scores+torch.where(probed,compatibility+.25*matches,torch.zeros_like(scores))
        learning_prob=scores.softmax(-1)
        prob=scores.masked_fill(~probed,-torch.inf).softmax(-1)
        choice=prob.argmax(-1)
        if self.training:
            random_child=torch.randint(2,(b,t),device=payload.device)
            choice=torch.where(explore,random_child,choice)
        if forced_child is not None:
            choice=torch.full_like(choice,int(forced_child))
            # Tests/interventions force discovery first so this child was probed.
            if not bool(probed.gather(-1,choice[...,None]).all()):
                raise ValueError('Forced child must be probed; use force_discovery=True.')
        return {'context':context,'parent_prior':final_prior,'parent_initial_prior':prior,
                'parent_lace_ids':final_lace['lace_ids'],'parent_lace_balance':final_lace['balance'],
                'expected':expected,'local_priors':local_priors,'local_evidence':local_evidence,
                'child_contexts':child_contexts,'prob':prob,'learning_prob':learning_prob,'choice':choice,'gains':gains,
                'probed':probed,'fast':fast,'matches':matches,'upward':upward,'positions':positions}

    def forward(self,events,targets=None,state=None,reset=None,backend='chunk',hat_memory=True,
                expert_memory=True,reinforce=True,trace=False,**routing):
        if torch.is_tensor(events): events=ByteBatch.static(events)
        events.validate(self.cfg.max_links,self.cfg.max_slots)
        b,t=events.values.shape
        state=state or self.initial_state(b,events.values.device)
        if state.events_seen.shape!=(b,) or state.events_seen.device!=events.values.device:
            raise ValueError('Stream state batch/device mismatch.')
        if reset is not None: state=state.reset(reset)
        h=self.hat(events,state.hat,backend,hat_memory,trace)
        positions=state.events_seen[:,None]+torch.arange(t,device=events.values.device)
        route=self.route_only(h['payload'],positions,reinforce=reinforce,**routing)
        c=self.cfg.child_config();payload=h['payload']
        logits=payload.new_zeros(b*t,256)
        observed=payload.new_zeros(b*t,3,c.groups)
        executed=torch.zeros(b*t,3,c.active_laces,dtype=torch.long,device=payload.device)
        aux=payload.new_zeros(());executions=[]
        child_states=[list(x) for x in state.children]
        for which,child in enumerate(self.children_blocks):
            mask=route['choice']==which
            if bool((mask==mask[:1]).all()):
                groups=[(torch.arange(b,device=payload.device),mask[0].nonzero().flatten())]
            else:
                groups=[(torch.tensor([row],device=payload.device),mask[row].nonzero().flatten()) for row in range(b)]
            for rows,index in groups:
                if index.numel()==0: continue
                ri,ti=rows[:,None],index[None,:]
                ctx=route['child_contexts'][which][ri,ti]
                expected=route['expected'][ri,ti,which]
                down=expected.clamp_min(1e-8).log();down=down-down.mean(-1,keepdim=True)
                prior,_=child.lace(route['local_evidence'][ri,ti,which]+self.cfg.downward_gain*down,ctx)
                previous=tuple(v[rows] for v in state.children[which])
                result=child.execute(payload[ri,ti],prior,previous,backend,expert_memory,trace)
                for layer,end in enumerate(result['state']):
                    child_states[which][layer]=child_states[which][layer].index_copy(0,rows,end)
                flat=(ri*t+ti).flatten()
                logits=logits.index_copy(0,flat,result['logits'].reshape(-1,256))
                observed=observed.index_copy(0,flat,result['feedback_prior'].reshape(-1,3,c.groups))
                executed=executed.index_copy(0,flat,result['feedback_lace_ids'].reshape(-1,3,c.active_laces))
                aux=aux+result['aux']*flat.numel()/(b*t)
                if trace:
                    for n,row in enumerate(rows.tolist()):
                        executions.append({'child':which,'batch_row':row,'source_positions':positions[row,index],
                                           'trace':[{k:v[n:n+1] if v.ndim else v for k,v in tr.items()} for tr in result['trace']]})
        logits=logits.reshape(b,t,256)
        picked=route['learning_prob'].gather(-1,route['choice'][...,None])
        logits=logits*(1+picked-picked.detach())
        aux=aux+switch_balance(route['learning_prob'],route['choice'])+.1*route['parent_lace_balance']
        mean=route['learning_prob'].mean((0,1))
        parent_balance=-mean.clamp_min(1e-8).log().mean()-math.log(2)
        ce=semantic=utility=baseline_loss=loss=None;packet=None
        if targets is not None:
            if targets.shape!=(b,t): raise ValueError('Targets must have shape [B,T].')
            mask=targets!=-100
            if not bool(mask.any()): raise ValueError('At least one supervised target is needed.')
            losses=F.cross_entropy(logits.reshape(-1,256),targets.reshape(-1),reduction='none').reshape(b,t)
            base=F.linear(route['context'],self.hat.byte_embedding.weight)
            baseline=F.cross_entropy(base.reshape(-1,256),targets.reshape(-1),reduction='none').reshape(b,t)
            ce=losses[mask].mean();baseline_loss=baseline[mask].mean()
            gain=(baseline.detach()-losses.detach()).clamp(-2,2)
            expected=route['expected'].gather(2,route['choice'][...,None,None,None].expand(-1,-1,1,3,c.groups)).squeeze(2)
            target=observed.reshape(b,t,3,c.groups).detach()
            kl=(target*(target.clamp_min(1e-8).log()-expected.clamp_min(1e-8).log())).sum(-1).mean(-1)
            semantic=((.25+.75*gain.sigmoid())*kl)[mask].mean()
            predicted=route['gains'].gather(-1,route['choice'][...,None]).squeeze(-1)
            utility=F.smooth_l1_loss(predicted[mask],gain[mask])
            packet={'eligibility':route['parent_prior'][mask].detach(),
                    'choice':route['choice'][mask].detach(),'gain':gain[mask].detach()}
            loss=ce+self.cfg.auxiliary_loss_weight*aux+self.cfg.baseline_loss_weight*baseline_loss
            loss=loss+self.cfg.parent_balance_weight*parent_balance+self.cfg.hat_balance_weight*h['aux']
            if reinforce: loss=loss+self.cfg.semantic_loss_weight*semantic+self.cfg.utility_loss_weight*utility
        final=StreamState(h['state'],tuple(tuple(v) for v in child_states),state.events_seen+t)
        return {'logits':logits,'ce':ce,'loss':loss,'aux':aux,'semantic_loss':semantic,
                'utility_loss':utility,'baseline_loss':baseline_loss,'packet':packet,
                'route':route,'executions':executions,'hat':h,'state':final,
                'executed_lace_ids':executed.reshape(b,t,3,c.active_laces)}

    @torch.no_grad()
    def commit_feedback(self,packet):
        if not self.training: raise RuntimeError('Feedback updates are training-only.')
        if packet is None: return
        e=packet['eligibility'].reshape(-1,3,self.cfg.groups)
        selected=F.one_hot(packet['choice'].reshape(-1),2).to(e.dtype)
        gain=packet['gain'].reshape(-1)
        mass=torch.einsum('nlg,nc->lgc',e,selected)/gain.numel()
        value=torch.einsum('nlg,nc,n->lgc',e,selected,gain)/gain.numel()
        decay=self.cfg.memory_decay
        self.parent.route_mass.mul_(decay).add_(mass,alpha=1-decay)
        self.parent.route_value_sum.mul_(decay).add_(value,alpha=1-decay)
        self.parent.feedback_updates.add_(1)

    def parameter_summary(self):
        shared=sum(p.numel() for p in self.shared_banks.parameters())
        private=[sum(p.numel() for block in child.blocks for p in block.experts.private.parameters())
                 for child in self.children_blocks]
        memory_values=self.cfg.kron_heads*self.cfg.kron_key*self.cfg.kron_value
        return {'total_unique_parameters':sum(p.numel() for p in self.parameters()),
                'hat_parameters':sum(p.numel() for p in self.hat.parameters()),
                'shared_experts':sum(self.shared_counts),'dedicated_experts_per_child':self.cfg.experts-sum(self.shared_counts),
                'unique_expert_count':2*self.cfg.experts-sum(self.shared_counts),
                'shared_expert_weights':shared,'private_expert_weights_per_child':private,
                'unique_expert_weights':shared+sum(private),'shared_counts_per_layer':list(self.shared_counts),
                'active_experts_per_event':3*self.cfg.active_groups*self.cfg.experts_per_group,
                'expert_recurrent_state_bytes_per_sequence':2*3*memory_values*4,
                'hat_recurrent_state_bytes_per_active_link':self.cfg.hat_levels*memory_values*4,
                'byte_alphabet':256,'kv_cache_entries':0,'context': 'fixed recurrent state; context config is training window only'}

    @torch.no_grad()
    def generate(self,events,count=200,temperature=.8,seed=123,**kwargs):
        if torch.is_tensor(events): events=ByteBatch.static(events)
        if not bool((events.links==0).all()): raise ValueError('Text generation uses the static byte bank.')
        rng=torch.Generator(device=events.values.device).manual_seed(seed)
        result=self(events,**kwargs);parts=[events.values]
        for i in range(count):
            sample=torch.multinomial((result['logits'][:,-1]/temperature).softmax(-1),1,generator=rng)
            parts.append(sample)
            result=self(ByteBatch.static(sample),state=result['state'],**kwargs)
        return torch.cat(parts,1),result['state']


if __name__=='__main__':
    import json
    torch.set_num_threads(2)
    m=HATSHoE();print(json.dumps(m.parameter_summary(),indent=2))
