"""One semantic-only SHoE routes to two original-shape 1M semantic SHoEs.

The children share exactly 512 expert MLPs by object identity. Tokens are
dispatched into causal child subsequences; only their selected child executes.
Cheap child probes precede dispatch. Executed child laces and next-character
outcomes train parent semantics, bid calibration, and next-pass route memory.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
if __package__:
    from .semantic_core import Config, SHoEGPT, ExpertBank, LaceController, normalize, prefix_mean, switch_balance
else:
    from semantic_core import Config, SHoEGPT, ExpertBank, LaceController, normalize, prefix_mean, switch_balance


@dataclass
class HierarchyConfig:
    vocab_size: int = 65
    context: int = 128
    seed: int = 20261103
    fast_threshold: float = .75
    match_threshold: float = .25
    minimum_route_mass: float = .002
    exploration: float = .10
    fast_warmup_updates: int = 80
    upward_gain: float = .5
    downward_gain: float = .35
    memory_decay: float = .97
    semantic_loss_weight: float = .03
    utility_loss_weight: float = .03
    baseline_loss_weight: float = .05
    auxiliary_loss_weight: float = .01
    parent_balance_weight: float = .05

    def child_config(self, index=0):
        return Config(vocab_size=self.vocab_size, context=self.context, seed=self.seed+1009*index)


class SplitExpertBank(nn.Module):
    """Local IDs [shared prefix, private suffix], without copying shared weights."""
    def __init__(self, c, shared, private_count):
        super().__init__()
        self.shared = shared
        self.private = ExpertBank(c, private_count)
        self.shared_count = shared.up.shape[0]

    def forward(self, x, ids, gates):
        b,t,d = x.shape
        k = ids.shape[-1]
        flat_ids, flat_gates = ids.reshape(-1), gates.reshape(-1)
        owners = torch.arange(b*t,device=x.device).repeat_interleave(k)
        flat_x = x.reshape(-1,d)
        out = x.new_zeros(b*t,d)
        for shared, bank in [(True,self.shared),(False,self.private)]:
            selected = ((flat_ids < self.shared_count) == shared).nonzero().flatten()
            if selected.numel() == 0:
                continue
            local = flat_ids[selected] - (0 if shared else self.shared_count)
            token_ids = owners[selected]
            values = bank(flat_x[token_ids][None],local[None,:,None],
                          flat_gates[selected][None,:,None])[0]
            out = out.index_add(0,token_ids,values)
        return out.reshape(b,t,d)/math.sqrt(k)


class ExpertSHoE(SHoEGPT):
    """Original 1M child shape; its vocabulary table decodes latent inputs.

The original standalone token API is inherited. Hierarchical execution accepts
continuous payloads and original position IDs, with no intervening tokenizer.
"""
    def __init__(self,c,shared_banks):
        super().__init__(c)
        for layer,(block,count) in enumerate(zip(self.blocks,c.expert_counts)):
            block.experts = SplitExpertBank(c,shared_banks[layer],count-shared_banks[layer].up.shape[0])

    def probe(self,context):
        evidence = torch.stack([b.router.pilot(context) for b in self.blocks],-2)
        prior, lace = self.lace(evidence,context)
        match = (evidence/4).amax(-1).mean(-1)
        return {'evidence':evidence,'prior':prior,'match':match,'lace_ids':lace['lace_ids']}

    def execute(self,payload,positions,prior,trace=False):
        x = payload + self.position_embedding(positions)
        context = F.layer_norm(x+.25*prefix_mean(x),(self.cfg.width,))
        aux, actual, details = x.new_zeros(()), [], []
        for layer,block in enumerate(self.blocks):
            x,route,memory = block(x,prior[:,:,layer])
            aux = aux+route['aux']
            actual.append(route['group_scores'])
            if trace:
                details.append({k:v.detach() for k,v in {**route,**memory}.items()})
        logits = F.linear(self.final_norm(x),self.token_embedding.weight)
        # This message describes the executed computation. It is a target for
        # parent learning, and cannot feed target information into predictions.
        with torch.no_grad():
            executed_prior, lace = self.lace(torch.stack(actual,-2).detach(),context.detach())
        return {'logits':logits,'aux':aux/self.cfg.layers,'feedback_prior':executed_prior,
                'feedback_lace_ids':lace['lace_ids'],'trace':details}


class ParentSHoE(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c = c
        self.queries = nn.ModuleList([nn.Linear(c.width,c.address_dim,bias=False) for _ in range(c.layers)])
        self.prototypes = nn.Parameter(torch.randn(c.layers,c.groups,c.address_dim))
        self.lace = LaceController(c)
        self.route_context = nn.Linear(c.width,2)
        self.gain_predictor = nn.Linear(c.width,2)
        self.route_affinity = nn.Parameter(torch.randn(c.layers,c.groups,2)*.05)
        # Row-normalized relations translate independent local group spaces.
        relation = torch.full((2,c.layers,c.groups,c.groups),-1.)
        gen = torch.Generator().manual_seed(c.seed+207)
        for child in range(2):
            for layer in range(c.layers):
                relation[child,layer,torch.arange(c.groups),torch.randperm(c.groups,generator=gen)] = 2.
        self.relation_logits = nn.Parameter(relation+.02*torch.randn(relation.shape,generator=gen))
        self.register_buffer('route_mass',torch.zeros(c.layers,c.groups,2))
        self.register_buffer('route_value_sum',torch.zeros(c.layers,c.groups,2))
        self.register_buffer('feedback_updates',torch.zeros((),dtype=torch.long))
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,std=.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def initial(self,context):
        evidence = torch.stack([4*F.linear(normalize(q(context)),normalize(self.prototypes[l]))
                                for l,q in enumerate(self.queries)],-2)
        prior,lace = self.lace(evidence,context)
        return evidence,prior,lace

    def scores(self,prior,context,use_memory=True):
        learned = torch.einsum('...lg,lgc->...c',prior,self.route_affinity)/self.c.layers
        mean_value = self.route_value_sum/(self.route_mass+1e-8)
        values = torch.einsum('...lg,lgc->...c',prior,mean_value)/self.c.layers
        mass = torch.einsum('...lg,lgc->...c',prior,self.route_mass)/self.c.layers
        gains = self.gain_predictor(context)
        scores = self.route_context(context)+learned+.5*gains
        if use_memory: scores = scores+.5*values
        return scores,gains,values,mass


class HierarchicalSHoE(nn.Module):
    def __init__(self,cfg=None):
        super().__init__()
        self.cfg = cfg or HierarchyConfig()
        c = self.cfg.child_config()
        c.validate()
        self.source_embedding = nn.Embedding(c.vocab_size,c.width)
        self.source_position = nn.Embedding(c.context,c.width)
        nn.init.normal_(self.source_embedding.weight,std=.02)
        nn.init.normal_(self.source_position.weight,std=.02)
        self.shared_counts = (171,171,170)
        self.shared_banks = nn.ModuleList([ExpertBank(c,n) for n in self.shared_counts])
        self.children_blocks = nn.ModuleList([ExpertSHoE(self.cfg.child_config(i),self.shared_banks) for i in range(2)])
        self.parent = ParentSHoE(c)

    def encode(self,tokens):
        positions = torch.arange(tokens.shape[1],device=tokens.device)[None].expand(tokens.shape[0],-1)
        return self.source_embedding(tokens)+self.source_position(positions),positions

    def route_only(self,payload,positions=None,reinforce=True,force_discovery=False,
                   disable_upward=False,forced_child=None,scramble_upward=False):
        """No child attention or expert forward is called anywhere in this method."""
        cfg,c = self.cfg,self.cfg.child_config()
        b,t,d = payload.shape
        if positions is None: positions=torch.arange(t,device=payload.device)[None].expand(b,-1)
        context = F.layer_norm(payload+.25*prefix_mean(payload),(d,))
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
            z=payload+child.position_embedding(positions)
            child_contexts.append(F.layer_norm(z+.25*prefix_mean(z),(d,)))

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

    def forward(self,tokens,targets=None,**kwargs):
        payload,positions=self.encode(tokens)
        return self.forward_payload(payload,positions,targets,**kwargs)

    def forward_payload(self,payload,positions=None,targets=None,reinforce=True,trace=False,**routing):
        b,t,d=payload.shape
        if t>self.cfg.context: raise ValueError('Context length exceeded.')
        route=self.route_only(payload,positions,reinforce=reinforce,**routing)
        positions=route['positions']
        c=self.cfg.child_config()
        logits=payload.new_zeros(b*t,c.vocab_size)
        feedback_prior=payload.new_zeros(b*t,c.layers,c.groups)
        executed_laces=torch.zeros(b*t,c.layers,c.active_laces,dtype=torch.long,device=payload.device)
        aux=payload.new_zeros(())
        executions=[]
        # Packed subsequences preserve source order and original position IDs.
        # Every token executes one child. Empty children execute nothing.
        for which,child in enumerate(self.children_blocks):
            for row in range(b):
                index=(route['choice'][row]==which).nonzero().flatten()
                if index.numel()==0: continue
                ctx=route['child_contexts'][which][row,index][None]
                expected=route['expected'][row,index,which]
                down=expected.clamp_min(1e-8).log()
                down=down-down.mean(-1,keepdim=True)
                bottom_prior,_=child.lace(
                    (route['local_evidence'][row,index,which]+self.cfg.downward_gain*down)[None],ctx)
                result=child.execute(payload[row,index][None],positions[row,index][None],bottom_prior,trace)
                flat_index=row*t+index
                logits=logits.index_copy(0,flat_index,result['logits'][0])
                feedback_prior[flat_index]=result['feedback_prior'][0]
                executed_laces[flat_index]=result['feedback_lace_ids'][0]
                aux=aux+result['aux']*index.numel()/(b*t)
                if trace:
                    executions.append({'child':which,'batch_row':row,'source_positions':positions[row,index].detach(),
                                       'trace':result['trace']})
        logits=logits.reshape(b,t,-1)
        picked_prob=route['learning_prob'].gather(-1,route['choice'][...,None])
        # Hard top-1 forward with a selected-gate straight-through gradient.
        # No counterfactual unselected expert output is manufactured.
        logits=logits*(1+picked_prob-picked_prob.detach())
        aux=aux+switch_balance(route['learning_prob'],route['choice'])+.1*route['parent_lace_balance']
        mean_route=route['learning_prob'].mean((0,1))
        parent_balance=-mean_route.clamp_min(1e-8).log().mean()-math.log(2)
        baseline_logits=F.linear(route['context'],self.source_embedding.weight)
        ce=semantic_loss=utility_loss=baseline_loss=None
        packet=None
        if targets is not None:
            losses=F.cross_entropy(logits.reshape(-1,c.vocab_size),targets.reshape(-1),reduction='none').reshape(b,t)
            baseline=F.cross_entropy(baseline_logits.reshape(-1,c.vocab_size),targets.reshape(-1),reduction='none').reshape(b,t)
            ce,baseline_loss=losses.mean(),baseline.mean()
            gain=(baseline.detach()-losses.detach()).clamp(-2,2)
            expected=route['expected'].gather(2,route['choice'][...,None,None,None].expand(-1,-1,1,c.layers,c.groups)).squeeze(2)
            target=feedback_prior.reshape(b,t,c.layers,c.groups).detach()
            kl=(target*(target.clamp_min(1e-8).log()-expected.clamp_min(1e-8).log())).sum(-1).mean(-1)
            credit=.25+.75*gain.sigmoid()
            semantic_loss=(credit*kl).mean()
            predicted_gain=route['gains'].gather(-1,route['choice'][...,None]).squeeze(-1)
            utility_loss=F.smooth_l1_loss(predicted_gain,gain)
            packet={'eligibility':route['parent_prior'].detach(),'choice':route['choice'].detach(),
                    'gain':gain.detach()}
        loss=None
        if ce is not None:
            loss=ce+self.cfg.auxiliary_loss_weight*aux+self.cfg.baseline_loss_weight*baseline_loss
            loss=loss+self.cfg.parent_balance_weight*parent_balance
            if reinforce:
                loss=loss+self.cfg.semantic_loss_weight*semantic_loss+self.cfg.utility_loss_weight*utility_loss
        return {'logits':logits,'ce':ce,'loss':loss,'aux':aux,'semantic_loss':semantic_loss,
                'utility_loss':utility_loss,'baseline_loss':baseline_loss,'packet':packet,
                'route':route,'executions':executions,'executed_lace_ids':executed_laces.reshape(b,t,c.layers,c.active_laces)}

    @torch.no_grad()
    def commit_feedback(self,packet):
        """Explicit next-pass update, after the optimizer step; never in forward."""
        if not self.training: raise RuntimeError('Feedback updates are training-only.')
        if packet is None: return
        eligibility=packet['eligibility'].reshape(-1,3,32)
        selected=F.one_hot(packet['choice'].reshape(-1),2).to(eligibility.dtype)
        gain=packet['gain'].reshape(-1)
        mass=torch.einsum('nlg,nc->lgc',eligibility,selected)/gain.numel()
        value=torch.einsum('nlg,nc,n->lgc',eligibility,selected,gain)/gain.numel()
        decay=self.cfg.memory_decay
        self.parent.route_mass.mul_(decay).add_(mass,alpha=1-decay)
        self.parent.route_value_sum.mul_(decay).add_(value,alpha=1-decay)
        self.parent.feedback_updates.add_(1)

    def parameter_summary(self):
        shared=sum(p.numel() for p in self.shared_banks.parameters())
        private=[sum(p.numel() for b in child.blocks for p in b.experts.private.parameters())
                 for child in self.children_blocks]
        lower_sizes=[sum(p.numel() for p in child.parameters()) for child in self.children_blocks]
        total=sum(p.numel() for p in self.parameters())
        return {'total_unique_parameters':total,'each_child_logical_parameters':lower_sizes,
                'shared_expert_weights':shared,'private_expert_weights_per_child':private,
                'unique_expert_weights':shared+sum(private),'unique_expert_count':1536,
                'shared_experts':512,'dedicated_experts_per_child':512,
                'shared_counts_per_layer':list(self.shared_counts),
                'child_expert_counts':[342,341,341],
                'parent_and_input_parameters':total-(sum(lower_sizes)-shared),
                'active_expert_weights_per_source_token':38400,
                'active_experts_per_source_token':48}

    @torch.no_grad()
    def generate(self,tokens,count=240,temperature=.8,seed=123,**kwargs):
        generator=torch.Generator(device=tokens.device).manual_seed(seed)
        for _ in range(count):
            out=self(tokens[:,-self.cfg.context:],**kwargs)['logits'][:,-1]/temperature
            token=torch.multinomial(out.softmax(-1),1,generator=generator)
            tokens=torch.cat([tokens,token],1)
        return tokens


if __name__=='__main__':
    import json
    m=HierarchicalSHoE()
    print(json.dumps(m.parameter_summary(),indent=2))
