"""Natural-byte streaming on the multi-source graph and resumable sampling.

The three ordinary input views receive the same bytes, using their existing
independent HAT states. Memory producers stay inactive, and their outgoing
edges are masked. This is an explicit interface policy, not learned routing.
"""
from dataclasses import dataclass
import math
import torch
from .multifact_data import OWNERS
from .retained_composition import empty_inputs, masks
from .state import GraphState


def forward(model, values, targets=None, state=None, reset=None, rounds=2, feedback=True):
    if values.ndim != 2 or values.dtype != torch.long or values.shape[1] < 1:
        raise ValueError('Nonempty int64 [batch,time] byte values required.')
    if values.device.type != 'cpu':
        raise ValueError('The reference streaming interface currently runs on CPU.')
    n,t=values.shape
    inputs=empty_inputs(model,n,t)
    for view in ('A','B','C'):inputs[view]=values
    available={v:torch.full((n,t),v in ('A','B','C'),dtype=torch.bool) for v in inputs}
    edges={e.key:torch.zeros(n,t,dtype=torch.bool) for e in model.cfg.edges if e.source in OWNERS}
    return model(inputs,targets,state=state,reset=reset,rounds=rounds,feedback=feedback,
        availability=available,node_masks=masks(model,n,t,tuple(o for o in model.nodes if o not in OWNERS)),
        edge_masks=edges)


@dataclass
class ByteSession:
    """State plus next-byte logits and a private RNG; no text replay on resume."""
    state: GraphState
    next_logits: torch.Tensor
    rng_state: torch.Tensor
    rounds: int = 2
    temperature: float = .8

    def to_dict(self):
        return dict(schema='shoebox-byte-session-v1',state=self.state.to_dict(),
            next_logits=self.next_logits,rng_state=self.rng_state,rounds=self.rounds,temperature=self.temperature)

    @classmethod
    def from_dict(cls,data):
        if data['schema']!='shoebox-byte-session-v1':raise ValueError('Unknown byte-session schema.')
        return cls(GraphState.from_dict(data['state']),data['next_logits'],data['rng_state'],
                   data['rounds'],data['temperature'])


@torch.no_grad()
def start(model,prompt,seed=123,temperature=.8,rounds=2,state=None):
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('Finite positive temperature required.')
    raw=prompt.encode('utf-8') if isinstance(prompt,str) else bytes(prompt)
    if not raw:raise ValueError('A nonempty prompt is required.')
    values=torch.tensor(list(raw),dtype=torch.long)[None]
    was_training=model.training
    try:
        model.eval();result=forward(model,values,state=state,rounds=rounds)
        return ByteSession(result['state'].detach(),result['logits'][:,-1].clone(),
            torch.Generator().manual_seed(seed).get_state(),rounds,temperature)
    finally:model.train(was_training)


@torch.no_grad()
def generate(model,session,count):
    if not isinstance(count,int) or count<0:raise ValueError('Nonnegative integer count required.')
    rng=torch.Generator();rng.set_state(session.rng_state)
    state,logits=session.state,session.next_logits
    pieces=[];cost=0;was_training=model.training
    try:
        model.eval()
        for _ in range(count):
            value=torch.multinomial((logits/session.temperature).softmax(-1),1,generator=rng)
            result=forward(model,value,state=state,rounds=session.rounds)
            pieces.append(value);state=result['state'];logits=result['logits'][:,-1]
            cost+=result['expert_executions']
        values=torch.cat(pieces,1) if pieces else torch.empty(logits.shape[0],0,dtype=torch.long)
        return values,ByteSession(state.detach(),logits.clone(),rng.get_state(),session.rounds,session.temperature),cost
    finally:model.train(was_training)


def main():
    import argparse
    from pathlib import Path
    from .config import GraphConfig
    from .model import RegionLanguageModel
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--prompt',default='ROMEO:\n')
    p.add_argument('--count',type=int,default=128);p.add_argument('--seed',type=int,default=123)
    p.add_argument('--temperature',type=float,default=.8);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume-session',type=Path)
    a=p.parse_args();torch.set_num_threads(1)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=RegionLanguageModel(GraphConfig.from_dict(saved['config']));model.load_state_dict(saved['weights'])
    import hashlib
    fingerprint=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    if a.resume_session:
        old=torch.load(a.resume_session,map_location='cpu',weights_only=True)
        if old['checkpoint_sha256']!=fingerprint:raise ValueError('Session belongs to a different checkpoint.')
        session=ByteSession.from_dict(old['session'])
    else:session=start(model,a.prompt,a.seed,a.temperature)
    tokens,session,_=generate(model,session,a.count)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_bytes(bytes(tokens[0].tolist()))
    torch.save(dict(checkpoint_sha256=fingerprint,session=session.to_dict()),a.out.with_suffix('.session.pt'))
    print(bytes(tokens[0].tolist()).decode('utf-8',errors='replace'))


if __name__=='__main__':main()
