"""Bounded fair actor fabric and causal snapshot oracle (single-process CPU).

Transport is separate from neural policy. No background worker, RPC, hardware
concurrency or differentiable cross-device communication is implied by this shim.
The round scheduler commits offers only after all reads of the visible cut.
"""
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Envelope:
    identity: tuple
    endpoint: str
    submitted: int
    deadline: int
    payload: object


@dataclass(frozen=True)
class FabricState:
    queues: dict = field(default_factory=dict)
    seen: dict = field(default_factory=dict)
    cursor: int = 0
    accepted: int = 0
    duplicate: int = 0
    dropped: int = 0
    expired: int = 0
    completed: int = 0
    deferred: int = 0

    def to_dict(self):
        from .ports import plain_dict
        from dataclasses import is_dataclass
        data=plain_dict(self)
        for endpoint,queue in self.queues.items():
            for envelope,raw in zip(queue,data['queues'][endpoint]):
                raw['payload_kind']=type(envelope.payload).__name__ if is_dataclass(envelope.payload) else 'plain'
        return dict(schema=1,state=data)

    @classmethod
    def from_dict(cls,saved):
        from .hippocampus_interface import (CueRequest,BindingOffer,LocalAnchor,EngramHandle,
                                           ReinstatementRequest,BindingHandle,CreditReceipt)
        if saved['schema']!=1: raise ValueError('Unknown fabric schema.')
        data=dict(saved['state']); queues={}
        for endpoint,queue in data['queues'].items():
            restored=[]
            for raw in queue:
                raw=dict(raw); kind=raw.pop('payload_kind'); payload=raw['payload']
                if kind=='CueRequest': payload=CueRequest(**payload)
                elif kind=='BindingOffer':
                    anchor=dict(payload['anchor']); anchor['handle']=EngramHandle(**anchor['handle'])
                    payload=BindingOffer(**(dict(payload)|{'anchor':LocalAnchor(**anchor)}))
                elif kind=='ReinstatementRequest':
                    payload=ReinstatementRequest(**(dict(payload)|{'binding':BindingHandle(**payload['binding']),
                                                                  'handle':EngramHandle(**payload['handle'])}))
                elif kind=='CreditReceipt':
                    payload=CreditReceipt(**(dict(payload)|{'contributors':tuple((BindingHandle(**b),EngramHandle(**a))
                                                                               for b,a in payload['contributors'])}))
                elif kind!='plain': raise ValueError('Unsupported queued payload type.')
                restored.append(Envelope(**(raw|{'payload':payload})))
            queues[endpoint]=tuple(restored)
        return cls(**(data|{'queues':queues}))


class RecallFabric:
    def __init__(self,config):
        self.config=config
        self.endpoints={}

    def register(self,region,handler):
        if region in self.endpoints:
            raise ValueError('Each regional endpoint must have one owner.')
        self.endpoints[region]=handler

    def submit(self,state,envelope):
        if envelope.endpoint not in self.endpoints:
            raise ValueError('Unknown regional endpoint.')
        # Producer monotone sequence is part of the contract. High watermarks
        # keep idempotency bounded; reordered older actions become explicit drops.
        producer,sequence=envelope.identity
        if sequence <= state.seen.get(producer,-1):
            return replace(state,duplicate=state.duplicate+1),'duplicate'
        pending=sum(map(len,state.queues.values()))
        if pending >= self.config.queue_capacity:
            return replace(state,dropped=state.dropped+1),'overload'
        queues=dict(state.queues)
        queues[envelope.endpoint]=(*queues.get(envelope.endpoint,()),envelope)
        seen=dict(state.seen); seen[producer]=sequence
        return replace(state,queues=queues,seen=seen,accepted=state.accepted+1),'queued'

    def service(self,state,round_index,snapshot):
        from collections import deque
        queues=dict(state.queues)
        # Persist round-robin queue order; do not scan the endpoint registry.
        ready=deque(queues)
        results=[]; served=expired=attempts=0
        limit=sum(map(len,queues.values()))
        while ready and served < self.config.service_budget and attempts < limit:
            endpoint=ready.popleft()
            item=queues[endpoint][0]
            attempts+=1
            if item.submitted > round_index:
                # A future head may block its own causal FIFO, never its peers.
                ready.append(endpoint)
                continue
            tail=queues[endpoint][1:]
            if tail:
                queues[endpoint]=tail
                ready.append(endpoint)
            else:
                del queues[endpoint]
            if item.deadline < round_index:
                results.append((item.identity,'expired',None)); expired+=1
            else:
                value=self.endpoints[endpoint](item.payload,snapshot)
                results.append((item.identity,'ok',value))
            served+=1
        queues={endpoint:queues[endpoint] for endpoint in ready}
        pending=sum(map(len,queues.values()))
        return replace(state,queues=queues,cursor=state.cursor+served,expired=state.expired+expired,
                       completed=state.completed+served-expired,deferred=state.deferred+pending),tuple(results)

    @staticmethod
    def snapshot(state):
        # Frozen records and copy-on-write dictionaries are the immutable cut.
        return state

    @staticmethod
    def reset_namespace(state,producer):
        return replace(state,queues={k:tuple(e for e in v if e.identity[0]!=producer)
                                    for k,v in state.queues.items() if any(e.identity[0]!=producer for e in v)},
                       seen={k:v for k,v in state.seen.items() if k!=producer})
