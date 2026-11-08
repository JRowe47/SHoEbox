"""Per-row regional state and two genuine publications per routed port."""
from dataclasses import replace
import torch
from torch.nn import functional as F
from .transport import Frame
from .region import RegionState


def vector(value, batch, device):
    return value if torch.is_tensor(value) else torch.full((batch,),value,dtype=torch.long,device=device)


def validity(frame):
    return frame.valid if frame.valid is not None else torch.ones(frame.values.shape[0],dtype=torch.bool,device=frame.values.device)


def full_order(rows, batch):
    if rows.numel()!=batch:return None
    return torch.equal(rows,torch.arange(batch,device=rows.device))


def scatter_tensor(old, new, rows):
    ordered=full_order(rows,old.shape[0])
    if ordered is not None:
        return new if ordered else new.index_select(0,rows.argsort())
    return old.index_copy(0,rows,new)


def empty_frame(source, batch, width, device):
    return Frame(source,torch.full((batch,),-1,dtype=torch.long,device=device),
                 torch.full((batch,),-1,dtype=torch.long,device=device),
                 torch.zeros(batch,width,dtype=torch.long,device=device),
                 valid=torch.zeros(batch,dtype=torch.bool,device=device))


def take_frame(frame, rows, *, scalar=False):
    batch = frame.values.shape[0]
    take = lambda t: None if t is None else t.index_select(0,rows)
    event = vector(frame.event,batch,rows.device).index_select(0,rows)
    revision = vector(frame.revision,batch,rows.device).index_select(0,rows)
    if scalar:
        event, revision = int(event.item()), int(revision.item())
    return replace(frame,event=event,revision=revision,values=take(frame.values),
                   distribution=take(frame.distribution),soft=take(frame.soft),strength=take(frame.strength),
                   valid=take(frame.valid),episode_events=take(frame.episode_events),
                   episode_revisions=take(frame.episode_revisions))


def scatter_frame(old, new, rows):
    batch = old.values.shape[0]
    ordered=full_order(rows,batch)
    if ordered is not None:
        result=new if ordered else take_frame(new,rows.argsort())
        return replace(result,origin_source=new.origin_source or old.origin_source)
    values = old.values.index_copy(0,rows,new.values)
    def payload(key, *, hard=False):
        a,b = getattr(old,key),getattr(new,key)
        if a is None and b is None:
            return None
        sample = b if b is not None else a
        if a is None:
            a = F.one_hot(old.values,256).to(sample.dtype) if hard else sample.new_zeros(batch,old.width,256)
        if b is None:
            b = F.one_hot(new.values,256).to(sample.dtype) if hard else sample.new_zeros(len(rows),old.width,256)
        return a.index_copy(0,rows,b)
    def optional(key, fill):
        a,b = getattr(old,key),getattr(new,key)
        if a is None and b is None:
            return None
        sample = b if b is not None else a
        if a is None: a = sample.new_full((batch,),fill)
        if b is None: b = sample.new_full((len(rows),),fill)
        return a.index_copy(0,rows,b)
    return replace(old, values=values,
                   event=vector(old.event,batch,rows.device).index_copy(0,rows,vector(new.event,len(rows),rows.device)),
                   revision=vector(old.revision,batch,rows.device).index_copy(0,rows,vector(new.revision,len(rows),rows.device)),
                   valid=validity(old).index_copy(0,rows,validity(new)),
                   distribution=payload('distribution',hard=True),soft=payload('soft'),
                   strength=optional('strength',1),origin_source=new.origin_source or old.origin_source,
                   episode_events=optional('episode_events',-1),episode_revisions=optional('episode_revisions',-1))


def publish_rows(ports, name, outputs, rows, batch):
    result = dict(ports)
    for width, frame in outputs.items():
        key = f'{name}:{width}'
        history = result.get(key)
        previous, latest = history if history else (empty_frame(name,batch,width,rows.device),empty_frame(name,batch,width,rows.device))
        fresh = vector(latest.event,batch,rows.device).index_select(0,rows) != frame.event
        shift = rows[fresh]
        if shift.numel():
            previous = scatter_frame(previous,take_frame(latest,shift),shift)
        result[key] = (previous,scatter_frame(latest,frame,rows))
    return result


def take_node(node, rows):
    return RegionState(tuple(t.index_select(0,rows) for t in node.hat),
                       tuple(t.index_select(0,rows) for t in node.blocks),
                       node.clock.index_select(0,rows),node.bytes_seen.index_select(0,rows))


def scatter_node(old, new, rows):
    ordered=full_order(rows,old.clock.numel())
    if ordered is not None:
        return new if ordered else take_node(new,rows.argsort())
    return RegionState(tuple(a.index_copy(0,rows,b) for a,b in zip(old.hat,new.hat)),
                       tuple(a.index_copy(0,rows,b) for a,b in zip(old.blocks,new.blocks)),
                       old.clock.index_copy(0,rows,new.clock),old.bytes_seen.index_copy(0,rows,new.bytes_seen))
