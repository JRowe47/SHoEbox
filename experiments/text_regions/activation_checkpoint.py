"""Activation recomputation with explicit tensor boundaries for graph records."""
import torch
from torch.utils.checkpoint import checkpoint
from .ports import map_tensors, tensor_list
from .runtime import training_arithmetic_scope


class _SparseRecompute(torch.autograd.Function):
    """Reentrant recomputation without materializing unused page cotangents.

    PyTorch's ordinary reentrant checkpoint supplies zero gradients for unused
    structured outputs. Skipping truly unused outputs preserves sparse AdamW
    behavior while releasing the long forward graph, including its metadata.
    Training uses loss.backward(); autograd.grad probes use the ordinary path.
    """
    @staticmethod
    def forward(ctx, function, *inputs):
        ctx.function=function
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*inputs)
        ctx.cpu_rng=torch.get_rng_state()
        ctx.devices=sorted({x.device.index for x in inputs if x.device.type=='cuda'})
        ctx.device_rng=[torch.cuda.get_rng_state(d) for d in ctx.devices]
        with torch.no_grad():
            return function(*inputs)

    @staticmethod
    def backward(ctx, *gradients):
        inputs=tuple(x.detach().requires_grad_(x.requires_grad) for x in ctx.saved_tensors)
        with torch.random.fork_rng(devices=ctx.devices):
            torch.set_rng_state(ctx.cpu_rng)
            for device,rng in zip(ctx.devices,ctx.device_rng):
                torch.cuda.set_rng_state(rng,device)
            with torch.enable_grad():
                outputs=ctx.function(*inputs)
            active=[(x,g) for x,g in zip(outputs,gradients) if g is not None and x.requires_grad]
            if active:
                torch.autograd.backward(tuple(x for x,_ in active),tuple(g for _,g in active))
        return (None,*(x.grad if x.requires_grad else None for x in inputs))


def enable_associative_checkpointing(model, segment=16, min_positions=768):
    """Execution-only training wrapper; weights/optimizer/protocol stay intact."""
    if getattr(model, '_associative_checkpointing', False):
        return model
    if model.hippocampus is None:
        raise ValueError('This execution wrapper requires associative state.')
    if type(segment) is not int or segment<1 or type(min_positions) is not int or min_positions<0:
        raise ValueError('Checkpoint segment must be positive.')
    original = model.forward

    def forward(inputs, *args, **kwargs):
        if not model.training or not torch.is_grad_enabled():
            return original(inputs, *args, **kwargs)
        if args:
            raise ValueError('Checkpointed training supplies controls by keyword.')
        options = dict(kwargs)
        state = options.pop('state', None)
        if state is None:
            first = next(iter(inputs.values())) if isinstance(inputs, dict) else inputs
            values = first if torch.is_tensor(first) else first.values
            state = model.initial_state(values.shape[0], values.device)
        elif state.hippocampus is None:
            from dataclasses import replace
            memory = model.hippocampus.initial_state(len(state.events_seen),state.events_seen.device,
                                                     model.readout.weight.dtype)
            memory = replace(memory, rows=tuple(replace(row, namespace=int(state.epochs[i]),
                round=int(state.rounds_seen[i])) for i,row in enumerate(memory.rows)))
            state = replace(state,hippocampus=memory)
        first = next(iter(inputs.values())) if isinstance(inputs,dict) else inputs
        values = first if torch.is_tensor(first) else first.values
        length=values.shape[1]
        if values.numel()<=min_positions:
            return original(inputs,state=state,**options)
        if length<=segment:
            return checkpoint_tree(lambda x,s: original(x,state=s,**options),inputs,state)
        if any(options.get(k) for k in ('trace','return_view_activity','return_edge_activity',
                'return_round_activity','return_node_activity','return_publication_activity',
                'return_recall_activity','return_write_activity','substitutions')):
            raise ValueError('Segmented training recomputation does not aggregate diagnostic captures.')
        targets=options.pop('targets',None)
        def cut(value,begin,end):
            if isinstance(value,dict): return {k:cut(v,begin,end) for k,v in value.items()}
            return value[:,begin:end] if torch.is_tensor(value) else value.slice(begin,end)
        parts=[]
        for begin in range(0,length,segment):
            end=min(length,begin+segment); local=dict(options)
            for key in ('availability','node_masks','edge_masks'):
                if key in local: local[key]=cut(local[key],begin,end)
            if local.get('reset') is not None:
                reset=local['reset']
                local['reset']=cut(reset,begin,end) if reset.ndim==2 else (reset if begin==0 else None)
            # Bind each segment's controls now; backward occurs after this loop.
            function=lambda x,s,controls=local:original(x,state=s,**controls)
            result=checkpoint_tree(function,cut(inputs,begin,end),state)
            parts.append(result); state=result['state']
        result=dict(parts[-1]); result['logits']=torch.cat([r['logits'] for r in parts],1)
        per_node=3*model.cfg.active_groups*model.cfg.experts_per_group
        activations=[r['expert_executions']/per_node for r in parts]
        hats=[n+r['source_occurrences'] for n,r in zip(activations,parts)]
        result['aux']=sum(r['aux']*n for r,n in zip(parts,activations))/max(sum(activations),1)
        result['hat_aux']=sum(r['hat_aux']*n for r,n in zip(parts,hats))/max(sum(hats),1)
        for key in ('expert_executions','external_events','source_occurrences','internal_rounds'):
            result[key]=sum(r[key] for r in parts)
        result['hippocampus_work']=[w for r in parts for w in r['hippocampus_work']]
        if targets is not None:
            from torch.nn import functional as F
            result['ce']=F.cross_entropy(result['logits'].reshape(-1,256),targets.reshape(-1),ignore_index=-100)
            result['loss']=result['ce']+model.cfg.auxiliary_loss_weight*result['aux']+model.cfg.hat_balance_weight*result['hat_aux']
        return result

    model.forward = forward
    model._associative_checkpointing = True
    model._associative_checkpoint_segment = segment
    model._associative_checkpoint_min_positions = min_positions
    return model


def checkpoint_tree(function, *inputs):
    """Preserve structured state while retaining only segment-boundary tensors.

    Reentrant checkpointing drops the internal autograd graph as well as its
    saved tensors. Its API only follows flat tensor inputs/outputs, so flatten
    the complete graph state explicitly. Templates contain zero-size markers,
    never live activation references. The training API uses loss.backward();
    autograd.grad probes use the ordinary, non-checkpointed path.
    """
    def has_associative_state(value):
        if getattr(value, 'hippocampus', None) is not None:
            return True
        if isinstance(value, dict):
            return any(has_associative_state(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return any(has_associative_state(v) for v in value)
        return False

    associative=has_associative_state(inputs)
    marker = torch.empty(0)
    template = map_tensors(inputs, lambda unused: marker)
    flat_inputs = tensor_list(inputs)
    output_template = []

    def recompute(anchor, *flat):
        iterator = iter(flat)
        args = map_tensors(template, lambda unused: next(iterator))
        with training_arithmetic_scope():
            result = function(*args)
        if not output_template:
            output_template.append(map_tensors(result, lambda unused: marker))
        return tuple(tensor_list(result))

    # Also works at a fresh/reset boundary whose input tensors have no history.
    # This scalar is an autograd anchor, never a trainable model coefficient.
    device = flat_inputs[0].device
    anchor = torch.zeros((), device=device, requires_grad=True)
    outputs = (_SparseRecompute.apply(recompute, anchor, *flat_inputs) if associative else
               checkpoint(recompute, anchor, *flat_inputs, use_reentrant=True))
    iterator = iter(outputs)
    return map_tensors(output_template[0], lambda unused: next(iterator))
