"""Activation recomputation with explicit tensor boundaries for graph records."""
import torch
from torch.utils.checkpoint import checkpoint
from .ports import map_tensors, tensor_list
from .runtime import training_arithmetic_scope


def enable_associative_checkpointing(model):
    """Execution-only training wrapper; weights/optimizer/protocol stay intact."""
    if getattr(model, '_associative_checkpointing', False):
        return model
    if model.hippocampus is None:
        raise ValueError('This execution wrapper requires associative state.')
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
        return checkpoint_tree(lambda x,s: original(x,state=s,**options),inputs,state)

    model.forward = forward
    model._associative_checkpointing = True
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

    if has_associative_state(inputs):
        # Reentrant checkpoint materializes zero cotangents for unused returned
        # pages. Those zeros allocate AdamW moments/decay for cold owner weights.
        # Non-reentrant recomputation preserves None gradients and structured
        # sparse state; legacy execution keeps its original checkpoint engine.
        def structured(*args):
            with training_arithmetic_scope():
                return function(*args)
        return checkpoint(structured, *inputs, use_reentrant=False)

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
    outputs = checkpoint(recompute, anchor, *flat_inputs, use_reentrant=True)
    iterator = iter(outputs)
    return map_tensors(output_template[0], lambda unused: next(iterator))
