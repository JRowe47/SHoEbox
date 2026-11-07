"""Activation recomputation with explicit tensor boundaries for graph records."""
import torch
from torch.utils.checkpoint import checkpoint
from .ports import map_tensors, tensor_list
from .runtime import training_arithmetic_scope


def checkpoint_tree(function, *inputs):
    """Preserve structured state while retaining only segment-boundary tensors.

    Reentrant checkpointing drops the internal autograd graph as well as its
    saved tensors. Its API only follows flat tensor inputs/outputs, so flatten
    the complete graph state explicitly. Templates contain zero-size markers,
    never live activation references. The training API uses loss.backward();
    autograd.grad probes use the ordinary, non-checkpointed path.
    """
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
