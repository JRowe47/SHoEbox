"""Selective producer-state commits at unchanged candidate computation."""
import torch

from .state import NodeState


def commit_producer_state(previous, candidate, actions, probabilities=None, eligible=None):
    """Commit owner contents, retaining the candidate execution clock.

    Rows use the model's flattened batch/owner order. A hard hold is an exact
    identity for both contextualizer and recurrent block contents. Optional
    zero-valued tangents train the existing admission policy inside the live
    graph. They do not restore recurrent derivatives across detach boundaries.
    """
    if actions.dtype != torch.bool or actions.numel() != previous.clock.numel():
        raise ValueError('State actions must be bool with one entry per owner row.')
    actions = actions.reshape(-1)
    tangent = None
    if probabilities is not None:
        if eligible is None or probabilities.shape != eligible.shape or probabilities.numel() != actions.numel():
            raise ValueError('State policy probabilities and eligibility must match owner rows.')
        tangent = ((probabilities - probabilities.detach()) * eligible).reshape(-1)

    def commit(old, new):
        shape = (-1,) + (1,) * (old.ndim - 1)
        selected = torch.where(actions.reshape(shape), new, old)
        if tangent is not None:
            selected = selected + tangent.reshape(shape) * (new - old)
        return selected

    return NodeState(tuple(commit(a, b) for a, b in zip(previous.contextualizer, candidate.contextualizer)),
                     tuple(commit(a, b) for a, b in zip(previous.blocks, candidate.blocks)),
                     candidate.clock)
