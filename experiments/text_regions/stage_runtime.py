"""Batch compatible contextualizers within one graph execution stage.

Each node retains its own weights and state. Only the independent tensor work
is combined; graph inputs are gathered from one snapshot and commit afterward.
"""
import torch
from torch.nn import functional as F
from torch.nn.modules import module as module_hooks
from .runtime import cached, enabled
from .runtime_kernels import projected_event, direction


def stack(modules, name, transform=None):
    def compute():
        values = []
        for module in modules:
            value = module
            for part in name.split('.'):
                value = getattr(value, part)
            values.append(transform(value) if transform else value)
        return torch.stack(values)
    return cached(tuple(modules), ('stage', name), compute)


def linear(x, weight):
    shape = x.shape
    return torch.bmm(x.reshape(shape[0], -1, shape[-1]), weight.transpose(-1, -2)).reshape(
        *shape[:-1], weight.shape[-2])


def layer_norm(x, modules):
    return torch.vmap(lambda a, w, b: F.layer_norm(a, (a.shape[-1],), w, b))(
        x, stack(modules, 'weight'), stack(modules, 'bias'))


def memory(x, modules, states):
    n, b = x.shape[:2]
    m = modules[0]
    h, k, v = m.H, m.K, m.V
    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    packed = linear(x, stack(modules, 'projection.weight'))
    offset = h*(2*k+v)+2*h
    f = linear(packed[..., offset:offset+v], stack(modules, 'forget_projection.weight')).to(acc)
    f = f+stack(modules, 'dt_bias')[:, None, None].to(acc)
    decay = stack(modules, 'A_log', lambda w: w.to(acc).exp())[:, None].expand(n, b, h).reshape(n*b, h)
    norm = stack(modules, 'norm_weight')[:, None, None, None].expand(n, b, 1, 1, v).reshape(n*b, 1, 1, v).to(acc)
    gate = stack(modules, 'output_gate_bias')[:, None, None].expand(n, b, 1, h*v).reshape(n*b, 1, h*v)
    z, final = projected_event(packed.reshape(n*b, 1, -1), f.reshape(n*b, 1, h*k),
        decay, norm, gate, states.reshape(n*b, h, k, v), h, k, v, m.safe_zero)
    return linear(z.reshape(n, b, 1, h*v), stack(modules, 'output.weight')), final.reshape(n, b, h, k, v)


def prepare_nodes(nodes, payloads, states):
    """Return per-node contextualizers and pilot evidence for full active rows."""
    adapters = tuple(node.adapter for node in nodes)
    x = torch.stack(payloads)[:, :, None]
    n, b = x.shape[:2]
    final = [[] for _ in nodes]
    aux = x.new_zeros(n)
    for level_index in range(len(adapters[0].levels)):
        levels = tuple(adapter.levels[level_index] for adapter in adapters)
        norms = tuple(level.norm for level in levels)
        previous = torch.stack([state.contextualizer[level_index] for state in states])
        context, updated = memory(layer_norm(x, norms), tuple(level.memory for level in levels), previous)
        h = x+context
        q = direction(linear(layer_norm(h, norms), stack(levels, 'query.weight')), levels[0].safe_zero)
        keys = stack(levels, 'keys', lambda w: F.normalize(w, dim=-1))
        scores = 4*linear(q, keys)
        probabilities = scores.softmax(-1)
        active = levels[0].cfg.hat_active_symbols
        ids = scores.topk(active, -1).indices
        weights = probabilities.gather(-1, ids)*keys.shape[1]/active
        values = stack(levels, 'values')
        selected = values[torch.arange(n, device=x.device)[:, None, None, None], ids]
        x = h+(selected*weights[..., None]).sum(-2)/float(active)**.5
        load = probabilities.new_zeros(n, keys.shape[1]).scatter_add_(
            1, ids.reshape(n, -1), probabilities.new_ones(n, b*active))/(b*active)
        aux = aux+keys.shape[1]*(probabilities.reshape(n, -1, keys.shape[1]).mean(1)*load).sum(-1)/len(adapters[0].levels)
        for i in range(n):
            # A carried state owns exactly its own storage, not the other
            # nodes' rows of this temporary batched allocation.
            final[i].append(updated[i].clone())
    x = layer_norm(x, tuple(a.norm for a in adapters))
    pilots = []
    for layer in range(len(nodes[0].core.blocks)):
        routers = tuple(node.core.blocks[layer].router for node in nodes)
        prototypes = stack(routers, 'prototypes', lambda w: F.normalize(w, dim=-1))
        pilots.append(4*linear(F.normalize(linear(x, stack(routers, 'query.weight')), dim=-1), prototypes))
    evidence = torch.stack(pilots, -2)
    return [(x[i], tuple(final[i]), aux[i], evidence[i]) for i in range(n)]


def prepare_stage(model, requests):
    if not enabled() or not model.batch_independent or not torch.is_grad_enabled() or len(requests) < 2:
        return {}
    # Partial masks keep the existing per-node gather/scatter path. Sorting
    # makes execution independent of the caller's requested traversal order.
    if any(getattr(module_hooks, name, {}) for name in (
            '_global_forward_hooks', '_global_forward_pre_hooks', '_global_backward_hooks', '_global_backward_pre_hooks')):
        return {}
    def hooked(node):
        return cached(node.adapter, 'stage_has_hooks', lambda: any(
            m._forward_hooks or m._forward_pre_hooks or m._backward_hooks or m._backward_pre_hooks
            for m in node.adapter.modules()))
    names = sorted(name for name, data in requests.items() if bool(data[3].all()) and not hooked(model.nodes[name]))
    if len(names) < 2:
        return {}
    prepared = prepare_nodes([model.nodes[name] for name in names],
                             [requests[name][0] for name in names],
                             [requests[name][2] for name in names])
    return dict(zip(names, prepared))
