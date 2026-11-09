"""Regional AdamW groups and a resumable bidirectional learning cycle.

Only parameter learning is paused. Forward execution, recurrent state updates,
and derivatives with respect to inputs remain live through frozen regions.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy
import math
import torch
from .train_fineweb import optimizer_for


def parameter_role(name):
    if name.startswith(('embedding.', 'regions.A1.', 'regions.B1.')):
        return 'fast'
    if name.startswith('decoder.local.'):
        return 'local_projection'
    if name.startswith(('regions.L2.', 'regions.S2.', 'regions.L3.')):
        return 'upper'
    if name.startswith(('future_heads.', 'regional_objectives.predictors.')):
        return 'heads'
    if name.startswith(('decoder.', 'regions.Y.')):
        return 'reader'
    raise ValueError(f'Unassigned hierarchy parameter: {name}')


def parameter_owner(name):
    """Ownership metadata for later local/global gradient routing, not a detach."""
    if name.startswith('regional_objectives.predictors.'):
        return name.split('.')[2]
    if name.startswith('regions.'):
        return name.split('.')[1]
    return parameter_role(name)


def restore_named_optimizer(model, optimizer, checkpoint):
    """Restore Adam by verified names, never onto an extended positional list.

    Old checkpoints omitted names. Reconstruct only their two documented group
    schemas from saved model names, roles, decay flags and exact tensor shapes.
    New checkpoints always carry explicit per-group names.
    """
    saved = checkpoint['optimizer']
    if saved is None:
        raise ValueError('Continuation requires saved Adam history.')
    parameters = dict(model.named_parameters())
    shared = {n: p for n, p in parameters.items() if n in checkpoint['model']}
    for name, p in shared.items():
        if p.shape != checkpoint['model'][name].shape:
            raise ValueError(f'Optimizer parameter shape changed: {name}')
    names = checkpoint.get('optimizer_names')
    if names is None:
        if checkpoint.get('objective_config', {}).get('mode', 'legacy') != 'legacy':
            raise ValueError('Latent checkpoints require named optimizer state.')
        names = []
        for group in saved['param_groups']:
            selected = [n for n, p in shared.items()
                        if ('role' not in group or parameter_role(n) == group['role'])
                        and (p.ndim < 2 or getattr(p, '_no_weight_decay', False)) == (group['weight_decay'] == 0)]
            names.append(selected)
    if len(names) != len(saved['param_groups']):
        raise ValueError('Optimizer name/group count mismatch.')
    flat = [n for group in names for n in group]
    if len(flat) != len(set(flat)) or set(flat) != set(shared):
        raise ValueError('Optimizer names do not cover exactly the saved parameters.')
    current_groups = {id(p): g for g in optimizer.param_groups for p in g['params']}
    seen_ids, ages = set(), []
    for group, group_names in zip(saved['param_groups'], names):
        if len(group['params']) != len(group_names):
            raise ValueError('Optimizer group length mismatch.')
        for key, name in zip(group['params'], group_names):
            if key in seen_ids:
                raise ValueError('Duplicate saved optimizer parameter ID.')
            seen_ids.add(key)
            p = parameters[name]
            if 'role' in group and group['role'] != parameter_role(name):
                raise ValueError(f'Optimizer role mismatch: {name}')
            destination = current_groups[id(p)]
            for option in ('betas', 'eps', 'amsgrad', 'weight_decay'):
                if group[option] != destination[option]:
                    raise ValueError(f'Optimizer {option} changed for {name}.')
            if key not in saved['state']:
                continue
            state = deepcopy(saved['state'][key])
            for option in ('exp_avg', 'exp_avg_sq'):
                if state[option].shape != p.shape:
                    raise ValueError(f'Adam tensor shape mismatch: {name}')
            optimizer.state[p] = state
            ages.append(float(state['step']))
    if set(saved['state'])-seen_ids:
        raise ValueError('Orphan saved optimizer state.')
    return dict(shared_parameters=len(shared), new_parameters=sorted(set(parameters)-set(shared)),
                restored_states=len(ages), min_age=min(ages) if ages else 0, max_age=max(ages) if ages else 0,
                mapping='saved names' if 'optimizer_names' in checkpoint else 'verified legacy roles/shapes')


def regional_optimizer(model, lr, weight_decay=.01, legacy_state=None):
    groups = {}
    for name, parameter in model.named_parameters():
        decay = 0. if parameter.ndim < 2 or getattr(parameter, '_no_weight_decay', False) else weight_decay
        role = parameter_role(name)
        groups.setdefault((role, decay), []).append(parameter)
    optimizer = torch.optim.AdamW([
        dict(params=params, role=role, weight_decay=decay)
        for (role, decay), params in groups.items()
    ], lr=lr, betas=(.9, .95), foreach=True)
    if legacy_state is not None:
        restore_named_optimizer(model, optimizer, dict(optimizer=legacy_state, model=model.state_dict()))
    return optimizer


@dataclass(frozen=True)
class CycleConfig:
    mode: str = 'joint'
    warmup: int = 64
    upper: int = 32
    lower: int = 32
    joint: int = 64
    fast_lr_scale: float = .5
    minimum_span_improvement: float = .02
    attenuation_every: int = 4
    local_scale: float = .5

    def __post_init__(self):
        if self.mode not in ('joint', 'cycle'):
            raise ValueError('Training mode must be joint or cycle.')
        for name in ('warmup', 'upper', 'lower', 'joint', 'attenuation_every'):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in ('upper', 'lower', 'joint') else 0):
                raise ValueError(f'Invalid cycle {name}.')
        for name in ('fast_lr_scale', 'minimum_span_improvement', 'local_scale'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid cycle {name}.')
        if self.fast_lr_scale == 0:
            raise ValueError('Joint-phase fast learning rate must be positive.')

    def to_dict(self):
        return asdict(self)


class RegionalCycle:
    def __init__(self, config=None):
        self.config = config or CycleConfig()
        self.reference, self.ema = {}, {}
        self.lower_ready = False

    def observe(self, row):
        for scale in ('mid', 'high'):
            value = float(row[scale+'_nll'])
            if not math.isfinite(value) or value <= 0:
                continue
            self.reference.setdefault(scale, value)
            self.ema[scale] = .9*self.ema.get(scale, value)+.1*value
        # Latch readiness: a difficult subsequent chunk does not erase evidence
        # that these mappings learned. No held-out targets enter the controller.
        self.lower_ready |= all(
            scale in self.ema and self.ema[scale] < self.reference[scale]*(1-self.config.minimum_span_improvement)
            for scale in ('mid', 'high'))

    def state_dict(self):
        return dict(reference=dict(self.reference), ema=dict(self.ema), lower_ready=self.lower_ready)

    def load_state_dict(self, state):
        self.reference, self.ema = dict(state['reference']), dict(state['ema'])
        self.lower_ready = bool(state['lower_ready'])

    def plan(self, step):
        c = self.config
        if step < 0:
            raise ValueError('Step must be nonnegative.')
        if c.mode == 'joint':
            return dict(phase='joint', local_scale=1., fast_lr_scale=1.)
        if step < c.warmup:
            return dict(phase='warmup', local_scale=1., fast_lr_scale=1.)
        position = (step-c.warmup) % (c.upper+c.lower+c.joint)
        if position < c.upper:
            scale = c.local_scale if c.attenuation_every and (position+1) % c.attenuation_every == 0 else 1.
            return dict(phase='upper', local_scale=scale, fast_lr_scale=1.)
        if position < c.upper+c.lower:
            return dict(phase='lower' if self.lower_ready else 'joint_wait', local_scale=1.,
                        fast_lr_scale=1. if self.lower_ready else c.fast_lr_scale)
        return dict(phase='joint', local_scale=1., fast_lr_scale=c.fast_lr_scale)

    def apply(self, model, optimizer, step, lr):
        if self.config.mode == 'cycle' and model.cfg.local_only:
            raise ValueError('Regional cycles require the full hierarchy.')
        plan = self.plan(step)
        active = ({'upper', 'heads', 'reader'} if plan['phase'] == 'upper' else
                  {'fast'} if plan['phase'] == 'lower' else
                  {'fast', 'local_projection', 'upper', 'heads', 'reader'})
        for group in optimizer.param_groups:
            role = group['role']
            group['lr'] = lr*(plan['fast_lr_scale'] if role in ('fast', 'local_projection') else 1.)
            for parameter in group['params']:
                parameter.requires_grad_(role in active)
                # None makes AdamW skip moments AND decoupled weight decay.
                # Setting lr=0 or zeroing gradients would still age its moments.
                parameter.grad = None
        return dict(**plan, lower_ready=self.lower_ready,
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))


def gradient_norms(model):
    buckets = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            parts = name.split('.')
            key = ('predictor_'+parts[2] if parts[0] == 'regional_objectives' else
                   parts[1] if parts[0] in ('regions', 'future_heads') else parts[0])
            buckets.setdefault(key, []).append(parameter.grad)
    return {key: float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(values))))
            for key, values in buckets.items()}
