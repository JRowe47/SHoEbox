"""Versioned skeleton checkpoints: weights, optimizer, full state and RNG."""
from pathlib import Path
import os
import torch
from .config import SkeletonConfig
from .model import CorticalModel, CorticalState


FORMAT = 'shoebox-cortical-v2-skeleton-1'
HIERARCHY_FORMAT = 'shoebox-predictive-hierarchy-1'


def save(path, model, state, optimizer=None, *, step=0, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from .hierarchy import HierarchyModel
    format_name = HIERARCHY_FORMAT if isinstance(model, HierarchyModel) else FORMAT
    data = {'format': format_name, 'config': model.cfg.to_dict(), 'model': model.state_dict(),
            'state': None if state is None else state.detach().to_dict(),
            'optimizer': None if optimizer is None else optimizer.state_dict(),
            'rng': torch.get_rng_state(), 'step': step, 'metadata': metadata or {}}
    if isinstance(model, HierarchyModel):
        data['objective_config'] = model.objective_config.to_dict()
        if optimizer is not None:
            names = {id(p): n for n, p in model.named_parameters()}
            data['optimizer_names'] = [[names[id(p)] for p in g['params']] for g in optimizer.param_groups]
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(data, temporary)
    os.replace(temporary, path)


def load(path, *, device='cpu', optimizer_factory=None):
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get('format') not in (FORMAT, HIERARCHY_FORMAT):
        raise ValueError('Expected a cortical-v2 skeleton checkpoint; legacy checkpoints are incompatible.')
    dtype = next(value.dtype for value in data['model'].values() if value.is_floating_point())
    if data['format'] == HIERARCHY_FORMAT:
        from .hierarchy_config import HierarchyConfig, ObjectiveConfig
        from .hierarchy import HierarchyModel, HierarchyState
        model = HierarchyModel(HierarchyConfig(**data['config']),
                ObjectiveConfig(**data.get('objective_config', {}))).to(device=device, dtype=dtype)
        state_type = HierarchyState
    else:
        model = CorticalModel(SkeletonConfig(**data['config'])).to(device=device, dtype=dtype)
        state_type = CorticalState
    model.load_state_dict(data['model'], strict=True)
    state = None if data['state'] is None else state_type.from_dict(data['state'])
    if state is not None:
        if state.signature != data['config']:
            raise ValueError('Saved state/configuration signatures disagree.')
        # Older v2 files omit newly added fields whose defaults preserve v2.
        state.signature = model.cfg.to_dict()
    optimizer = None if optimizer_factory is None else optimizer_factory(model.parameters())
    if optimizer is not None and data['optimizer'] is not None:
        optimizer.load_state_dict(data['optimizer'])
    torch.set_rng_state(data['rng'].cpu())
    return model, state, optimizer, data
