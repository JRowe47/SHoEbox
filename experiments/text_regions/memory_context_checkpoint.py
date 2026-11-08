"""Explicit opt-in migration of the retained model, including AdamW moments."""
from dataclasses import replace
from .config import GraphConfig
from .memory_context import MemoryContextConfig
from .model import RegionLanguageModel
from .hippocampus_checkpoint import migrate_optimizer


def migrate_memory_context(parent, *, lr=None, options=None):
    config = GraphConfig.from_dict(parent['config'])
    original = RegionLanguageModel(config)
    original.load_state_dict(parent['weights'])
    config = replace(config,hippocampus=replace(config.hippocampus,enabled=True,
        cue_clock='event',reinstatement='expert',ca1_policy='bounded'),
        lace_separation=replace(config.lace_separation,enabled=False,alpha=0.,beta=0.),
        memory_context=options or MemoryContextConfig(enabled=True))
    if not config.memory_context.enabled:
        raise ValueError('Migration requires enabled memory contexts.')
    model = RegionLanguageModel(config)
    missing,unexpected = model.load_state_dict(parent['weights'],strict=False)
    if unexpected or any(not (n.startswith(('hippocampus.','memory_interface.')) or '.anchor_' in n) for n in missing):
        raise ValueError(f'Invalid memory-context migration: {missing}, {unexpected}')
    rate = parent['protocol']['lr'] if lr is None else lr
    optimizer = migrate_optimizer(original,parent['optimizer'],model,rate)
    return model,optimizer,tuple(missing)
