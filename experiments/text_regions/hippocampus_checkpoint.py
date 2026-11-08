"""Name-based parent migration; moments are initialized only for new tensors."""
from copy import deepcopy
import torch


def migrate_optimizer(parent_model,parent_state,model,lr):
    old_names=[name for name,_ in parent_model.named_parameters()]
    old_ids=[i for group in parent_state['param_groups'] for i in group['params']]
    if len(old_names)!=len(old_ids) or len(parent_state['param_groups'])!=1:
        raise ValueError('Expected the retained single-group AdamW parent.')
    old=dict(zip(old_names,old_ids))
    settings={k:v for k,v in parent_state['param_groups'][0].items() if k not in ('params','param_names')}
    settings['lr']=lr
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    fresh=optimizer.state_dict()
    fresh['param_groups'][0].update(settings)
    names=[name for name,_ in model.named_parameters()]
    for name,new_id in zip(names,fresh['param_groups'][0]['params']):
        if name in old and old[name] in parent_state['state']:
            fresh['state'][new_id]=deepcopy(parent_state['state'][old[name]])
    optimizer.load_state_dict(fresh)
    return optimizer


def load_parent(model,parent):
    missing,unexpected=model.load_state_dict(parent['weights'],strict=False)
    if unexpected or any(not (n.startswith('hippocampus.') or '.anchor_' in n) for n in missing):
        raise ValueError(f'Unexpected migration mismatch: {missing}, {unexpected}')
    return tuple(missing)
