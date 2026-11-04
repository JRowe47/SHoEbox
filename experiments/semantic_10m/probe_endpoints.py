"""Frozen-checkpoint evaluation: permute each layer's lace endpoints, then all."""
import hashlib
import json
from pathlib import Path
import torch
from model import Config, SHoEGPT
from train import load_data, evaluate


def fingerprint(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261103)
    root = Path(__file__).parent
    checkpoint = torch.load(root/'results/full/checkpoint.pt', map_location='cpu', weights_only=True)
    model = SHoEGPT(Config(**checkpoint['config'])).eval()
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    _, val, _, _ = load_data(root/'data/tiny_shakespeare.txt')
    original = model.lace.template_logits.detach().clone()
    before = fingerprint(model)
    baseline = evaluate(model,val,batches=16)
    rows = []
    permutation = torch.randperm(model.cfg.groups,generator=torch.Generator().manual_seed(999))
    try:
        for layer in range(model.cfg.layers):
            with torch.no_grad():
                model.lace.template_logits.copy_(original)
                model.lace.template_logits[:,layer].copy_(original[:,layer,permutation])
            loss = evaluate(model,val,batches=16)
            row = {'layer_one_based':layer+1,'loss':loss,'loss_change':loss-baseline}
            rows.append(row)
            print(json.dumps(row),flush=True)
        with torch.no_grad():
            model.lace.template_logits.copy_(original)
            for layer in range(model.cfg.layers):
                # Independent endpoint permutations, with layer 4 preserving seed 999.
                perm = torch.randperm(model.cfg.groups,generator=torch.Generator().manual_seed(996+layer))
                model.lace.template_logits[:,layer].copy_(original[:,layer,perm])
        all_loss = evaluate(model,val,batches=16)
    finally:
        with torch.no_grad():
            model.lace.template_logits.copy_(original)
    after = fingerprint(model)
    assert before == after
    result = {'normal_loss':baseline,'per_layer_scrambles':rows,
              'all_layers_independently_scrambled_loss':all_loss,
              'all_layers_loss_change':all_loss-baseline,
              'same_fixed_8192_character_probe':True,
              'restored_parameter_hash_verified':True,
              'single_layer_permutation_seed':999,
              'all_layer_permutation_seeds':[996+i for i in range(model.cfg.layers)]}
    (root/'results/full/endpoint_probes.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
