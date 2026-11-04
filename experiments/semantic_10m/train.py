"""Reproducible CPU shakedown; fixed validation probes; resumable final checkpoint."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import time
import torch
from model import Config, SHoEGPT


def load_data(path):
    raw = Path(path).read_text()
    cut1, cut2 = int(.9*len(raw)), int(.95*len(raw))
    chars = sorted(set(raw[:cut1]))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in raw[:cut2]], dtype=torch.long)
    return data[:cut1], data[cut1:cut2], chars, {
        'source_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'characters': len(raw), 'train_characters': cut1,
        'validation_characters': cut2-cut1, 'test_characters_untouched': len(raw)-cut2,
        'split': 'contiguous 90% train / 5% validation / 5% untouched test',
        'tokenizer': 'character vocabulary derived from training split',
    }


def batch(data, size, context, generator):
    starts = torch.randint(len(data)-context, (size,), generator=generator)
    ids = starts[:, None]+torch.arange(context+1)
    chunk = data[ids]
    return chunk[:, :-1], chunk[:, 1:]


@torch.no_grad()
def evaluate(model, data, batches=16, size=4, seed=333, **kwargs):
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(batches):
        x,y = batch(data, size, model.cfg.context, gen)
        losses.append(float(model(x,y,**kwargs)['ce']))
    model.train(was_training)
    return sum(losses)/len(losses)


@torch.no_grad()
def diagnostics(model, data, batches=8, lace_enabled=True):
    model.eval()
    gen = torch.Generator().manual_seed(5151)
    expert_counts = [torch.zeros(n, dtype=torch.long) for n in model.cfg.expert_counts]
    group_counts = [torch.zeros(model.cfg.groups, dtype=torch.long) for _ in range(model.cfg.layers)]
    token_group_counts = [torch.zeros(model.cfg.groups, dtype=torch.long) for _ in range(model.cfg.layers)]
    lace_counts = torch.zeros(model.cfg.laces, dtype=torch.long)
    changed_experts, total_experts = 0, 0
    changed_groups, total_groups = 0, 0
    pairs, dense_pairs = [0]*model.cfg.layers, 0
    example = None
    for _ in range(batches):
        x,y = batch(data, 4, model.cfg.context, gen)
        normal = model(x,y,trace=True,lace_enabled=lace_enabled)
        disabled = model(x,y,trace=True,lace_enabled=False)
        if normal['lace_ids'] is not None:
            lace_counts += torch.bincount(normal['lace_ids'].flatten(), minlength=model.cfg.laces)
        dense_pairs += x.shape[0]*x.shape[1]*(x.shape[1]+1)//2
        for l,(a,b) in enumerate(zip(normal['trace'], disabled['trace'])):
            expert_counts[l] += torch.bincount(a['experts'].flatten(), minlength=model.cfg.expert_counts[l])
            group_counts[l] += torch.bincount(a['groups'].flatten(), minlength=model.cfg.groups)
            token_group_counts[l] += torch.bincount(a['assignments'].flatten(), minlength=model.cfg.groups)
            shared = (a['experts'][..., :, None] == b['experts'][..., None, :]).any(-1)
            changed_experts += int((~shared).sum())
            total_experts += shared.numel()
            same_groups = (a['groups'][..., :, None] == b['groups'][..., None, :]).any(-1)
            changed_groups += int((~same_groups).sum())
            total_groups += same_groups.numel()
            pairs[l] += int(a['valid'].sum())
        if example is None:
            example = {
                'token_ids': x[0].tolist(),
                'lace_ids_at_last_token': normal['lace_ids'][0,-1].tolist() if normal['lace_ids'] is not None else None,
                'layers': [{'groups': a['groups'][0,-1].tolist(),
                            'experts': a['experts'][0,-1].tolist(),
                            'key_positions': a['reads'][0,-1][a['valid'][0,-1]].tolist()}
                           for a in normal['trace']]}
    return {
        'probe_tokens': batches*4*model.cfg.context,
        'experts_used_per_layer': [int((a>0).sum()) for a in expert_counts],
        'groups_used_per_layer': [int((a>0).sum()) for a in group_counts],
        'token_groups_used_per_layer': [int((a>0).sum()) for a in token_group_counts],
        'laces_used': int((lace_counts>0).sum()),
        'expert_usage': [a.tolist() for a in expert_counts],
        'group_usage': [a.tolist() for a in group_counts],
        'token_group_usage': [a.tolist() for a in token_group_counts],
        'lace_usage': lace_counts.tolist(),
        'selected_experts_changed_when_laces_disabled_fraction': changed_experts/total_experts,
        'selected_groups_changed_when_laces_disabled_fraction': changed_groups/total_groups,
        'valid_attention_pairs_per_layer': pairs,
        'dense_causal_pairs_per_layer': dense_pairs,
        'valid_pair_fraction_per_layer': [p/dense_pairs for p in pairs],
        'attention_slots_per_query': model.cfg.local_window+model.cfg.active_groups*model.cfg.keys_per_group,
        'example': example,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/tiny_shakespeare.txt')
    parser.add_argument('--out', default='results/demo')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.002)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--no-lace', action='store_true')
    parser.add_argument('--scale', choices=['1m', '10m'], default='10m')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train, val, chars, data_meta = load_data(args.data)
    config_factory = Config.scaled_10m if args.scale == '10m' else Config
    model = SHoEGPT(config_factory(vocab_size=len(chars), seed=args.seed))
    initial_parameter_sha256 = hashlib.sha256(b''.join(
        p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9,.95), weight_decay=.01)
    gen = torch.Generator().manual_seed(args.seed+1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = {'args': vars(args), 'config': asdict(model.cfg), 'parameters': model.parameter_summary(),
                'data': data_meta, 'torch': str(torch.__version__), 'python': platform.python_version(),
                'cpu': platform.processor(), 'cpu_count': os.cpu_count(),
                'schedule': 'warmup 40 steps, plateau to 80%, linear cooldown to 20% peak',
                'optimizer': 'AdamW beta=(0.9,0.95), weight_decay=0.01, grad_clip=1',
                'auxiliary_coefficient': .01,
                'evaluation': 'fixed windows, seed 333, character cross-entropy in nats',
                'initial_parameter_sha256': initial_parameter_sha256,
                'model_source_sha256': hashlib.sha256(Path('model.py').read_bytes()).hexdigest(),
                'train_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'permuted_lace_layer_zero_based': model.cfg.layers // 2,
                'execution': 'CPU float32, eager PyTorch, unfused sparse gather reference'}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    enabled = not args.no_lace
    initial = evaluate(model,val,batches=4,lace_enabled=enabled)
    print(json.dumps({'parameters': model.parameter_summary(), 'initial_val': initial}), flush=True)
    history = [{'step':0,'tokens':0,'validation_loss':initial}]
    start = time.perf_counter()
    training_seconds = 0
    recent = []
    stream_hash = hashlib.sha256()
    with (out/'history.jsonl').open('w') as log:
        log.write(json.dumps(history[0])+'\n')
        for step in range(1,args.steps+1):
            model.train()
            warmup = min(40, max(1,args.steps//10))
            scale = min(1,step/warmup)
            if step > .8*args.steps:
                scale *= 1-.8*(step-.8*args.steps)/(.2*args.steps)
            for group in optimizer.param_groups: group['lr'] = args.lr*scale
            x,y = batch(train,args.batch,model.cfg.context,gen)
            stream_hash.update(x.numpy().tobytes())
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            result = model(x,y,lace_enabled=enabled)
            loss = result['ce']+.01*result['aux']
            if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss at step {step}')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            training_seconds += time.perf_counter()-tick
            recent.append(float(result['ce'].detach()))
            if step%args.eval_every==0 or step==args.steps:
                validation = evaluate(model,val,batches=4,lace_enabled=enabled)
                row = {'step':step,'tokens':step*args.batch*model.cfg.context,
                       'training_loss_recent':sum(recent)/len(recent),
                       'validation_loss':validation,'lr':args.lr*scale,
                       'grad_norm':float(grad_norm),'training_seconds':training_seconds,
                       'elapsed_seconds':time.perf_counter()-start}
                history.append(row)
                log.write(json.dumps(row)+'\n'); log.flush()
                print(json.dumps(row),flush=True)
                recent=[]
    model.eval()
    validation = evaluate(model,val,batches=16,lace_enabled=enabled)
    train_probe = evaluate(model,train,batches=16,seed=334,lace_enabled=enabled)
    no_lace = evaluate(model,val,batches=16,lace_enabled=False)
    permutation = torch.randperm(model.cfg.groups,generator=torch.Generator().manual_seed(999))
    scrambled = evaluate(model,val,batches=16,lace_enabled=enabled,permutation=permutation)
    metrics = diagnostics(model,val,lace_enabled=enabled)
    prompt = 'ROMEO:\n'
    stoi = {ch:i for i,ch in enumerate(chars)}
    tokens = torch.tensor([[stoi[ch] for ch in prompt]])
    sample = model.generate(tokens,count=240,lace_enabled=enabled)
    text = ''.join(chars[i] for i in sample[0].tolist())
    (out/'sample.txt').write_text(text)
    summary = {'initial_validation_loss_2048_chars':initial,
               'final_validation_loss_8192_chars':validation,
               'final_train_probe_loss_8192_chars':train_probe,
               'frozen_model_no_lace_validation_loss':no_lace,
               'frozen_model_scrambled_middle_layer_lace_validation_loss':scrambled,
               'trained_without_laces':args.no_lace,
               'training_tokens':args.steps*args.batch*model.cfg.context,
               'training_seconds':training_seconds,
               'training_tokens_per_second':args.steps*args.batch*model.cfg.context/training_seconds,
               'total_seconds':time.perf_counter()-start,
               'validation_perplexity':math.exp(validation),
               'peak_process_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
               'training_input_sha256': stream_hash.hexdigest(),
               'notes':'Frozen interventions measure dependence, not a matched trained baseline.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'routing.json').write_text(json.dumps(metrics,indent=2))
    checkpoint = {'config':asdict(model.cfg),'model':model.state_dict(),
                  'optimizer':optimizer.state_dict(),'chars':chars,'step':args.steps,
                  'torch_rng_state':torch.get_rng_state(),'batch_rng_state':gen.get_state(),
                  'protocol':protocol}
    torch.save(checkpoint,out/'checkpoint.pt')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
