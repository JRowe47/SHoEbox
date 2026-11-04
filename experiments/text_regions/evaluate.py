"""Common byte-unit engineering measurements for the preserved baseline and graph."""
import argparse
import json
import math
from pathlib import Path
import platform
import resource
import time
import torch
from ..hat_minkron.model import HATConfig, HATSHoE, StreamingRouter
from ..hat_minkron.hierarchy_reference import SplitExpertBank
from ..hat_minkron.semantic_core import LaceController
from ..hat_minkron.train import optimizer_for
from .config import BASELINE_REVISION, GraphConfig
from .model import RegionLanguageModel
from .train import optimizer_bytes


class ComponentTimer:
    """Inclusive router+lace time and split-expert time; no nested bank counting."""
    def __init__(self, model):
        self.seconds = {'routing': 0., 'experts': 0.}
        self.hooks = []
        for module in model.modules():
            kind = ('experts' if isinstance(module, SplitExpertBank) else
                    'routing' if isinstance(module, (StreamingRouter, LaceController)) else None)
            if kind is None:
                continue
            starts = []
            def before(module, inputs, stack=starts):
                stack.append(time.perf_counter())
            def after(module, inputs, output, stack=starts, category=kind):
                self.seconds[category] += time.perf_counter()-stack.pop()
            self.hooks.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))

    def close(self):
        for hook in self.hooks:
            hook.remove()


def unique_active_experts(model, result, baseline):
    selected = set()
    if baseline:
        for execution in result['executions']:
            owner = str(execution['child'])
            for depth, record in enumerate(execution['trace']):
                shared = model.shared_counts[depth]
                for expert in record['experts'].unique().tolist():
                    selected.add((depth, 'shared' if expert < shared else owner, expert))
    else:
        shared = model.cfg.shared_per_block
        for record in result['trace']:
            if record['routing'] is None:
                continue
            for depth in range(3):
                for expert in record['routing']['experts'][:, depth].unique().tolist():
                    selected.add((depth, 'shared' if expert < shared else record['node'], expert))
    return len(selected)


def measure(baseline=False, rounds=1, batch=4, context=128, threads=2, seed=20261104, iterations=1):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if baseline:
        model = HATSHoE(HATConfig(context=max(16, context), seed=seed))
    else:
        model = RegionLanguageModel(GraphConfig(context=context, rounds=rounds,
                                               recurrent_feedback=rounds > 1, seed=seed))
    optimizer = optimizer_for(model, .002)
    x = torch.randint(0, 256, (batch, context))
    y = torch.roll(x, shifts=-1, dims=1)
    # Random bytes exercise execution and gradients; loss is not a quality score.
    model.eval()
    with torch.no_grad():
        model(x[:, :2])
        tick = time.perf_counter()
        for _ in range(iterations):
            output = model(x)
        inference_seconds = (time.perf_counter()-tick)/iterations
        state_bytes = output['state'].storage_bytes()
        timer = ComponentTimer(model)
        tick = time.perf_counter()
        traced = model(x, trace=True)
        instrumented_seconds = time.perf_counter()-tick
        timer.close()
        unique = unique_active_experts(model, traced, baseline)
        generated, _ = model.generate(x[:1, :2], count=8)
    model.train()
    tick = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        output = model(x, y)
        output['loss'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError('Nonfinite benchmark gradients.')
        optimizer.step()
        if baseline:
            model.commit_feedback(output['packet'])
        detached = output['state'].detach()
        del output
    training_seconds = (time.perf_counter()-tick)/iterations
    summary = model.parameter_summary()
    experts_per_byte = 48 if baseline else summary['expert_calls_per_round_per_external_event']*rounds
    return {'schema_version': 1, 'model': 'hat_minkron_baseline' if baseline else 'text_regions',
            'baseline_revision': BASELINE_REVISION, 'config': (vars(model.cfg) if baseline else model.cfg.to_dict()),
            'seed': seed, 'threads': threads, 'batch': batch, 'context': context, 'iterations': iterations,
            'python': platform.python_version(), 'torch': str(torch.__version__),
            'unique_parameters': summary['total_unique_parameters'], 'parameter_details': summary,
            'external_bytes_per_iteration': batch*context, 'internal_rounds_per_byte': 1 if baseline else rounds,
            'expert_calls_per_byte': experts_per_byte, 'expert_executions_per_iteration': batch*context*experts_per_byte,
            'unique_experts_in_inference_window': unique,
            'unique_expert_matrix_weights_in_inference_window': unique*2*model.cfg.width*model.cfg.expert_hidden,
            'inference_seconds': inference_seconds, 'training_step_seconds': training_seconds,
            'unclipped_gradient_norm': float(norm),
            'inference_bytes_per_second': batch*context/inference_seconds,
            'training_bytes_per_second': batch*context/training_seconds,
            'routing_forward_seconds_instrumented': timer.seconds['routing'],
            'expert_forward_seconds_instrumented': timer.seconds['experts'],
            'instrumented_inference_seconds': instrumented_seconds,
            'state_bytes': state_bytes, 'detached_training_state_bytes': detached.storage_bytes(),
            'optimizer_state_bytes': optimizer_bytes(optimizer),
            'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'sample_byte_count': generated.numel(),
            'measurement': 'CPU wall time; random byte plumbing fixture; fresh state per timed iteration; '
                           'separate inference with hooks measures router/lace and expert forward components only'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--context', type=int, default=128)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--iterations', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if min(args.rounds, args.batch, args.context, args.threads, args.iterations) < 1:
        p.error('Positive counts required.')
    result = measure(args.baseline, args.rounds, args.batch, args.context, args.threads, iterations=args.iterations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
