"""Paired release timings and gradient errors for regional execution changes."""
import argparse
import gc
import importlib
import json
from pathlib import Path
import statistics
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions import byte_stream
from experiments.text_regions.native_runtime import load_native


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sizes', nargs='+', default=['tiny', 'default', '25m'])
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--context', type=int, default=8)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--generate', type=int, default=16)
    p.add_argument('--no-stage-batching', action='store_true')
    p.add_argument('--profile', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(1)
    native = load_native()  # Exclude optional one-time build from steady-state timings.
    package = types.ModuleType('_previous')
    package.__path__ = [str(a.baseline_root.resolve())]
    sys.modules[package.__name__] = package
    old = importlib.import_module('_previous.experiments.text_regions.model')
    old_cfg = importlib.import_module('_previous.experiments.text_regions.config').GraphConfig
    old_stream = importlib.import_module('_previous.experiments.text_regions.byte_stream')
    output = dict(torch=torch.__version__, cpu=torch.backends.cpu.get_cpu_capability(), threads=1, native_kernel=native is not None,
                  protocol={**vars(a), 'baseline_root':str(a.baseline_root), 'out':str(a.out)}, measurements=[])
    for size in a.sizes:
        cfg = GraphConfig.tiny(rounds=2, recurrent_feedback=True) if size == 'tiny' else GraphConfig(rounds=2, recurrent_feedback=True)
        if size == '25m':
            previous_results = json.loads((ROOT/'experiments/text_regions/results/operator_runtime/measurements.json').read_text())
            cfg = GraphConfig.from_dict(next(row['config'] for row in previous_results['measurements'] if row['size'] == '25m'))
        torch.manual_seed(381)
        current = RegionLanguageModel(cfg)
        if a.no_stage_batching:
            current.batch_independent = False
        previous = old.RegionLanguageModel(old_cfg.from_dict(cfg.to_dict()))
        previous.load_state_dict(current.state_dict())
        generator = torch.Generator().manual_seed(918)
        x = torch.randint(0, 256, (a.batch, a.context), generator=generator)
        models = dict(previous=previous, current=current)
        record = dict(size=size, parameters=current.parameter_summary(), config=cfg.to_dict(), workloads={})
        for workload in ('train', 'inference', 'generation'):
            timings = {key: [] for key in models}
            checks = {}
            def run(model):
                interface = old_stream if model is previous else byte_stream
                if workload == 'generation':
                    if cfg.independent_sources:
                        session = interface.start(model, 'R', seed=17)
                        return interface.generate(model, session, a.generate)[0]
                    return model.generate(x[:1, :1], count=a.generate, seed=17)[0]
                with torch.set_grad_enabled(workload == 'train'):
                    target = x if workload == 'train' else None
                    result = interface.forward(model, x, target) if cfg.independent_sources else model(x, target)
                if workload == 'train':
                    result['loss'].backward()
                return result['logits'].detach()
            for key, model in models.items():
                model.zero_grad(set_to_none=True)
                run(model)
            for repeat in range(a.repeats):
                for key in (('previous', 'current') if repeat % 2 == 0 else ('current', 'previous')):
                    model = models[key]
                    model.zero_grad(set_to_none=True)
                    start = time.perf_counter()
                    result = run(model)
                    timings[key].append(time.perf_counter()-start)
                    if repeat == 0:
                        checks[key] = result
            difference = (checks['current']-checks['previous']).double()
            row = dict(seconds=timings, speedup=statistics.median(timings['previous'])/statistics.median(timings['current']),
                       max_output_error=float(difference.abs().max()), output_rms_error=float(difference.square().mean().sqrt()))
            if workload == 'train':
                errors = []
                missing = []
                for (name, left), (_, right) in zip(previous.named_parameters(), current.named_parameters()):
                    if (left.grad is None) != (right.grad is None):
                        missing.append(name)
                    elif left.grad is not None:
                        delta = (left.grad.double()-right.grad.double()).norm()
                        rel = delta/left.grad.double().norm().clamp_min(1e-30)
                        errors.append((float(rel), name, float(delta)))
                row['gradient_errors_largest'] = sorted(errors, reverse=True)[:10]
                row['gradient_absence_mismatches'] = missing
                assert not missing, missing
            record['workloads'][workload] = row
            print(json.dumps(dict(size=size, workload=workload, speedup=row['speedup'], max_error=row['max_output_error'])), flush=True)
        if a.profile:
            record['backward_allocations'] = {}
            for name, model in models.items():
                model.zero_grad(set_to_none=True)
                interface = old_stream if model is previous else byte_stream
                result = interface.forward(model, x[:, :1], x[:, :1]) if cfg.independent_sources else model(x[:, :1], x[:, :1])
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
                    result['loss'].backward()
                record['backward_allocations'][name] = {
                    event.key:dict(calls=event.count, allocated_bytes=event.cpu_memory_usage)
                    for event in prof.key_averages() if any(s in event.key for s in ('index_select_backward','to_dense','dense_rows','DenseBankGradientBackward'))}
                sizes = {p.numel()*p.element_size() for key, p in model.named_parameters()
                         if key.endswith(('.up', '.down')) and p.ndim == 3}
                banks = [event for event in prof.events() if
                         (event.name == 'aten::index_select_backward' or event.name.endswith('::dense_rows')
                          or event.name == 'aten::to_dense') and event.cpu_memory_usage in sizes]
                record['backward_allocations'][name]['expert_dense_bank_buffers'] = dict(
                    calls=len(banks), allocated_bytes=sum(event.cpu_memory_usage for event in banks))
                del result
        output['measurements'].append(record)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(output, indent=2)+'\n')
        del models, current, previous
        gc.collect()


if __name__ == '__main__':
    main()
