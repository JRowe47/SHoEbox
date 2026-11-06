"""Interleaved CPU timings for the reference and optimized regional runtime."""
import argparse
import copy
import cProfile
from dataclasses import replace
import gc
import hashlib
import importlib
import importlib.machinery
import json
from pathlib import Path
import platform
import pstats
import resource
import statistics
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.text_regions import byte_stream, gradient_balance
from experiments.text_regions.config import GraphConfig
from experiments.text_regions.model import RegionLanguageModel
from experiments.text_regions.joint_text import optimizer_for
from experiments.text_regions.data import OrderedByteStreams
from experiments.text_regions.multifact import table_batches
from experiments.text_regions.multifact_data import make_batch
from experiments.text_regions.multifact_eval import evaluate


def digest(tensors):
    h = hashlib.sha256()
    for x in tensors:
        h.update(x.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sizes', nargs='+', choices=('checkpoint', 'default', '25m'), default=['checkpoint'])
    p.add_argument('--workloads', nargs='+', choices=('text_train', 'text_inference', 'joint_update', 'memory_eval'),
                   default=['text_train', 'text_inference', 'joint_update', 'memory_eval'])
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--context', type=int, default=16)
    p.add_argument('--memory-count', type=int, default=128)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--baseline-root', type=Path, help='Untouched source checkout to measure as reference.')
    a = p.parse_args()
    if min(a.threads, a.repeats, a.batch, a.context, a.memory_count) < 1:
        p.error('Counts must be positive.')
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    saved = torch.load(a.checkpoint, weights_only=True, map_location='cpu')
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       machine=platform.machine(), cpu_capability=torch.backends.cpu.get_cpu_capability(),
                       threads=torch.get_num_threads(), torch_build=torch.__config__.show())
    result = dict(environment=environment, checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                  protocol=vars(a) | {'checkpoint': str(a.checkpoint), 'out': str(a.out),
                                     'baseline_root': str(a.baseline_root) if a.baseline_root else None}, measurements=[])
    legacy = {}
    if a.baseline_root:
        # Load the original tree under an independent package namespace. This
        # keeps paired timings in one process without adding new dispatch to
        # the old engine, or changing either source checkout.
        package = types.ModuleType('_shoebox_reference')
        package.__path__ = [str(a.baseline_root.resolve())]
        package.__spec__ = importlib.machinery.ModuleSpec(package.__name__, None, is_package=True)
        sys.modules[package.__name__] = package
        for name in ('model', 'config', 'byte_stream', 'gradient_balance', 'joint_text', 'multifact_eval'):
            legacy[name] = importlib.import_module('_shoebox_reference.experiments.text_regions.'+name)
    # Local deterministic bytes: no network or benchmark data dependency.
    raw = (b'ROMEO: What light through yonder window breaks?\nJULIET: It is the east.\n')*2048
    chunk = OrderedByteStreams.from_bytes(raw, a.batch).next_chunk(a.context)
    for size in a.sizes:
        cfg = GraphConfig.from_dict(saved['config'])
        if size == 'default':
            cfg = GraphConfig(rounds=2, recurrent_feedback=True)
        elif size == '25m':
            cfg = replace(cfg, width=128, groups=16, experts_per_block=864, shared_per_block=432,
                          expert_hidden=4, address_dim=16, laces=64, hat_symbols=64,
                          kron_heads=4, kron_key=8, kron_value=8)
        torch.manual_seed(20261105)
        model = RegionLanguageModel(cfg)
        if size == 'checkpoint':
            model.load_state_dict(saved['weights'])
        initial = copy.deepcopy(model.state_dict())
        opt = optimizer_for(model)
        if size == 'checkpoint':
            opt.load_state_dict(saved['optimizer'])
        initial_opt = copy.deepcopy(opt.state_dict())
        models, optimizers = dict(reference=model, optimized=model), dict(reference=opt, optimized=opt)
        if legacy:
            original = legacy['model'].RegionLanguageModel(legacy['config'].GraphConfig.from_dict(cfg.to_dict()))
            original.load_state_dict(initial)
            original_opt = legacy['joint_text'].optimizer_for(original)
            original_opt.load_state_dict(copy.deepcopy(initial_opt))
            models['reference'], optimizers['reference'] = original, original_opt
            del original, original_opt
        legacy_mode = False
        memory = make_batch('train', 0, 16, 20261106, 16, 'shared_left')
        tables = table_batches(memory, 'train', 0, 20261106)

        def text(targets=None):
            if cfg.independent_sources:
                interface = legacy['byte_stream'] if legacy_mode else byte_stream
                return interface.forward(model, chunk.values, targets)
            return model(chunk.values, targets, rounds=2, feedback=True)

        for workload in a.workloads:
            if size != 'checkpoint' and workload not in ('text_train', 'text_inference'):
                continue
            for instance in models.values():
                instance.load_state_dict(initial)
            record = dict(size=size, workload=workload, config=cfg.to_dict(), parameters=model.parameter_summary(),
                          batch=a.batch, context=a.context, timings={}, checks={})
            values = {mode: [] for mode in ('reference', 'optimized')}
            checks = {}
            def prepare(mode):
                nonlocal model, opt, legacy_mode
                model, opt = models[mode], optimizers[mode]
                legacy_mode = bool(legacy and mode == 'reference')
                if hasattr(model, 'set_execution_backend'):
                    model.set_execution_backend(mode)
                model.zero_grad(set_to_none=True)
                if workload == 'joint_update':
                    model.load_state_dict(initial)
                    opt.load_state_dict(copy.deepcopy(initial_opt))
                model.train(workload in ('text_train', 'joint_update'))

            def run():
                if workload == 'text_train':
                    r = text(chunk.targets)
                    r['loss'].backward()
                    return r
                if workload == 'text_inference':
                    with torch.no_grad():
                        return text()
                if workload == 'joint_update':
                    balance = legacy['gradient_balance'] if legacy_mode else gradient_balance
                    state, row = balance.update(model, opt, chunk, None, memory, tables, saved['reference'])
                    return dict(state=state, metrics=row)
                with torch.no_grad():
                    evaluator = legacy['multifact_eval'].evaluate if legacy_mode else evaluate
                    return evaluator(model, 'test', 20261106, a.memory_count, 16, 'shared_left',
                                    batch_size=a.memory_count)

            # One untimed warmup per backend. Alternate order on measured pairs.
            for mode in values:
                prepare(mode)
                r = run()
                del r
            for repeat in range(a.repeats):
                for mode in (('reference', 'optimized') if repeat % 2 == 0 else ('optimized', 'reference')):
                    prepare(mode)
                    start = time.perf_counter()
                    r = run()
                    elapsed = time.perf_counter()-start
                    values[mode].append(elapsed)
                    if repeat == 0:
                        if 'logits' in r:
                            checks[mode] = dict(logits=digest([r['logits']]), state=digest(r['state'].tensors()),
                                                expert_executions=r['expert_executions'])
                        elif 'metrics' in r:
                            checks[mode] = {k: r['metrics'][k] for k in ('text_nats', 'text_bytes', 'expert_executions')}
                        else:
                            checks[mode] = dict(fields=r['fields'])
                    print(json.dumps(dict(size=size, workload=workload, backend=mode, repeat=repeat,
                                          seconds=elapsed)), flush=True)
                    del r
            record['timings'] = {k: dict(seconds=v, median_seconds=statistics.median(v),
                                         min_seconds=min(v), max_seconds=max(v)) for k, v in values.items()}
            record['speedup'] = statistics.median(values['reference'])/statistics.median(values['optimized'])
            record['checks'] = checks
            record['peak_process_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            if a.profile and workload == 'text_train':
                for mode in values:
                    prepare(mode)
                    profiler = cProfile.Profile()
                    profiler.enable(); r = run(); profiler.disable()
                    path = a.out/f'{size}_{mode}'
                    profiler.dump_stats(str(path)+'.prof')
                    with Path(str(path)+'.txt').open('w') as stream:
                        stats = pstats.Stats(profiler, stream=stream)
                        stats.strip_dirs().sort_stats('cumtime').print_stats(50)
                        record.setdefault('python_calls', {})[mode] = stats.total_calls
                    del r
            result['measurements'].append(record)
            (a.out/'measurements.json').write_text(json.dumps(result, indent=2)+'\n')
        del model, opt, models, optimizers, initial, initial_opt
        gc.collect()


if __name__ == '__main__':
    main()
