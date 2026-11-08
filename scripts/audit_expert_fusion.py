"""Check the completed training endpoint and bounded expert gradient storage."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.hat_minkron.semantic_core import ExpertBank
from shoebox import cortical_wide
from shoebox.checkpoint import load
from shoebox.fused_experts import expert_forward, GRADIENT_GROUP_CALLS
from shoebox.fused_kron import native_status
from shoebox.profile_wide import probe
from shoebox.train_fineweb import atomic_json, source_digest
from shoebox.transport import map_tensors


def bank_allocation_audit():
    result = {}
    saved = {}
    for mode in ('baseline', 'indexed'):
        torch.manual_seed(906)
        cfg = cortical_wide().hat_config().child_config()
        bank = ExpertBank(cfg, 32)
        x = torch.randn(4, 1, cfg.width, requires_grad=True)
        ids = torch.randint(0, 32, (4, 1, 4))
        gates = torch.randn(4, 1, 4, requires_grad=True)
        cache = {}
        def cached(module, name, compute):
            if name not in cache:
                cache[name] = compute()
            return cache[name]
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                     profile_memory=True, record_shapes=True) as prof:
            outputs = [(expert_forward(bank, x+i*.001, ids, gates, cached) if mode == 'indexed'
                        else bank(x+i*.001, ids, gates)) for i in range(64)]
            loss = torch.stack(outputs).square().mean()
            loss.backward()
        saved[mode] = [loss.detach(), x.grad, gates.grad, bank.up.grad, bank.down.grad]
        counts = defaultdict(int)
        allocated = 0
        for row in prof.key_averages(group_by_input_shape=True):
            counts[row.key] += row.count
            allocated += max(0, row.self_cpu_memory_usage)
        boundary = sum(n for key, n in counts.items() if key.endswith('::dense_row_grads'))
        result[mode] = {'positive_allocation_bytes':allocated,
                        'dense_bank_gradient_allocations':counts['IndexBackward0'] if mode == 'baseline' else boundary,
                        'index_backward_calls':counts['IndexBackward0'],
                        'coalesce_calls':counts['aten::_coalesce'],
                        'native_dense_group_calls':boundary}
    for a, b in zip(saved['baseline'], saved['indexed']):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    assert result['baseline']['dense_bank_gradient_allocations'] == 128
    assert result['indexed']['dense_bank_gradient_allocations'] == 128//GRADIENT_GROUP_CALLS
    return {'scope':'one 32-expert bank, B4/K4/H16/D64, 64 sequential calls, forward and backward', **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = args.root/'train'
    training = json.loads((run/'results.json').read_text())
    matched = json.loads((args.root/'matched/summary.json').read_text())
    tests = (args.root/'final_checks.log').read_text()
    assert tests.rstrip().endswith('OK') and 'skipped' not in tests.lower()
    count = int(re.search(r'Ran (\d+) tests', tests).group(1))
    assert count == 28
    model, state, _, checkpoint = load(run/'last.pt')
    model.set_execution_backend('fused')
    assert native_status()['available'] and training['native']['available']
    assert training['complete'] and training['trained_targets'] == 1000 and training['updates'] == 8
    assert training['expert_kernel']['requested']
    assert training['source_sha256'] == matched['source_sha256'] == source_digest()
    assert checkpoint['metadata']['protocol']['source_sha256'] == source_digest()
    assert checkpoint['step'] == 8 and state.event == state.memory.writes == 250
    assert state.batch == 4 and state.storage_bytes() == training['final_state_bytes']
    raw = (run/'profile_train.bin').read_bytes()
    assert len(raw) == 1004 and hashlib.sha256(raw).hexdigest() == training['data_sha256'] == matched['data_sha256']
    def finite(value):
        assert torch.isfinite(value).all()
        return value
    map_tensors(checkpoint, finite)
    data = torch.tensor(list(raw), dtype=torch.long).reshape(4, -1)
    final = probe(model, data[:, :33])
    assert final['nll'] == training['final_probe']['nll'] and final['ports'] == training['final_probe']['ports']
    with torch.no_grad():
        continued = model(data[:, :1], state=state)
    assert continued['state'].event == 251 and torch.isfinite(continued['logits']).all()
    for filename, key in [('history.jsonl', 'history'), ('ports.jsonl', 'port_history')]:
        assert [json.loads(line) for line in (run/filename).read_text().splitlines()] == training[key]
    for name in ('A1', 'B1', 'L2'):
        rows = [r for r in training['temporal_gradients'] if r['region'] == name and r['lag'] == 31]
        assert rows and all(r['finite'] and r['gradient_l1'] > 0 for r in rows)
    summary = {'complete': True, 'tests_passed': count, 'endpoint_reload_exact': True,
               'source_sha256': source_digest(), 'data_sha256': training['data_sha256'],
               'parameters': training['parameters'], 'trained_targets': 1000, 'updates': 8,
               'initial_probe_nll': training['initial_probe']['nll'],
               'final_probe_nll': training['final_probe']['nll'],
               'training_seconds': training['training_seconds'],
               'training_bytes_per_second': training['training_bytes_per_second'],
               'training_peak_rss_mib': training['peak_rss_mib'],
               'state_bytes': state.storage_bytes(), 'lag_31_task_gradients_verified': ['A1', 'B1', 'L2'],
               'matched': matched, 'bank_allocation':bank_allocation_audit()}
    for mode in ('baseline', 'indexed'):
        operator = json.loads((run/f'operators_{mode}.json').read_text())
        totals = defaultdict(lambda: {'calls':0, 'self_seconds':0., 'positive_allocation_bytes':0})
        for row in operator['operators']:
            name = row['operator']
            if name.startswith('shoebox_minkron_'):
                name = 'native::'+name.split('::')[-1]
            out = totals[name]
            out['calls'] += row['calls']
            out['self_seconds'] += row['self_cpu_seconds']
            out['positive_allocation_bytes'] += max(0, row['self_cpu_memory_bytes'])
        summary[f'{mode}_operators'] = {
            'scope':operator['scope'], 'total_self_seconds':operator['total_self_cpu_seconds'],
            'positive_allocation_bytes':operator['positive_self_allocated_bytes'],
            'top_self': sorted([{'operator':k, **v} for k, v in totals.items()],
                               key=lambda r:r['self_seconds'], reverse=True)[:15]}
    atomic_json(args.root/'summary.json', summary)
    atomic_json(ROOT/'docs/cortical_expert_fusion_results.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('matched', 'baseline_operators', 'indexed_operators')}), flush=True)


if __name__ == '__main__':
    main()
