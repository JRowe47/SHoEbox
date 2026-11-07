"""Inspect trained regional selections and verify exact stream continuation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.text_regions.data import OrderedByteStreams, split_bytes
from experiments.text_regions.ports import tensor_list
from experiments.text_regions.state import GraphState
from experiments.text_regions.train import load_checkpoint


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'artifacts/regional_recall/graph')
    parser.add_argument('--data', type=Path, default=ROOT/'data/tiny_shakespeare.txt')
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, _, state, saved = load_checkpoint(args.run/'checkpoint.pt', restore_rng=False)
    model.eval()
    train, _, test = split_bytes(args.data.read_bytes())
    streams = OrderedByteStreams.from_bytes(train, saved['protocol']['batch'])
    streams.load_state_dict(saved['streams'])
    chunk = streams.next_chunk(16)
    full = model(chunk.values, state=state, reset=chunk.reset)
    first = model(chunk.values[:, :7], state=state, reset=chunk.reset[:, :7])
    restored = GraphState.from_dict(first['state'].to_dict())
    second = model(chunk.values[:, 7:], state=restored, reset=chunk.reset[:, 7:])
    joined = torch.cat((first['logits'], second['logits']), 1)
    torch.testing.assert_close(full['logits'], joined, rtol=0, atol=0)
    for a, b in zip(tensor_list(full['state']), tensor_list(second['state'])):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    heldout = OrderedByteStreams.from_bytes(test, 4).next_chunk(32)
    result = model(heldout.values, heldout.targets, reset=heldout.reset, return_recall_activity=True)
    nodes = {}
    for name, record in result['recall_activity'].items():
        weights = record['weights'][..., :-1]
        assert not bool((weights * ~record['valid'][..., None, :]).any())
        workspace = model.connections.recall[name].workspace
        maps = workspace.payload_merge.weight.reshape(model.cfg.width, model.cfg.recall_slots, model.cfg.width)
        nodes[name] = dict(mean_active_slots=float(weights.sum(-1).sum(-1).mean()),
            mean_unique_ports=float(weights.sum(-2).clamp(max=1).sum(-1).mean()),
            edges=result['recall_edges'][name],
            query_feedback_norm=float(workspace.query_feedback.weight.norm()),
            null_feedback_norm=float(workspace.null_feedback.weight.norm()),
            role_map_deviation=float((maps-maps.mean(1, keepdim=True)).norm()))
    controls = {control: float(model(heldout.values, heldout.targets, reset=heldout.reset,
                                     recall_control=control)['ce'])
                for control in ('pooled', 'no_feedback', 'blocked')}
    audit = dict(parameters=model.parameter_summary(), split='heldout test',
        evaluated_targets=heldout.targets.numel(), nll=float(result['ce']),
        controls=controls, nodes=nodes, exact_chunk_resume=True,
        max_logit_drift=float((full['logits']-joined).abs().max()),
        feedback_bytes_per_stream=sum(t.numel()*t.element_size()
            for t in result['state'].recall_feedback.values())//4)
    (args.run/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(dict(exact_chunk_resume=True, nodes=len(nodes), test_nll=audit['nll'])))


if __name__ == '__main__':
    main()
