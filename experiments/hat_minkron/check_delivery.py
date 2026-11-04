"""Post-training checks on the exact delivered model and carried state."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from events import ByteBatch
from model import HATConfig,HATSHoE,StreamState
from verify import same_state


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parent
    saved=torch.load(root/'results/language/checkpoint.pt',weights_only=False,map_location='cpu')
    model=HATSHoE(HATConfig(**saved['config'])).eval();model.load_state_dict(saved['model'])
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    data=(root/'data/tiny_shakespeare.txt').read_bytes()
    x=torch.tensor(list(data[int(.9*len(data)):int(.9*len(data))+513]),dtype=torch.long)[None]
    links=(torch.arange(513)//17%3)[None];slots=torch.where(links==0,x,torch.where(links==1,0,7))
    events=ByteBatch(x,links,slots)
    before={n:v.clone() for n,v in model.named_buffers()}
    full=model(events);parts=[];state=None
    for a,b in [(0,7),(7,129),(129,130),(130,273),(273,513)]:
        result=model(events.slice(a,b),state=state);state=result['state'];parts.append(result['logits'])
    merged=torch.cat(parts,1)
    error=float((full['logits']-merged).abs().max())
    torch.testing.assert_close(full['logits'],merged,atol=1e-4,rtol=1e-4)
    same_state(full['state'],state,atol=1e-4,rtol=1e-4)
    changed=ByteBatch(x.clone(),links,slots.clone());changed.values[:,201:]=(changed.values[:,201:]+47)%256
    changed.slots.copy_(torch.where(links==0,changed.values,slots))
    altered=model(changed)
    torch.testing.assert_close(full['logits'][:,:201],altered['logits'][:,:201],atol=1e-6,rtol=1e-6)
    prefix=model(events.slice(0,101));raw=prefix['state'].detach().to_dict()
    restored=HATSHoE(HATConfig(**saved['config'])).eval();restored.load_state_dict(saved['model'])
    continuation=restored(events.slice(101),state=StreamState.from_dict(raw))
    reference=model(events.slice(101),state=prefix['state'])
    torch.testing.assert_close(continuation['logits'],reference['logits'],rtol=0,atol=0)
    for n,v in model.named_buffers(): assert torch.equal(v,before[n]),n
    sample,_=model.generate(ByteBatch.text('ROMEO:\n'),count=32,seed=123)
    expected=(root/'results/language/sample.txt').read_text()
    assert expected.startswith(bytes(sample[0].tolist()).decode('utf-8',errors='replace'))
    one=ByteBatch.static(x[:,:32]);a=model(one);size=a['state'].storage_bytes()
    for _ in range(8): a=model(one,state=a['state'])
    assert a['state'].storage_bytes()==size==8208
    all_links=torch.arange(8)[None]
    eight=ByteBatch(torch.zeros(1,8,dtype=torch.long),all_links,torch.zeros_like(all_links))
    r=model(eight);assert r['state'].storage_bytes()==22600
    for layer in range(3):
        assert model.children_blocks[0].blocks[layer].experts.shared is model.children_blocks[1].blocks[layer].experts.shared
    # The sample checkpoint includes runtime state, separate from model weights.
    carried=StreamState.from_dict(saved['sample_stream_state'])
    model(ByteBatch.text('X'),state=carried)
    checks={'finite_weights':True,'trained_streaming_events':513,'trained_streaming_max_logit_error':error,
        'trained_causality':'PASS','checkpoint_continuation':'PASS','generation_prefix_reproduces':True,
        'inference_buffers_unchanged':True,'shared_parameter_aliases':True,
        'one_link_complete_state_bytes':size,'eight_link_complete_state_bytes':22600,
        'no_kv_cache':not any('attention' in n.lower() for n,_ in model.named_modules()),
        'parameter_summary':model.parameter_summary()}
    (root/'results/delivery_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
    provenance=json.loads((root/'PROVENANCE.json').read_text())
    provenance['final_code_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}
    provenance['post_training_changes']='Input validation tightened to enforce immutable static byte addresses and reject implicit float-to-byte casting; valid training computation unchanged. Added tests, documentation and delivery checks.'
    (root/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2))


if __name__=='__main__': main()
