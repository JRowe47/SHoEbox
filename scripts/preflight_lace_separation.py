"""Frozen endpoint parity, task-credit observability and matched timing probes."""
import hashlib,json,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_lace_separation import LaceRun
from scripts.lace_cohorts import from_record
from scripts.run_memory_selectivity import write_json
from scripts.evaluate_lace_separation import load_model
from experiments.text_regions.lace_separation import LaceCredit
from experiments.text_regions.paired_recall import paired_forward,fork_state
from experiments.text_regions.task_memory_train import byte_loss
from experiments.text_regions.activation_checkpoint import enable_associative_checkpointing


def probe(run,batch,instrument):
    model=run.model;model.train();model.zero_grad(set_to_none=True)
    capture=LaceCredit(batch['history'].shape[1]+batch['query'].shape[1]-1,run.cfg.lace_separation) if instrument else None
    model._lace_capture=capture;started=time.perf_counter()
    result=paired_forward(model,batch);logits=result['logits'][:,-1]
    if capture:capture.supervise(logits,batch['answers'])
    loss=F.cross_entropy(logits,batch['answers'])+.1*byte_loss(result['logits'],batch['syntax'])+.01*result['aux']+.001*result['hat_aux']
    loss.backward();elapsed=time.perf_counter()-started
    records={}
    if capture:
        for (owner,round_),r in capture.records.items():
            z,h,b,v=r['cache']
            records[f'{owner}:{round_}']=dict(input_delta_rms=float((z[::2]-z[1::2]).square().mean().sqrt()),
                input_equal=bool(torch.equal(z[::2],z[1::2])),gradient_norm=float(r['gradient'].norm()))
    mechanism=capture.apply() if capture else None
    model._lace_capture=None
    return logits.detach().clone(),dict(forward_backward_seconds=elapsed,mechanism=mechanism,decisions=records)


def main():
    torch.set_num_threads(1)
    path=ROOT/'artifacts/hippocampus_binding/ca1/bounded/last.pt'
    parent=torch.load(path,weights_only=True,map_location='cpu');raw=(ROOT/'data/tiny_shakespeare.txt').read_bytes()
    lock=json.loads((ROOT/'artifacts/lace_separation/cohorts.json').read_text())
    batches=[from_record(r) for r in lock['cohorts']['diagnostic']['records'] if r['template']==0]
    out=dict(parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),timing={},diagnostics={},parity={})
    baseline,_=load_model(path)
    with torch.no_grad():expected=paired_forward(baseline,batches[0])['logits'][:,-1]
    for label,arm,instrument in [('bypass','A',False),('head','B',False),('instrumented','B',True),('credit','D',True)]:
        run=LaceRun(parent,raw,arm);enable_associative_checkpointing(run.model,segment=16,min_positions=512)
        logits,metrics=probe(run,batches[0],instrument)
        # Eval and training kernels can differ by tiny FP rounding; compare heads
        # under identical training arithmetic below, and report both separately.
        out['parity'][label]=float((logits-expected).abs().max())
        if label=='bypass':training_reference=logits
        else:assert torch.equal(logits,training_reference),f'Zero head mismatch: {label}'
        out['timing'][label]=metrics
        print(json.dumps(dict(stage=label,seconds=metrics['forward_backward_seconds'])),flush=True)
    for batch,gap in zip(batches,(0,32)):
        run=LaceRun(parent,raw,'D');enable_associative_checkpointing(run.model,segment=16,min_positions=512)
        _,metrics=probe(run,batch,True);out['diagnostics'][str(gap)]=metrics
        assert any(not r['input_equal'] for r in metrics['decisions'].values()),'Question inputs exactly identical'
    run=LaceRun(parent,raw,'B');run.model.eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(811)
        for node in run.model.nodes.values():torch.nn.init.normal_(node.lace_head.V,std=.2)
    batch=batches[0]
    with torch.no_grad():
        prefix=run.model(batch['history']);state=fork_state(prefix['state'],batch['rows'])
        normal=run.model(batch['query'],state=state)['logits'][:,-1]
        run.model._lace_cue_rows=torch.arange(len(batch['query']))^1
        swapped=run.model(batch['query'],state=state)['logits'][:,-1]
        run.model._lace_cue_rows=None
    out['synthetic_head_cue_sensitivity']=dict(max_logit_difference=float((normal-swapped).abs().max()),
        rms_logit_difference=float((normal-swapped).square().mean().sqrt()),
        note='Frozen parent; diagnostic deterministic nonzero V, not a trained result.')
    out['parameters']=sum(p.numel() for p in run.model.parameters())
    out['head_parameters']=sum(p.numel() for n,p in run.model.named_parameters() if '.lace_head.' in n)
    write_json(ROOT/'artifacts/lace_separation/preflight.json',out)

if __name__=='__main__':main()
