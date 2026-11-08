"""Check a completed long CPU run and compare its port diagnostics with initialization."""
import argparse
import json
import math
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from shoebox.checkpoint import load
from shoebox.cpu_run import stream
from shoebox.train_cpu import diagnostics, file_sha
from shoebox.train_fineweb import atomic_json, evaluate, source_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():p.error('Use a new audit output file.')
    result=json.loads((args.run/'results.json').read_text());protocol=result['protocol']
    assert result['complete'] and result['trained_targets']==protocol['targets']
    assert result['checkpoint_trained_targets']==protocol['targets']
    assert protocol['source_sha256']==source_digest()
    torch.set_num_threads(protocol['threads'])
    validation,checksum=stream(args.validation,protocol['validation_targets'],protocol['batch'])
    assert checksum==protocol['validation_sha256']
    model,state,_,checkpoint=load(args.run/'last.pt')
    model.set_execution_backend('cpu').set_cpu_workers(protocol['workers'])
    assert state.event*state.batch==protocol['targets']
    assert checkpoint['metadata']['summary']['trained_targets']==protocol['targets']
    assert checkpoint['step']==result['updates']
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    aligned_counts=torch.zeros(256,dtype=torch.long)
    def count_predictions(module,inputs,output):
        aligned_counts.add_(torch.bincount(output['logits'].argmax(-1).flatten(),minlength=256))
    handle=model.register_forward_hook(count_predictions)
    actual=evaluate(model,validation,context=protocol['context'])
    handle.remove()
    expected=result['final_validation']
    for name in ('nll','bits_per_byte','accuracy','targets'):assert actual[name]==expected[name],(name,actual[name],expected[name])
    training,train_checksum=stream(args.data,protocol['targets'],protocol['batch'])
    assert train_checksum==protocol['data_sha256']
    counts=torch.bincount(training[:,1:].reshape(-1),minlength=256).double()
    probabilities=(counts+1)/(counts.sum()+256)
    labels=validation[:,1:]
    majority=int(counts.argmax())
    unigram={'smoothing':1.,'training_targets':protocol['targets'],
             'nll':float(-probabilities[labels].log().mean()),
             'majority_byte':majority,'accuracy':float((labels==majority).double().mean())}
    # Keep each source stream intact but pair it with another stream's targets.
    # This preserves the input marginal and exposes dependence on the actual
    # preceding observations rather than only the overall byte distribution.
    old_training=model.training;model.eval()
    permuted=validation.roll(1,0);swap_state=None;swap_sum=0.;predictions=torch.zeros(256,dtype=torch.long)
    with torch.no_grad():
        for cursor in range(0,validation.shape[1]-1,protocol['context']):
            length=min(protocol['context'],validation.shape[1]-1-cursor)
            out=model(permuted[:,cursor:cursor+length],targets=validation[:,cursor+1:cursor+length+1],state=swap_state)
            swap_state=out['state'].detach(compact=True)
            swap_sum+=float(out['ce'])*out['logits'].shape[0]*length
            predictions+=torch.bincount(out['logits'].argmax(-1).flatten(),minlength=256)
    model.train(old_training)
    swapped={'nll':swap_sum/protocol['validation_targets'],
             'nll_increase':swap_sum/protocol['validation_targets']-actual['nll'],
             'predicted_byte_counts':{str(i):int(n) for i,n in enumerate(predictions) if n}}
    counts=[r['trained_targets'] for r in result['history']]
    assert all(b>a for a,b in zip([0]+counts[:-1],counts)) and counts[-1]==protocol['targets']
    assert all(math.isfinite(r[k]) for r in result['history'] for k in ('nll','gradient_norm'))
    del model,state,checkpoint
    initial,_,_,metadata=load(args.initial)
    initial.set_execution_backend('cpu').set_cpu_workers(protocol['workers'])
    assert file_sha(args.initial)==result['initial_checkpoint']['sha256']
    initial_ports=diagnostics(initial,validation,protocol['context'])
    report={'complete':True,'trained_targets':result['trained_targets'],
            'lifetime_trained_targets':result['lifetime_trained_targets'],'updates':result['updates'],
            'reload_exact':True,'all_parameters_and_logged_gradients_finite':True,
            'source_sha256':source_digest(),'checkpoint_sha256':file_sha(args.run/'last.pt'),
            'best_checkpoint_sha256':file_sha(args.run/'best.pt'),'reloaded_validation':actual,
            'unigram_baseline':unigram,'swapped_input_probe':swapped,
            'aligned_predicted_byte_counts':{str(i):int(n) for i,n in enumerate(aligned_counts) if n},
            'initial_diagnostics':initial_ports,'final_diagnostics':result['diagnostics']}
    atomic_json(args.out,report)
    print(json.dumps({k:report[k] for k in ('complete','trained_targets','lifetime_trained_targets','updates','reload_exact')}),flush=True)


if __name__=='__main__':main()
