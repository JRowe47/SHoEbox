"""Audit the matched runs and saved prediction counts without refitting."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from experiments.text_regions.composition import answer_query, encode_sources
from experiments.text_regions.composition_data import batch_from_keys, evaluation_keys
from experiments.text_regions.composition_train import load_checkpoint, source_hashes
from experiments.text_regions.model import RegionLanguageModel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('artifacts/mastery'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    runs={name:{file:torch.load(args.root/name/(file+'.pt'),weights_only=True)
                for file in ('initial','best','last')} for name in ('control','regional')}
    a,b=[runs[n]['initial'] for n in ('control','regional')]
    assert a['rng']==b['rng']
    assert all(torch.equal(v,b['weights'][k]) for k,v in a['weights'].items())
    assert set(b['weights'])-set(a['weights'])=={'regional_readout.weight','regional_readout.bias'}
    assert runs['control']['last']['rng']==runs['regional']['last']['rng']
    groups=0; summaries={}
    prior=json.loads(Path('experiments/text_regions/results/composition/reconstruction/test.json').read_text())
    assert prior['checkpoint_sha256']==hashlib.sha256((args.root/'parent.pt').read_bytes()).hexdigest()
    for name,r in runs.items():
        assert r['last']['step']==600 and len(r['last']['history'])==600
        assert r['last']['counters']['episodes']==38400
        assert r['last']['counters']['expert_executions']==10137600
        assert r['last']['protocol']['source_hashes']==source_hashes()
        candidates=[row for row in r['last']['history'] if 'validation_nats' in row]
        chosen=min(candidates,key=lambda row:row['validation_nats'])
        assert r['best']['step']==chosen['step']
        result=json.loads((args.root/name/'test.json').read_text())
        assert result['checkpoint_sha256']==hashlib.sha256((args.root/name/'best.pt').read_bytes()).hexdigest()
        for key,row in result['controls'].items():
            assert row['episodes']==len(row['predictions'])==len(row['targets'])==592
            assert row['correct']==sum(x==y for x,y in zip(row['predictions'],row['targets']))
            assert row['accuracy']==row['correct']/row['episodes']
            k=int(key[1])
            assert row['expert_executions']==592*(96+84*k)
            assert row['source_occurrences']==592*11
            if key in ('k1','k2','k4'):
                assert len(set(map(tuple,row['keys'])))==592
                assert row['keys']==prior['controls'][key]['keys']
            groups+=1
        for split,details in json.loads((args.root/name/'diagnostics.json').read_text())['splits'].items():
            for skill,row in details['skills'].items():
                assert row['correct']==sum(x==y for x,y in zip(row['predictions'],row['targets']))
                assert len(set(map(tuple,row['keys'])))==row['episodes']
                groups+=1
            for key,row in details['local'].items():
                if 'predictions' in row:
                    assert row['correct']==sum(x==y for x,y in zip(row['predictions'],row['targets']))
                    groups+=1
        # Remove both training-only heads; ordinary graph predictions stay exact.
        model,_,_,saved=load_checkpoint(args.root/name/'best.pt')
        inference=RegionLanguageModel(model.cfg)
        weights={k:v for k,v in saved['weights'].items()
                 if not k.startswith(('source_reconstruction.','regional_readout.'))}
        inference.load_state_dict(weights)
        batch=batch_from_keys(evaluation_keys('test',64))
        with torch.no_grad():
            original=answer_query(model,batch,encode_sources(model,batch)['state'],2)
            stripped=answer_query(inference,batch,encode_sources(inference,batch)['state'],2)
        torch.testing.assert_close(original['logits'],stripped['logits'],rtol=0,atol=0)
        torch.save({'schema':'shoebox-region-inference-v1','config':model.cfg.to_dict(),
                    'weights':weights,'source_checkpoint_sha256':result['checkpoint_sha256']},
                   args.root/name/'inference.pt')
        summaries[name]={'selected_step':r['best']['step'],'validation_nats':chosen['validation_nats'],
                         'test_accuracy':result['controls']['k2']['accuracy'],
                         'inference_parameters':sum(v.numel() for v in inference.parameters()),
                         'training_heads_removed_without_prediction_change':True}
    output={'common_initial_tensors':len(a['weights']),'identical_initial_graph':True,
            'identical_sampler_continuation':True,'audited_prediction_groups':groups,
            'runs':summaries}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
