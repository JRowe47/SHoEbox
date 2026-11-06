"""Post-test validation coverage and batch-invariance diagnostic; no reselection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.text_regions.joint_text import load
from experiments.text_regions.multifact_eval import evaluate
from audit_multifact import same


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    model,_,_,saved=load(a.run/'last.pt');seed=saved['protocol']['seed']
    final_row=next(r for r in reversed(saved['history']) if 'memory_validation' in r)
    weakest=min(final_row['memory_validation'],key=lambda r:min(f['accuracy'] for f in r['fields']))
    kind=weakest['kind'];count=saved['protocol']['validation_count']
    repeated=evaluate(model,'validation',seed,count,16,kind)
    same(repeated,weakest)
    expanded=evaluate(model,'validation',seed,512,16,kind,batch_size=512)
    test=evaluate(model,'test',seed,count,16,kind,batch_size=count)
    path=a.run/'evaluation.json'
    if not path.exists():path=a.run/'evaluation.partial.json'
    stored=json.loads(path.read_text())['final']['memory'][kind]
    validation_same=all(f['predictions']==e['predictions'][:count] and
        f['targets']==e['targets'][:count] for f,e in zip(repeated['fields'],expanded['fields']))
    test_same=all(f['predictions']==e['predictions'][:count] and
        f['targets']==e['targets'][:count] for f,e in zip(test['fields'],stored['fields']))
    result=dict(schema='shoebox-post-test-validation-diagnostic-v1',seed=seed,step=saved['step'],
        chosen_after_observing_test_results=True,affects_selection=False,
        selection_rule='Weakest of the five final validation task minima; earliest tie',
        kind=kind,checkpoint_sha256=hashlib.sha256((a.run/'last.pt').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        saved_validation_exact=True,validation_batch_predictions_exact=validation_same,
        test_batch_predictions_exact=test_same,original=repeated,expanded=expanded,test_subset=test)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('original','expanded','test_subset')},indent=2))
    print(json.dumps(dict(original=[f['accuracy'] for f in repeated['fields']],
        expanded=[f['accuracy'] for f in expanded['fields']],test=[f['accuracy'] for f in test['fields']])))
    if not validation_same or not test_same:raise SystemExit('Batch-size dependence detected; inspect predictions.')


if __name__=='__main__':main()
