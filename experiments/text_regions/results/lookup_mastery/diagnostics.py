"""Score lookup skills, existing local heads and retained source tables."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from experiments.text_regions.composition import encode_sources, answer_query
from experiments.text_regions.composition_curriculum import evaluate_skills
from experiments.text_regions.composition_data import batch_from_keys, evaluation_keys
from experiments.text_regions.composition_train import load_checkpoint


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--splits',nargs='+',default=['validation','test'])
    args=p.parse_args()
    torch.set_num_threads(1)
    model,_,_,saved=load_checkpoint(args.checkpoint)
    model.eval()
    results={}
    for split in args.splits:
        skills=evaluate_skills(model,split)
        local={}
        for start in range(0,len(evaluation_keys(split)),64):
            batch=batch_from_keys(evaluation_keys(split)[start:start+64])
            prefix=encode_sources(model,batch)
            query=answer_query(model,batch,prefix['state'],2,return_round_activity=True)
            if hasattr(model,'regional_readout'):
                targets={'P_first_lookup':batch.first.gather(1,batch.query-65).flatten()-65,
                         'Q_final_answer':batch.answers-65}
                values={'P_first_lookup':query['round_activity']['P']['payload'][:,0,0],
                        'Q_final_answer':query['round_activity']['Q']['payload'][:,0,1]}
                for name,target in targets.items():
                    pred=model.regional_readout(values[name]).argmax(-1)
                    row=local.setdefault(name,dict(correct=0,count=0,predictions=[],targets=[]))
                    row['correct']+=int(pred.eq(target).sum());row['count']+=batch.size
                    row['predictions'].extend(pred.tolist());row['targets'].extend(target.tolist())
            if hasattr(model,'source_reconstruction'):
                for stage,result in [('prefix',prefix),('query',query)]:
                    for node,target in [('B',batch.first),('C',batch.second)]:
                        pred=model.source_reconstruction(result['state'].ports.latest[node].payload).reshape(-1,4,4).argmax(-1)
                        correct=pred.eq(target-65)
                        row=local.setdefault(stage+'_'+node,dict(correct=0,count=0,exact_tables=0,tables=0))
                        row['correct']+=int(correct.sum());row['count']+=correct.numel()
                        row['exact_tables']+=int(correct.all(-1).sum());row['tables']+=batch.size
        for row in local.values():
            row['accuracy']=row['correct']/row['count']
            if 'tables' in row:row['table_accuracy']=row['exact_tables']/row['tables']
        results[split]={'skills':skills,'local':local}
    output=dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                checkpoint_step=saved['step'],splits=results,note='Existing heads only; no fitting.')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({split:{'skills':{k:v['accuracy'] for k,v in r['skills'].items()},
                            'local':{k:v['accuracy'] for k,v in r['local'].items()}}
                      for split,r in results.items()},indent=2))


if __name__=='__main__':main()
