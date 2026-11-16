"""Input-only shortcut baselines; true answers are used only for scoring."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from experiments.mad.tasks import (TaskSpec,dataset,fingerprint,VALUE,KEY,MAP,SEP,END,QUERY,SET,KEEP,ANSWER,PAD,TASKS)

def predictions(ep,spec):
    # Stop before the first query/output marker. No scored answer byte enters
    # any candidate's context, even in teacher-forced multi-query fixtures.
    data=list(ep.data)
    cut=next((i for i in range(2,len(data)) if data[i] in (QUERY,ANSWER)),len(data))
    prefix=data[:cut]
    supplied=[]
    if spec.task in ("recall","noisy_recall","fuzzy_recall"):
        i=2
        while i<len(prefix):
            if prefix[i]==MAP:
                j=prefix.index(SEP,i+1);k=prefix.index(END,j+1)
                supplied.extend(x-VALUE for x in prefix[j+1:k])
                i=k+1
            else:i+=1
    elif spec.task=="selective_copy":
        supplied=[prefix[i+1]-VALUE for i,v in enumerate(prefix[:-1]) if v==KEEP]
    elif spec.task=="compression":
        supplied=[x-VALUE for x in prefix[2:] if VALUE<=x<VALUE+spec.vocab]
    n=sum(ep.mask) if spec.task!="compression" else spec.length
    result={}
    if supplied:
        counts=Counter(supplied)
        majority=min(counts,key=lambda x:(-counts[x],x))
        result["query_blind_majority"]=[majority]*n
        result["last_supplied_value"]=[supplied[-1]]*n
        result["uniform_supplied_value_distribution"]={str(k):v/len(supplied) for k,v in counts.items()}
    if spec.task in ("variable_tracking","composition"):
        state={}
        for i in range(2,min(len(prefix),18),4):
            assert prefix[i]==SET
            state[prefix[i+1]-KEY]=prefix[i+2]-VALUE
        q=data[data.index(QUERY)+1]-KEY
        result["initial_binding_ignore_updates"]=[state[q]]*n
    if spec.task=="pointer_chasing":
        nodes=sorted({prefix[i+1]-KEY for i,x in enumerate(prefix[:-1]) if x==MAP})
        start=data[data.index(QUERY)+1]-KEY
        choices=[x for x in nodes if x!=start]
        result["lowest_nonstart_node"]=[choices[0]]*n
        result["uniform_nonstart_node_distribution"]={str(k):1/len(choices) for k in choices}
    result["uniform_alphabet_distribution"]={str(k):1/spec.vocab for k in range(spec.vocab)}
    return result

def summarize(rows,spec):
    totals={}
    for ep in rows:
        pred=predictions(ep,spec)
        for name,values in pred.items():
            row=totals.setdefault(name,dict(correct=0.,tokens=0,exact=0.,cases=0))
            if name.endswith("_distribution"):
                probs=[values.get(str(t),0.) for t in ep.answer]
                row["correct"]+=sum(probs)
                # Independent draws for this expected-exact reference.
                row["exact"]+=float(np.prod(probs))
            else:
                assert len(values)==len(ep.answer)
                correct=[a==b for a,b in zip(values,ep.answer)]
                row["correct"]+=sum(correct);row["exact"]+=all(correct)
            row["tokens"]+=len(ep.answer);row["cases"]+=1
    return {name:dict(token_accuracy=v["correct"]/v["tokens"],exact_accuracy=v["exact"]/v["cases"]) for name,v in totals.items()}

def audit(directory):
    output=[]
    for path in Path(directory).rglob("*.json"):
        try:d=json.loads(path.read_text())
        except (ValueError,OSError):continue
        for cell in d.get("cells",[]):
            spec=TaskSpec(**cell["spec"])
            config=d["config"]
            rows=dataset(spec,config["eval_examples"],"test",seed=config["protocol"]["data_seed"],map_seed=config["protocol"]["map_seed"])
            assert fingerprint(rows)==cell["hashes"]["test"]
            output.append(dict(source=str(path),task=cell["task"],setting=cell["setting"],steps=cell["steps"],test_hash=fingerprint(rows),baselines=summarize(rows,spec)))
    return output

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--directory",type=Path,required=True);args=p.parse_args()
    result=audit(args.directory)
    (args.directory/"shortcut_baselines.json").write_text(json.dumps(result,indent=2)+"\n")
    for row in result:print("SHORTCUT_BASELINE",json.dumps(row),flush=True)
